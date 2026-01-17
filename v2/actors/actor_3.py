import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
from datetime import datetime
from collections import defaultdict


# ==========================================
# 1. INFERENCE ENGINE: GRU-BASED WORLD MODEL
# ==========================================

class InferenceEngine(nn.Module):
    """
    The learned Infer() function that converts raw context to latent situation.
    
    This implements temporal compression: takes raw, messy context from the world
    and maintains a "hidden state" that represents the agent's internal understanding.
    
    The hidden state IS the situation x_n - a latent representation that the Actor
    uses to propose affordance tubes without directly accessing spatial features.
    """
    def __init__(self, raw_ctx_dim, latent_dim=64):
        super().__init__()
        self.raw_ctx_dim = raw_ctx_dim
        self.latent_dim = latent_dim
        # The core Infer() mechanism: GRU for temporal compression
        self.gru = nn.GRUCell(raw_ctx_dim, latent_dim)
        # LayerNorm to keep latent values stable for the Actor
        self.ln = nn.LayerNorm(latent_dim)
        
    def forward(self, raw_ctx, h_prev):
        """
        Convert raw context to latent situation.
        
        Args:
            raw_ctx: (B, raw_ctx_dim) raw context from world (e.g., 43-dim observation)
            h_prev: (B, latent_dim) previous hidden state (situation from last step)
        
        Returns:
            h_next: (B, latent_dim) new hidden state (current situation x_n)
        """
        # GRU processes raw context and updates hidden state
        h_next = self.gru(raw_ctx, h_prev)
        # Normalize to keep latent values stable
        h_next = self.ln(h_next)
        return h_next  # This hidden state IS the situation x_n


# ==========================================
# 2. GEOMETRY UTILITIES: CATMULL-ROM SPLINE
# ==========================================

class CausalSpline:
    @staticmethod
    def interpolate(knots, sigmas, resolution=40):
        """
        knots: (B, K, 3)
        sigmas: (B, K, 1) - Learned radius at each knot
        """
        B, K, D = knots.shape
        device = knots.device
        
        # Safety check: need at least 2 knots for interpolation
        if K < 2:
            # Return just the start and end points
            start = knots[:, 0:1, :]  # (B, 1, 3)
            end = knots[:, -1:, :] if K > 1 else knots[:, 0:1, :]
            trajectory = torch.cat([start, end], dim=1)
            sigma_traj = torch.cat([sigmas[:, 0:1, :], sigmas[:, -1:, :] if K > 1 else sigmas[:, 0:1, :]], dim=1)
            return trajectory, sigma_traj
        
        # We concatenate knots and sigmas to interpolate them together
        # Combined state: [x, y, z, sigma]
        combined = torch.cat([knots, sigmas], dim=-1) # (B, K, 4)

        # Ghost knots for C-R Spline
        p0 = 2 * combined[:, 0:1] - combined[:, 1:2]
        pn = 2 * combined[:, -1:] - combined[:, -2:-1]
        padded = torch.cat([p0, combined, pn], dim=1)

        points_per_seg = max(4, int(np.ceil(resolution / max(1, K - 1))))
        t = torch.linspace(0, 1, points_per_seg, device=device)[:-1]
        t1, t2, t3 = t, t**2, t**3

        all_segments = []
        for i in range(1, K):
            P0, P1, P2, P3 = [padded[:, i+j].unsqueeze(1) for j in range(-1, 3)]
            
            seg = 0.5 * (2*P1 + (-P0 + P2)*t1.view(1,-1,1) + 
                         (2*P0 - 5*P1 + 4*P2 - P3)*t2.view(1,-1,1) + 
                         (-P0 + 3*P1 - 3*P2 + P3)*t3.view(1,-1,1))
            all_segments.append(seg)

        # Add endpoint
        all_segments.append(combined[:, -1:].view(B, 1, 4))
        full_trajectory = torch.cat(all_segments, dim=1)
        
        return full_trajectory[:, :, :3], full_trajectory[:, :, 3:]


# ==========================================
# 3. THE ACTOR: CAUSAL KNOT ENGINE (LATENT-AGNOSTIC)
# ==========================================

class Actor(nn.Module):
    """
    Actor operates on latent situations from InferenceEngine.
    
    The Actor no longer cares where its input comes from - it's agnostic to
    whether the latent representation encodes spatial features, temporal patterns,
    or any other world structure. It simply proposes affordance tubes based on
    the latent situation and intent.
    """
    def __init__(self, latent_dim, intent_dim, n_ports=4, n_knots=6, interp_res=40):
        super().__init__()
        self.n_ports = n_ports
        self.n_knots = n_knots
        self.interp_res = interp_res
        
        # Encoder: processes latent situation + intent
        # Actor doesn't perform spatial calculations - it operates in latent space
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim + intent_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256)
        )
        
        # Port Proposer
        # Output size: M * (Logit + K*3 + K*1 + K*1) where last K*1 is knot existence logits
        # Note: K*3 are now RESIDUAL OFFSETS, not absolute positions
        self.port_head = nn.Linear(256, n_ports * (1 + n_knots * 5))
        
    def forward(self, latent_situation, intent, previous_velocity=None):
        """
        Args:
            latent_situation: (B, latent_dim) latent situation from InferenceEngine
            intent: (B, 3) intent/target direction tensor
            previous_velocity: Optional (B, 3) or (3,) tensor of previous path velocity for G1 continuity
                              If provided, enforces that first knot direction matches this velocity
        
        Returns:
            logits: (B, M)
            mu_t:   (B, M, T, 3) - Interpolated Tube
            sigma_t:(B, M, T, 1) - Interpolated Radius
            knot_mask: (B, M, K) - Knot existence mask (0.0 = inactive, 1.0 = active)
            knot_steps: (B, M, K, 3) - Residual offsets (for loss calculation)
        """
        B = latent_situation.size(0)
        situation = self.encoder(torch.cat([latent_situation, intent], dim=-1))
        
        raw_out = self.port_head(situation).view(B, self.n_ports, -1)
        logits = raw_out[:, :, 0]
        
        # Extract Knot Steps (RESIDUAL OFFSETS), Sigmas, and Knot Existence Logits
        geo_params = raw_out[:, :, 1:].view(B, self.n_ports, self.n_knots, 5)
        
        # RESIDUAL KNOT LOGIC: Predict relative offsets instead of absolute positions
        # Use tanh to bound the step size of each knot (prevents erratic snapping)
        knot_steps = torch.tanh(geo_params[..., :3]) * 2.0  # (B, M, K, 3) bounded to [-2, 2]
        
        sigmas_raw = F.softplus(geo_params[..., 3:4]) + 0.1
        knot_existence_logits = geo_params[..., 4:5]  # (B, M, K, 1)
        
        # Apply sigmoid to get knot existence mask (0.0 = inactive, 1.0 = active)
        # First and last knots are always active (required for spline)
        knot_mask_raw = torch.sigmoid(knot_existence_logits).squeeze(-1)  # (B, M, K)
        # Create mask with first and last always active (non-inplace operation)
        device = knot_mask_raw.device
        ones_first = torch.ones(B, self.n_ports, 1, device=device)
        ones_last = torch.ones(B, self.n_ports, 1, device=device)
        knot_mask = torch.cat([ones_first, knot_mask_raw[:, :, 1:-1], ones_last], dim=-1)
        
        # RESIDUAL KNOT CONSTRUCTION: The spline 'grows' from the origin (0,0,0)
        # Use cumsum to make each knot relative to the one before it
        knots_rel = torch.cumsum(knot_steps, dim=2)  # (B, M, K, 3)
        # Causal Anchoring: Ensure the first knot is always exactly at the start
        knots_rel = knots_rel - knots_rel[:, :, 0:1, :]
        
        # G1 CONTINUITY: Enforce by construction (not penalty)
        # If previous_velocity is provided, ensure first knot direction matches it
        # This prevents "jagged zags" structurally rather than through loss penalties
        if previous_velocity is not None:
            # Normalize previous velocity to get direction
            if previous_velocity.dim() == 1:
                prev_vel = previous_velocity.unsqueeze(0)  # (1, 3)
            else:
                prev_vel = previous_velocity  # (B, 3) or (1, 3)
            
            # Ensure prev_vel is (B, 3) for broadcasting
            if prev_vel.size(0) == 1 and B > 1:
                prev_vel = prev_vel.expand(B, -1)
            elif prev_vel.size(0) != B:
                prev_vel = prev_vel[:B]  # Take first B if needed
            
            prev_vel_norm = prev_vel / (torch.norm(prev_vel, dim=-1, keepdim=True) + 1e-6)  # (B, 3)
            
            # Project first knot direction onto previous velocity direction
            # knots_rel[:, :, 1, :] is the second knot (first after origin)
            # We want the direction from knot 0 to knot 1 to align with prev_vel_norm
            if self.n_knots > 1:
                # Get current first segment direction
                first_segment = knots_rel[:, :, 1:2, :]  # (B, M, 1, 3)
                first_segment_norm = first_segment / (torch.norm(first_segment, dim=-1, keepdim=True) + 1e-6)
                
                # Project onto previous velocity direction
                # We want: first_segment_dir = prev_vel_norm * scale
                # So: scale = dot(first_segment_dir, prev_vel_norm)
                # Then: first_segment = prev_vel_norm * scale * magnitude
                prev_vel_expanded = prev_vel_norm.view(B, 1, 1, 3)  # (B, 1, 1, 3) broadcasts to (B, M, 1, 3)
                
                # Get magnitude of first segment
                first_segment_mag = torch.norm(first_segment, dim=-1, keepdim=True)  # (B, M, 1, 1)
                
                # Align direction: use previous velocity direction, preserve magnitude
                # Blend: 80% previous velocity direction, 20% network's learned direction
                # This allows the network to learn while enforcing continuity
                aligned_direction = 0.8 * prev_vel_expanded + 0.2 * first_segment_norm
                aligned_direction = aligned_direction / (torch.norm(aligned_direction, dim=-1, keepdim=True) + 1e-6)
                aligned_first_segment = aligned_direction * first_segment_mag
                
                # Replace first segment with aligned version
                knots_rel = torch.cat([knots_rel[:, :, :1, :], aligned_first_segment, knots_rel[:, :, 2:, :]], dim=2)
        
        # INTENT BIAS: Push the last knot toward the intent
        # This ensures that even with random weights, tubes are biased toward the goal
        # Reshape intent to (B, 1, 3) to broadcast across ports (M dimension)
        intent_bias = intent.view(B, 1, 3)  # (B, 1, 3) broadcasts to (B, M, 3)
        # Add a fraction of intent to the last knot to guide the network (non-inplace)
        # Using 0.5 means the network learns to add the other 0.5 through training
        knots_rel_last = knots_rel[:, :, -1:, :] + intent_bias.unsqueeze(2) * 0.5
        knots_rel = torch.cat([knots_rel[:, :, :-1, :], knots_rel_last], dim=2)
        
        # Filter inactive knots: keep only knots with mask > 0.5
        # This creates variable-length knot sequences per port
        mu_dense_list = []
        sigma_dense_list = []
        
        for b in range(B):
            for m in range(self.n_ports):
                # Get active knots for this port
                active_mask = knot_mask[b, m] > 0.5  # (K,)
                active_indices = torch.where(active_mask)[0]
                device = active_indices.device if len(active_indices) > 0 else knots_rel.device
                
                # Ensure at least first and last knots are included
                if len(active_indices) == 0:
                    # If no knots active, use first and last
                    active_indices = torch.tensor([0, self.n_knots - 1], device=device)
                else:
                    # Ensure first knot is included
                    if not torch.any(active_indices == 0):
                        active_indices = torch.cat([torch.tensor([0], device=device), active_indices])
                    # Ensure last knot is included
                    if not torch.any(active_indices == self.n_knots - 1):
                        active_indices = torch.cat([active_indices, torch.tensor([self.n_knots - 1], device=device)])
                    # Sort and remove duplicates
                    active_indices = torch.unique(torch.sort(active_indices)[0])
                
                # Extract active knots and sigmas
                active_knots = knots_rel[b, m, active_indices, :].unsqueeze(0)  # (1, K_active, 3)
                active_sigmas = sigmas_raw[b, m, active_indices, :].unsqueeze(0)  # (1, K_active, 1)
                
                # Interpolate with filtered knots
                mu_dense_port, sigma_dense_port = CausalSpline.interpolate(
                    active_knots, active_sigmas, resolution=self.interp_res
                )
                
                mu_dense_list.append(mu_dense_port.squeeze(0))  # (T, 3)
                sigma_dense_list.append(sigma_dense_port.squeeze(0))  # (T, 1)
        
        # Pad to same length for batching (use max length)
        max_T = max(m.shape[0] for m in mu_dense_list)
        device = mu_dense_list[0].device
        
        mu_t_padded = []
        sigma_t_padded = []
        for mu, sigma in zip(mu_dense_list, sigma_dense_list):
            T = mu.shape[0]
            if T < max_T:
                # Pad with last value
                pad_mu = mu[-1:].repeat(max_T - T, 1)
                pad_sigma = sigma[-1:].repeat(max_T - T, 1)
                mu_padded = torch.cat([mu, pad_mu], dim=0)
                sigma_padded = torch.cat([sigma, pad_sigma], dim=0)
            else:
                mu_padded = mu
                sigma_padded = sigma
            mu_t_padded.append(mu_padded)
            sigma_t_padded.append(sigma_padded)
        
        # Stack and reshape
        mu_t = torch.stack(mu_t_padded).view(B, self.n_ports, max_T, 3)
        sigma_t = torch.stack(sigma_t_padded).view(B, self.n_ports, max_T, 1)
        
        return logits, mu_t, sigma_t, knot_mask, knot_steps

    def select_port(self, logits, mu_t, intent_target):
        """Select port based on closest tube endpoint to target, weighted by logits."""
        end_points = mu_t[:, :, -1, :] 
        dist = torch.norm(end_points - intent_target.unsqueeze(1), dim=-1)
        score = F.log_softmax(logits, dim=-1) - dist
        return torch.argmax(score, dim=-1)


# ==========================================
# 4. SIMULATION WRAPPER (FIXED)
# ==========================================

class SimulationWrapper:
    def __init__(self, obstacles, bounds=None):
        """
        Args:
            obstacles: List of (position, radius) tuples
            bounds: Optional dict with 'min' and 'max' keys, each a list of 3 floats [x, y, z]
                   If None, defaults to [-2, -2, -2] to [12, 12, 2]
        """
        self.obstacles = obstacles 
        
        # Set default bounds if not provided
        if bounds is None:
            self.bounds = {
                'min': [-2.0, -2.0, -2.0],
                'max': [12.0, 12.0, 2.0]
            }
        else:
            self.bounds = bounds
        
        # Convert to tensors for easier computation
        self.bounds_min = torch.tensor(self.bounds['min'], dtype=torch.float32)
        self.bounds_max = torch.tensor(self.bounds['max'], dtype=torch.float32)
    
    def get_raw_observation(self, current_pos, target_pos, max_obstacles=10):
        """
        Get RAW context observation (for InferenceEngine).
        
        This is the raw, messy context from the world that the InferenceEngine
        will compress into a latent situation. The Actor never sees this directly.
        
        Args:
            current_pos: (1, 3) or (3,) current position tensor
            target_pos: (3,) or (1, 3) target position tensor
            max_obstacles: Maximum number of obstacles to include (default: 10)
                          Observation is always this size, padded with zeros if fewer obstacles
        
        Returns:
            raw_ctx: torch.Tensor of shape (raw_ctx_dim,)
            raw_ctx_dim = 3 (rel_goal) + max_obstacles * 4 (rel_obs_pos + obs_radius)
        """
        # Ensure current_pos is (3,)
        if current_pos.dim() > 1:
            current_pos_flat = current_pos.squeeze(0)
        else:
            current_pos_flat = current_pos
        
        # Ensure target_pos is (3,)
        if target_pos.dim() > 1:
            target_pos_flat = target_pos.squeeze(0)
        else:
            target_pos_flat = target_pos
        
        # Relative goal vector
        rel_goal = target_pos_flat - current_pos_flat  # (3,)
        
        # Get obstacle context: relative positions and radii
        # Sort obstacles by distance to current position (nearest first)
        device = current_pos_flat.device if hasattr(current_pos_flat, 'device') else None
        obstacle_info = []
        for obs_p, obs_r in self.obstacles:
            obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=device)
            rel_obs_pos = obs_pos - current_pos_flat  # (3,)
            obs_radius = torch.tensor([obs_r], dtype=torch.float32, device=device)  # (1,)
            distance = torch.norm(rel_obs_pos)
            obstacle_info.append((distance.item(), rel_obs_pos, obs_radius))
        
        # Sort by distance and take nearest max_obstacles
        obstacle_info.sort(key=lambda x: x[0])
        obstacle_info = obstacle_info[:max_obstacles]
        
        # Build obstacle features: [rel_pos_x, rel_pos_y, rel_pos_z, radius] for each obstacle
        obs_parts = [rel_goal]
        for _, rel_obs_pos, obs_radius in obstacle_info:
            obs_parts.append(rel_obs_pos)  # (3,)
            obs_parts.append(obs_radius)   # (1,)
        
        # Pad with zeros if we have fewer than max_obstacles
        # The actor can learn to ignore zero-padded obstacles (radius=0 means no obstacle)
        # This decouples observation from exact obstacle count - add/remove obstacles freely!
        num_obstacles_present = len(obstacle_info)
        if num_obstacles_present < max_obstacles:
            # Pad with zero vectors: [0, 0, 0, 0] for each missing obstacle
            zeros_3d = torch.zeros(3, dtype=torch.float32, device=device)
            zeros_1d = torch.zeros(1, dtype=torch.float32, device=device)
            for _ in range(max_obstacles - num_obstacles_present):
                obs_parts.append(zeros_3d)  # Zero relative position
                obs_parts.append(zeros_1d)  # Zero radius (indicates no obstacle - actor learns to ignore)
        
        # Concatenate all parts
        raw_ctx = torch.cat(obs_parts, dim=0)  # (3 + max_obstacles*4,)
        
        return raw_ctx
    
    def execute(self, mu_t, sigma_t, current_pos):
        """
        Execute a tube in the world, checking for obstacles and binding violations.
        
        PHILOSOPHY: All physics (obstacles, boundaries, etc.) creates deviation from
        the planned tube. This deviation surfaces through binding failure:
        - Exception: binding failed (deviation > sigma)
        - Magnitude: deviation distance
        - Position: where the violation occurred (actual_p after physics)
        
        Args:
            mu_t: (T, 3) relative tube trajectory
            sigma_t: (T, 1) tube radii
            current_pos: (1, 3) current position
        
        Returns:
            actual_p: (T, 3) actual path taken
        """
        # Ensure mu_t and sigma_t are properly shaped - should be (T, 3) and (T, 1) or (T,)
        if mu_t.dim() > 2:
            mu_t = mu_t.view(-1, 3)
        elif mu_t.dim() == 1:
            mu_t = mu_t.unsqueeze(0)
        
        if sigma_t.dim() > 2:
            sigma_t = sigma_t.view(-1)
        elif sigma_t.dim() == 2:
            sigma_t = sigma_t.squeeze(-1) if sigma_t.shape[1] == 1 else sigma_t[:, 0]
        elif sigma_t.dim() == 0:
            sigma_t = sigma_t.unsqueeze(0)
        
        # Transform relative tube to global coordinates
        mu_global = mu_t + current_pos
        
        actual_path = []
        actual_path.append(current_pos.squeeze().clone())
        
        # If tube is too short (only 1 point), return current position
        if len(mu_global) <= 1:
            return torch.stack(actual_path)
        
        # Execute step by step
        for t in range(1, len(mu_global)):
            expected_p = mu_global[t]
            
            # Handle sigma_t indexing - ensure we get a scalar
            t_idx = min(t, len(sigma_t) - 1)
            affordance_r = sigma_t[t_idx].item() if hasattr(sigma_t[t_idx], 'item') else float(sigma_t[t_idx])
            
            # Start with expected position
            actual_p = expected_p.clone()
            
            # Check for obstacles (physics that creates deviation)
            for obs_p, obs_r in self.obstacles:
                obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=actual_p.device)
                d = torch.norm(actual_p - obs_pos)
                if d < obs_r:
                    # Collision: push away from obstacle (creates deviation from expected)
                    vec = actual_p - obs_pos
                    if d > 1e-6:
                        actual_p = obs_pos + (vec / d) * obs_r
                    else:
                        # If exactly on obstacle, push in a random direction
                        actual_p = obs_pos + torch.tensor([obs_r, 0, 0], dtype=torch.float32, device=actual_p.device)
            
            # Check boundaries (physics that creates deviation - same as obstacles)
            # Boundaries should push back, creating deviation that can trigger binding failure
            device = actual_p.device
            bounds_min = self.bounds_min.to(device)
            bounds_max = self.bounds_max.to(device)
            
            # Check each dimension for boundary violations
            for dim in range(3):
                if actual_p[dim] < bounds_min[dim]:
                    # Violated lower bound: push to boundary (creates deviation)
                    actual_p[dim] = bounds_min[dim]
                elif actual_p[dim] > bounds_max[dim]:
                    # Violated upper bound: push to boundary (creates deviation)
                    actual_p[dim] = bounds_max[dim]
            
            # Binding check: if deviation exceeds sigma, stop early
            # This captures ALL physics deviations: obstacles, boundaries, etc.
            # Exception: binding failed
            # Magnitude: deviation distance
            # Position: where the violation occurred (actual_p)
            deviation = torch.norm(actual_p - expected_p)
            if deviation > affordance_r:
                # Binding broken - physics (obstacle/boundary) created too much deviation
                # Record the exception: position where binding failed
                actual_path.append(actual_p.clone())
                break
            
            actual_path.append(actual_p.clone())
        
        return torch.stack(actual_path) 
        
    def run_episode(self, inference_engine, actor, start_pos, target_pos, max_observed_obstacles=10, latent_dim=64):
        """
        Run a full episode using the InferenceEngine and Actor for evaluation.
        
        Args:
            inference_engine: The InferenceEngine model to use
            actor: The Actor model to use
            start_pos: Starting position [x, y, z]
            target_pos: Target position [x, y, z]
            max_observed_obstacles: Maximum obstacles in observation
            latent_dim: Dimension of latent situation space
        
        Returns:
            path: numpy array of actual path taken
        """
        current_pos = torch.tensor(start_pos, dtype=torch.float32).view(1, 3)
        target = torch.tensor(target_pos, dtype=torch.float32).view(1, 3)
        
        path_history = [current_pos.squeeze().numpy()]
        steps = 0
        max_steps = 40
        
        # Reset InferenceEngine hidden state at start of episode
        h_state = torch.zeros(1, latent_dim)
        
        while steps < max_steps:
            rel_goal = target - current_pos
            
            # Get raw observation (for InferenceEngine)
            raw_obs = self.get_raw_observation(current_pos, target, max_obstacles=max_observed_obstacles)  # (raw_ctx_dim,)
            raw_obs = raw_obs.unsqueeze(0)  # (1, raw_ctx_dim) for batch dimension
            
            with torch.no_grad():
                # INFER: Convert raw context to latent situation
                x_n = inference_engine(raw_obs, h_state)  # (1, latent_dim)
                h_state = x_n  # Update hidden state
                
                # ACT: Propose affordances based on latent situation
                logits, mu_t, sigma_t, knot_mask, _ = actor(x_n, rel_goal, previous_velocity=None)
                
                best_idx = actor.select_port(logits, mu_t, rel_goal).item()
                
                # Get the committed tube (T, 3)
                # We add current_pos to transform from relative -> global
                committed_mu = mu_t[0, best_idx] + current_pos
                committed_sigma = sigma_t[0, best_idx]

            pts = len(committed_mu)
            
            if pts < 5:
                break

            # EXECUTE COMMITMENT (Bind -> Contradict)
            # Start from t=1 because t=0 is our current position
            for t in range(1, pts):
                expected_p = committed_mu[t]
                affordance_r = committed_sigma[t]
                
                # Move
                actual_p = expected_p.clone()
                
                # Simple Physics: Check for Obstacles (creates deviation)
                for obs_p, obs_r in self.obstacles:
                    d = torch.norm(actual_p - torch.tensor(obs_p))
                    if d < obs_r:
                        # Simple bounce/block (pushes back, creating deviation)
                        vec = actual_p - torch.tensor(obs_p)
                        actual_p = torch.tensor(obs_p) + (vec / d) * obs_r
                
                # Check boundaries (physics that creates deviation - same as obstacles)
                # Boundaries push back, creating deviation that can trigger binding failure
                # Convert bounds to same format as obstacles (tensors)
                bounds_min_t = torch.tensor(self.bounds['min'], dtype=torch.float32)
                bounds_max_t = torch.tensor(self.bounds['max'], dtype=torch.float32)
                
                # Check each dimension for boundary violations
                for dim in range(3):
                    if actual_p[dim] < bounds_min_t[dim]:
                        # Violated lower bound: push to boundary (creates deviation)
                        actual_p[dim] = bounds_min_t[dim]
                    elif actual_p[dim] > bounds_max_t[dim]:
                        # Violated upper bound: push to boundary (creates deviation)
                        actual_p[dim] = bounds_max_t[dim]
                
                # BINDING CHECK
                # If reality (actual_p) is too far from plan (expected_p)
                # This captures ALL physics deviations: obstacles, boundaries, etc.
                # Exception: binding failed
                # Magnitude: deviation distance
                # Position: where the violation occurred (actual_p)
                deviation = torch.norm(actual_p - expected_p)
                if deviation > affordance_r:
                    # Update state to where we actually ended up
                    # This is the exception position - where binding failed
                    current_pos = actual_p.view(1, 3)
                    path_history.append(current_pos.squeeze().numpy())
                    break  # Break inner loop to Re-plan
                
                # If valid, commit to this step
                current_pos = actual_p.view(1, 3)
                path_history.append(current_pos.squeeze().numpy())
                
                if torch.norm(current_pos - target) < 1.0:
                    return np.array(path_history)

            steps += 1

        return np.array(path_history)

# ==========================================
# 5. TRAINING & LIVE PLOTTING
# ==========================================

class LivePlotter:
    def __init__(self, obstacles, target_pos, bounds=None):
        """
        Args:
            obstacles: List of (position, radius) tuples
            target_pos: Initial target position
            bounds: Optional dict with 'min' and 'max' keys for bounding box
        """
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.obstacles = obstacles
        self.target_pos = target_pos
        
        # Set default bounds if not provided
        if bounds is None:
            self.bounds = {
                'min': [-2.0, -2.0, -2.0],
                'max': [12.0, 12.0, 2.0]
            }
        else:
            self.bounds = bounds
        
        # Plot bounding box as wireframe
        self._plot_bounding_box()
        
        # Plot obstacles
        for obs_p, obs_r in obstacles:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = obs_p[0] + obs_r * np.outer(np.cos(u), np.sin(v))
            y = obs_p[1] + obs_r * np.outer(np.sin(u), np.sin(v))
            z = obs_p[2] + obs_r * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color='red', alpha=0.3, edgecolor='none')
        
        # Plot target (store marker for updates)
        self.target_marker = self.ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
                       color='green', s=300, marker='*', edgecolors='darkgreen',
                       linewidths=2, label='Target', zorder=10)
        
        # Plot start
        self.ax.scatter(0, 0, 0, color='black', s=100, label='Start')
        
        # Storage for plot elements (only current episode)
        self.tube_lines = []
        self.path_lines = []
        self.tube_spheres = []  # Store sphere surfaces for cleanup
        self.tube_markers = []  # Store tube start/end markers
        self.path_markers = []  # Store path endpoint markers
        self.current_pos_marker = None
        self.current_episode = -1  # Track current episode to detect transitions
        self.reached_goal_markers = []  # Store markers for goals reached in current episode
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Live Training: Tube Warping Visualization')
        self.ax.legend()
        self.ax.set_xlim(self.bounds['min'][0], self.bounds['max'][0])
        self.ax.set_ylim(self.bounds['min'][1], self.bounds['max'][1])
        self.ax.set_zlim(self.bounds['min'][2], self.bounds['max'][2])
    
    def _plot_bounding_box(self):
        """Plot the bounding box as a wireframe cube."""
        min_b = self.bounds['min']
        max_b = self.bounds['max']
        
        # Define the 8 vertices of the bounding box
        vertices = np.array([
            [min_b[0], min_b[1], min_b[2]],  # 0: min corner
            [max_b[0], min_b[1], min_b[2]],  # 1
            [max_b[0], max_b[1], min_b[2]],  # 2
            [min_b[0], max_b[1], min_b[2]],  # 3
            [min_b[0], min_b[1], max_b[2]],  # 4
            [max_b[0], min_b[1], max_b[2]],  # 5
            [max_b[0], max_b[1], max_b[2]],  # 6
            [min_b[0], max_b[1], max_b[2]],  # 7: max corner
        ])
        
        # Define the 12 edges of the cube
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # Vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            points = vertices[edge]
            self.ax.plot3D(*points.T, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Boundary' if edge == edges[0] else '')
        
    def update(self, mu_t, sigma_t, actual_path, current_pos, episode, step):
        """
        Update the plot with new tube and path information.
        
        Args:
            mu_t: (T, 3) planned tube trajectory (relative)
            sigma_t: (T, 1) tube radii
            actual_path: (T, 3) actual path taken
            current_pos: (1, 3) current position
            episode: current episode number
            step: current step number
        """
        # Clear previous episode's visualization when starting a new episode
        if episode != self.current_episode:
            self._clear_episode()
            self.current_episode = episode
        
        # Keep only last few steps for clarity within current episode
        max_history = 10
        if len(self.tube_lines) > max_history:
            for line in self.tube_lines[:-max_history]:
                line.remove()
            self.tube_lines = self.tube_lines[-max_history:]
        
        if len(self.path_lines) > max_history:
            for line in self.path_lines[:-max_history]:
                line.remove()
            self.path_lines = self.path_lines[-max_history:]
        
        # Clean up old markers and spheres
        if len(self.tube_markers) > max_history * 2:  # 2 markers per tube (start/end)
            for marker in self.tube_markers[:-max_history * 2]:
                marker.remove()
            self.tube_markers = self.tube_markers[-max_history * 2:]
        
        if len(self.path_markers) > max_history * 2:
            for marker in self.path_markers[:-max_history * 2]:
                marker.remove()
            self.path_markers = self.path_markers[-max_history * 2:]
        
        if len(self.tube_spheres) > max_history * 5:  # ~5 spheres per tube
            for sphere in self.tube_spheres[:-max_history * 5]:
                sphere.remove()
            self.tube_spheres = self.tube_spheres[-max_history * 5:]
        
        # Transform tube to global coordinates
        mu_global = mu_t + current_pos
        
        # Plot planned tube with radius visualization
        mu_np = mu_global.detach().cpu().numpy()
        sigma_np = sigma_t.detach().cpu().numpy()
        
        # Ensure mu_np is 2D (T, 3)
        if mu_np.ndim > 2:
            mu_np = mu_np.reshape(-1, 3)
        elif mu_np.ndim == 1:
            mu_np = mu_np.reshape(1, -1)
        
        # Ensure sigma_np is 1D
        if sigma_np.ndim > 1:
            sigma_np = sigma_np.squeeze()
        if sigma_np.ndim == 0:
            sigma_np = np.array([sigma_np.item()])
        
        # Plot tube centerline
        tube_line, = self.ax.plot(mu_np[:, 0], mu_np[:, 1], mu_np[:, 2], 
                                  color='cyan', alpha=0.7, linestyle='--', 
                                  linewidth=2.5, label='Planned Tube' if len(self.tube_lines) == 0 else '')
        self.tube_lines.append(tube_line)
        
        # Plot starting point of tube
        start_point = mu_np[0]
        start_marker = self.ax.scatter(start_point[0], start_point[1], start_point[2], 
                                      color='cyan', s=30, alpha=0.6, marker='o')
        self.tube_markers.append(start_marker)
        
        # Plot end point of tube
        end_point = mu_np[-1]
        end_marker = self.ax.scatter(end_point[0], end_point[1], end_point[2], 
                                     color='cyan', s=50, alpha=0.8, marker='x', linewidths=2)
        self.tube_markers.append(end_marker)
        
        # Visualize tube radius at sample points (simplified - just show spheres at key points)
        # Skip sphere visualization if arrays have unexpected shapes to avoid errors
        try:
            sample_indices = np.linspace(0, len(mu_np)-1, min(5, len(mu_np)), dtype=int)
            for idx in sample_indices:
                idx_int = int(idx)
                # Extract coordinates - ensure we get scalars by using flat indexing
                cx = float(mu_np.flat[idx_int * 3 + 0])
                cy = float(mu_np.flat[idx_int * 3 + 1])
                cz = float(mu_np.flat[idx_int * 3 + 2])
                
                # Extract radius
                if sigma_np.ndim == 0:
                    radius = float(sigma_np)
                else:
                    radius = float(sigma_np.flat[idx_int] if sigma_np.size > idx_int else sigma_np.flat[0])
                
                # Draw a small sphere to represent tube radius
                u = np.linspace(0, 2 * np.pi, 8)
                v = np.linspace(0, np.pi, 8)
                u_grid, v_grid = np.meshgrid(u, v)
                
                x = cx + radius * np.sin(v_grid) * np.cos(u_grid)
                y = cy + radius * np.sin(v_grid) * np.sin(u_grid)
                z = cz + radius * np.cos(v_grid)
                
                sphere_surface = self.ax.plot_surface(x, y, z, color='cyan', alpha=0.15, edgecolor='none', linewidth=0)
                self.tube_spheres.append(sphere_surface)
        except (TypeError, ValueError, AttributeError, IndexError) as e:
            # If sphere visualization fails, just skip it - tube centerline is still shown
            pass
        
        # Plot actual path taken
        if actual_path is not None and len(actual_path) > 0:
            path_np = actual_path.detach().cpu().numpy() if isinstance(actual_path, torch.Tensor) else actual_path
            
            # Ensure path_np is 2D
            if path_np.ndim == 1:
                path_np = path_np.reshape(1, -1)
            
            # Plot this step's actual path
            if len(path_np) > 1:
                path_line, = self.ax.plot(path_np[:, 0], path_np[:, 1], path_np[:, 2],
                                         color='blue', linewidth=3, alpha=0.8,
                                         label='Actual Path' if len(self.path_lines) == 0 else '')
                self.path_lines.append(path_line)
                
                # Plot path endpoints
                start_marker = self.ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2],
                                              color='blue', s=40, alpha=0.7, marker='s')
                self.path_markers.append(start_marker)
                
                end_marker = self.ax.scatter(path_np[-1, 0], path_np[-1, 1], path_np[-1, 2],
                                            color='blue', s=60, alpha=0.9, marker='D')
                self.path_markers.append(end_marker)
            else:
                # Single point path - just plot as a marker
                single_marker = self.ax.scatter(path_np[0, 0], path_np[0, 1], path_np[0, 2],
                                               color='blue', s=80, alpha=0.8, marker='o')
                self.path_markers.append(single_marker)
        
        # Update current position marker (large, visible)
        if self.current_pos_marker is not None:
            self.current_pos_marker.remove()
        
        pos_np = current_pos.squeeze().detach().cpu().numpy() if isinstance(current_pos, torch.Tensor) else current_pos.squeeze()
        self.current_pos_marker = self.ax.scatter(pos_np[0], pos_np[1], pos_np[2],
                                                  color='orange', s=200, marker='o', 
                                                  edgecolors='red', linewidths=3,
                                                  label='Current Position', zorder=10)
        
        # Update title with episode/step info and key metrics
        avg_sigma = float(sigma_np.mean()) if len(sigma_np) > 0 else 0.0
        goals_reached_count = len(self.reached_goal_markers)
        self.ax.set_title(f'Episode {episode}, Step {step} | Avg σ: {avg_sigma:.3f} | Goals Reached: {goals_reached_count}')
        
        # Refresh plot with longer pause for visibility
        plt.draw()
        plt.pause(0.05)  # Increased pause time for better visibility
    
    def _clear_episode(self):
        """Clear all visualization elements from the previous episode."""
        # Remove all tube lines
        for line in self.tube_lines:
            line.remove()
        self.tube_lines = []
        
        # Remove all path lines
        for line in self.path_lines:
            line.remove()
        self.path_lines = []
        
        # Remove all markers
        for marker in self.tube_markers:
            marker.remove()
        self.tube_markers = []
        
        for marker in self.path_markers:
            marker.remove()
        self.path_markers = []
        
        # Remove all sphere surfaces
        for sphere in self.tube_spheres:
            sphere.remove()
        self.tube_spheres = []
        
        # Remove current position marker
        if self.current_pos_marker is not None:
            self.current_pos_marker.remove()
            self.current_pos_marker = None
        
        # Remove reached goal markers
        for marker in self.reached_goal_markers:
            marker.remove()
        self.reached_goal_markers = []
    
    def mark_goal_reached(self, goal_pos):
        """
        Mark a goal as reached and add it to the visualization.
        
        Args:
            goal_pos: (3,) numpy array or tensor with goal position
        """
        # Convert to numpy if needed
        if isinstance(goal_pos, torch.Tensor):
            goal_np = goal_pos.detach().cpu().numpy() if goal_pos.requires_grad else goal_pos.cpu().numpy()
        else:
            goal_np = np.array(goal_pos)
        
        # Ensure it's 1D
        if goal_np.ndim > 1:
            goal_np = goal_np.squeeze()
        
        # Add a marker for the reached goal (different color/style from current target)
        reached_marker = self.ax.scatter(
            goal_np[0], goal_np[1], goal_np[2],
            color='gold', s=200, marker='*', edgecolors='orange',
            linewidths=2, alpha=0.8, label='Reached Goal' if len(self.reached_goal_markers) == 0 else '',
            zorder=9
        )
        self.reached_goal_markers.append(reached_marker)
        self.ax.legend()
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)


def train_actor(inference_engine, actor, episodes=50, plot_live=True, max_observed_obstacles=10, latent_dim=64):
    """
    Train the InferenceEngine and Actor together in the environment.
    
    This implements dual-optimization: both models learn simultaneously.
    - InferenceEngine learns to represent context such that Actor can successfully bind
    - Actor learns to use that representation to propose tubes
    
    Args:
        inference_engine: InferenceEngine model (GRU-based World Model)
        actor: Actor model (operates on latent situations)
        episodes: Number of training episodes
        plot_live: Whether to show live visualization
        max_observed_obstacles: Maximum number of obstacles in observation (fixed size)
        latent_dim: Dimension of latent situation space
    """
    # Combine parameters for joint optimization
    optimizer = optim.Adam(
        list(inference_engine.parameters()) + list(actor.parameters()), 
        lr=1e-3
    )
    # Multiple obstacles creating a more challenging navigation environment
    # You can freely add/remove obstacles here - the observation will pad/truncate automatically
    obstacles = [
        ([5.0, 5.0, 0.0], 2.5),      # Central obstacle
        ([3.0, 8.0, 0.0], 1.5),      # Upper left
        # ([8.0, 3.0, 0.0], 1.5),      # Lower right
        # ([7.0, 7.0, 0.0], 1.8),      # Upper right cluster
        # ([2.0, 2.0, 0.0], 1.2),      # Lower left
        # ([9.0, 6.0, 0.0], 1.3),      # Right side
        # ([4.0, 9.0, 0.0], 1.4),      # Top area
    ]
    
    # Define environment boundaries
    bounds = {
        'min': [-2.0, -2.0, -2.0],
        'max': [12.0, 12.0, 2.0]
    }
    
    sim = SimulationWrapper(obstacles, bounds=bounds)
    target_pos = torch.tensor([10.0, 10.0, 0.0])
    
    # Calculate raw context dimension: 3 (rel_goal) + max_observed_obstacles * 4
    # This is FIXED regardless of actual obstacle count (uses padding/truncation)
    raw_ctx_dim = 3 + max_observed_obstacles * 4
    
    # Verify models are compatible
    if inference_engine.raw_ctx_dim != raw_ctx_dim:
        raise ValueError(f"InferenceEngine raw_ctx_dim mismatch: expected {raw_ctx_dim}, "
                        f"but model has {inference_engine.raw_ctx_dim}")
    
    if inference_engine.latent_dim != latent_dim:
        raise ValueError(f"InferenceEngine latent_dim mismatch: expected {latent_dim}, "
                        f"but model has {inference_engine.latent_dim}")
    
    if hasattr(actor, 'encoder'):
        # Check the first layer input size (should be latent_dim + intent_dim)
        first_layer = list(actor.encoder.children())[0]
        if isinstance(first_layer, nn.Linear):
            expected_input = first_layer.in_features - 3  # Subtract intent_dim
            if expected_input != latent_dim:
                raise ValueError(f"Actor latent_dim mismatch: expected {latent_dim}, "
                               f"but actor expects {expected_input}. "
                               f"Initialize Actor with latent_dim={latent_dim}")
    
    # Create session directory and file
    artifacts_dir = "/Users/zach/Documents/dev/cfd/tam/artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = os.path.join(artifacts_dir, f"training_session_{session_timestamp}.json")
    
    # Training data storage
    training_data = {
        "session_timestamp": session_timestamp,
        "episodes": episodes,
        "obstacles": obstacles,
        "bounds": bounds,
        "raw_ctx_dim": raw_ctx_dim,
        "latent_dim": latent_dim,
        "max_observed_obstacles": max_observed_obstacles,
        "initial_target": target_pos.tolist(),
        "episodes_data": []
    }
    
    # Initialize live plotter
    plotter = None
    if plot_live:
        plotter = LivePlotter(obstacles, target_pos.numpy(), bounds=bounds)
    
    print(f"Training {episodes} episodes... (Session: {session_timestamp})")
    print(f"Raw context dimension: {raw_ctx_dim} (3 goal + {max_observed_obstacles} max obstacles * 4)")
    print(f"Latent dimension: {latent_dim}")
    print(f"Actual obstacles in environment: {len(obstacles)}")

    for ep in range(episodes):
        current_pos = torch.zeros((1, 3))
        ep_loss = 0
        goals_reached = 0
        previous_velocity = None  # Track previous path direction for G1 continuity
        total_steps_taken = 0  # Track actual steps taken in this episode
        
        # Reset InferenceEngine hidden state at start of episode
        h_state = torch.zeros(1, latent_dim)  # (1, latent_dim) - initial situation
        
        # Each episode is a sequence of situations
        for step in range(10):
            total_steps_taken += 1  # Count this step
            
            # 1. INFER: Convert raw world data to latent situation
            raw_obs = sim.get_raw_observation(current_pos, target_pos, max_obstacles=max_observed_obstacles)  # (raw_ctx_dim,)
            raw_obs = raw_obs.unsqueeze(0)  # (1, raw_ctx_dim) for batch dimension
            
            # InferenceEngine compresses raw context into latent situation
            x_n = inference_engine(raw_obs, h_state)  # (1, latent_dim) - current situation
            h_state = x_n.detach()  # Pass memory forward (detach to prevent backprop through time)
            
            rel_goal = target_pos - current_pos 
            
            # 2. ACT: Propose affordances based on latent situation x_n
            # Actor operates on latent representation, not raw spatial features
            logits, mu_t, sigma_t, knot_mask, knot_steps = actor(x_n, rel_goal, previous_velocity=previous_velocity)
            
            # Selection (Categorical sampling for exploration)
            probs = F.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            idx = m.sample()
            
            # Extract selected tube - ensure we get (T, 3) shape, not (1, T, 3)
            idx_int = idx.item() if isinstance(idx, torch.Tensor) else idx
            selected_tube = mu_t[0, idx_int].squeeze(0)  # Remove any extra dimensions
            selected_sigma = sigma_t[0, idx_int].squeeze(0)
            
            # Ensure shapes are correct
            if selected_tube.dim() > 2:
                selected_tube = selected_tube.view(-1, 3)
            if selected_sigma.dim() > 2:
                selected_sigma = selected_sigma.view(-1, 1)
            
            # 1. Bind & Execute
            # The world responds with actual_p
            actual_p = sim.execute(selected_tube, selected_sigma, current_pos)
            
            # 3. SIMPLIFIED PRINCIPLED LOSS: Agency vs. Contradiction
            # With GRU and Residual Knots, we can use a cleaner formulation
            
            # A) CONTRADICTION: Did we stay in the tube?
            # Squared deviation makes hitting obstacles exponentially more painful
            expected_p = selected_tube + current_pos
            min_len = min(len(actual_p), len(expected_p))
            deviation = torch.norm(actual_p[:min_len] - expected_p[:min_len], dim=-1, keepdim=True)
            binding_loss = torch.mean((deviation**2) / (selected_sigma[:min_len] + 1e-6))
            
            # B) AGENCY COST: How 'expensive' was this path?
            # Extract selected knot_steps for this port
            selected_knot_steps = knot_steps[0, idx_int]  # (K, 3) - residual offsets
            
            # Penalize the sum of squared offsets (shorter, straighter paths are cheaper)
            path_cost = torch.mean(selected_knot_steps**2)
            
            # Penalize uncertainty (narrower tubes are better)
            uncertainty_cost = torch.mean(selected_sigma**2)
            
            # C) INTENT LOSS: Direct supervision to move toward goal
            tube_endpoint_global = (selected_tube[-1] + current_pos.squeeze()).unsqueeze(0)
            target_pos_expanded = target_pos.unsqueeze(0)
            intent_loss = F.mse_loss(tube_endpoint_global, target_pos_expanded)
            
            # Simplified Principled Loss
            loss = (1.0 * binding_loss          # Contradiction: stay in the tube
                   + 0.1 * path_cost            # Agency: minimize displacement (residual offsets)
                   + 0.05 * uncertainty_cost    # Agency: minimize uncertainty (narrower tubes)
                   + 0.5 * intent_loss)         # Intent: move toward goal
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update position and track velocity for G1 continuity
            current_pos = actual_p[-1].view(1, 3).detach()
            
            # Track velocity: direction of actual path taken (for G1 continuity)
            # Use the last segment of the actual path to capture real movement direction
            if len(actual_p) > 1:
                # Use last segment of actual path (what really happened)
                previous_velocity = (actual_p[-1] - actual_p[-2]).detach()  # (3,)
            elif len(actual_p) == 1 and previous_velocity is None:
                # First step: no previous velocity, use a small default direction
                # This will be updated on next step
                previous_velocity = torch.zeros(3, device=current_pos.device)
            # If path is only one point and we have previous velocity, keep it
            
            ep_loss += loss.item()
            
            # Update live plot
            if plotter is not None:
                plotter.update(selected_tube, selected_sigma, actual_p, current_pos, ep, step)
            
            # Check if goal is reached - if so, generate a new random goal
            if torch.norm(current_pos - target_pos) < 1.0:
                goals_reached += 1
                
                # Mark the reached goal in visualization
                if plotter is not None:
                    plotter.mark_goal_reached(target_pos)
                
                # Generate a new random goal (avoiding obstacles, current position, and respecting boundaries)
                # Use bounds with a small margin to ensure goals aren't right on the boundary
                margin = 0.5
                min_bounds = [b + margin for b in bounds['min']]
                max_bounds = [b - margin for b in bounds['max']]
                
                new_goal = torch.tensor([
                    random.uniform(min_bounds[0], max_bounds[0]),
                    random.uniform(min_bounds[1], max_bounds[1]),
                    random.uniform(min_bounds[2], max_bounds[2])
                ])
                
                # Ensure new goal is not too close to current position or obstacles
                min_dist = 3.0
                max_attempts = 50  # Prevent infinite loop
                attempts = 0
                while (torch.norm(new_goal - current_pos.squeeze()) < min_dist or
                       any(torch.norm(new_goal - torch.tensor(obs[0])) < obs[1] + 1.0 
                           for obs in obstacles)) and attempts < max_attempts:
                    new_goal = torch.tensor([
                        random.uniform(min_bounds[0], max_bounds[0]),
                        random.uniform(min_bounds[1], max_bounds[1]),
                        random.uniform(min_bounds[2], max_bounds[2])
                    ])
                    attempts += 1
                
                target_pos = new_goal
                
                # Update plotter with new goal (keep old target marker, just update position)
                if plotter is not None:
                    plotter.target_pos = target_pos.detach().numpy() if isinstance(target_pos, torch.Tensor) else target_pos
                    # Update target marker position
                    if plotter.target_marker is not None:
                        plotter.target_marker.remove()
                    plotter.target_marker = plotter.ax.scatter(
                        target_pos[0].item(), target_pos[1].item(), target_pos[2].item(),
                        color='green', s=300, marker='*', edgecolors='darkgreen',
                        linewidths=2, label='Target', zorder=10
                    )
                    plotter.ax.legend()
                    plt.draw()
                
                # Continue with new goal (don't break - keep learning!)
                
        # Save episode data
        final_dist = torch.norm(target_pos - current_pos).item()
        episode_data = {
            "episode": ep,
            "loss": float(ep_loss),
            "final_distance": float(final_dist),
            "goals_reached": goals_reached,
            "total_steps": total_steps_taken,
            "final_position": current_pos.squeeze().tolist(),
            "final_target": target_pos.tolist()
        }
        training_data["episodes_data"].append(episode_data)
        
        # Print summary every 5 episodes
        if ep % 5 == 0 or ep == episodes - 1:
            print(f"Episode {ep}/{episodes-1} | Loss: {ep_loss:.4f} | "
                  f"Final Dist: {final_dist:.2f} | Goals: {goals_reached} | Steps: {total_steps_taken}")
    
    # Save training data to file
    with open(session_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\nTraining complete. Session data saved to: {session_file}")
    
    if plotter is not None:
        print("Close the plot window to exit.")
        plt.ioff()
    plt.show()

# ==========================================
# 6. MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    # Fixed raw context dimension: independent of actual obstacle count
    # Raw context uses padding/truncation to handle variable obstacle counts
    MAX_OBSERVED_OBSTACLES = 10  # Maximum obstacles in observation (can add/remove obstacles freely!)
    raw_ctx_dim = 3 + MAX_OBSERVED_OBSTACLES * 4  # 3 + 10*4 = 43
    LATENT_DIM = 64  # Dimension of latent situation space
    
    # Initialize InferenceEngine (learns Infer() function)
    inference_engine = InferenceEngine(raw_ctx_dim=raw_ctx_dim, latent_dim=LATENT_DIM)
    
    # Initialize Actor (operates on latent situations)
    actor = Actor(latent_dim=LATENT_DIM, intent_dim=3, n_knots=6, interp_res=40)
    
    # Train both models together with live plotting
    # You can add/remove obstacles in train_actor without changing the models!
    train_actor(
        inference_engine=inference_engine,
        actor=actor,
        episodes=100,
        plot_live=True,
        max_observed_obstacles=MAX_OBSERVED_OBSTACLES,
        latent_dim=LATENT_DIM
    )