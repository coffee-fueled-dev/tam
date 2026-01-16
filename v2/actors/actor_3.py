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


# ==========================================
# 1. GEOMETRY UTILITIES: CATMULL-ROM SPLINE
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
# 2. THE ACTOR: CAUSAL KNOT ENGINE
# ==========================================

class Actor(nn.Module):
    def __init__(self, obs_dim, intent_dim, n_ports=4, n_knots=6, interp_res=40):
        super().__init__()
        self.n_ports = n_ports
        self.n_knots = n_knots
        self.interp_res = interp_res
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + intent_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256)
        )
        
        # Port Proposer
        # Output size: M * (Logit + K*3 + K*1)
        self.port_head = nn.Linear(256, n_ports * (1 + n_knots * 4))
        
    def forward(self, obs, intent):
        """
        Returns:
            logits: (B, M)
            mu_t:   (B, M, T, 3) - Interpolated Tube
            sigma_t:(B, M, T, 1) - Interpolated Radius
        """
        B = obs.size(0)
        situation = self.encoder(torch.cat([obs, intent], dim=-1))
        
        raw_out = self.port_head(situation).view(B, self.n_ports, -1)
        logits = raw_out[:, :, 0]
        
        # Extract Knots & Sigmas
        geo_params = raw_out[:, :, 1:].view(B, self.n_ports, self.n_knots, 4)
        knots_raw = geo_params[..., :3]
        sigmas_raw = F.softplus(geo_params[..., 3:]) + 0.1
        
        # Causal Anchoring: Knot 0 is always (0,0,0) relative to actor
        knots_rel = knots_raw - knots_raw[:, :, 0:1, :]
        
        # INTENT BIAS: Push the last knot toward the intent
        # This ensures that even with random weights, tubes are biased toward the goal
        # Reshape intent to (B, 1, 3) to broadcast across ports (M dimension)
        intent_bias = intent.view(B, 1, 3)  # (B, 1, 3) broadcasts to (B, M, 3)
        # Add a fraction of intent to the last knot to guide the network
        # Using 0.5 means the network learns to add the other 0.5 through training
        knots_rel[:, :, -1, :] = knots_rel[:, :, -1, :] + intent_bias * 0.5
        
        # Flatten for efficient interpolation
        knots_flat = knots_rel.view(B * self.n_ports, self.n_knots, 3)
        sigmas_flat = sigmas_raw.view(B * self.n_ports, self.n_knots, 1)
        
        # Interpolate
        mu_dense, sigma_dense = CausalSpline.interpolate(
            knots_flat, sigmas_flat, resolution=self.interp_res
        )
        
        # Reshape back to (B, M, T, 3)
        T = mu_dense.shape[1]  # Number of interpolated points
        mu_t = mu_dense.view(B, self.n_ports, T, 3)
        sigma_t = sigma_dense.view(B, self.n_ports, T, 1)
        
        return logits, mu_t, sigma_t

    def select_port(self, logits, mu_t, intent_target):
        # Heuristic: Closest tube end-point to target, weighted by logits
        end_points = mu_t[:, :, -1, :] 
        dist = torch.norm(end_points - intent_target.unsqueeze(1), dim=-1)
        score = F.log_softmax(logits, dim=-1) - dist
        return torch.argmax(score, dim=-1)

    def train_step(self, obs, intent, optimizer, sim_wrapper, current_pos=None):
        # 1. Propose Tube
        logits, mu_t, sigma_t = self(obs, intent)
        
        # 2. Select and Bind
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        idx = dist.sample()
        
        # 3. World Interaction (The Simulation)
        # returns actual path taken after potential collisions
        if current_pos is None:
            current_pos = torch.zeros((1, 3), device=mu_t.device)
        actual_path = sim_wrapper.execute(mu_t[0, idx], sigma_t[0, idx], current_pos)
        
        # 4. Calculate TA Losses with improved formulation
        expected_p = mu_t[0, idx] + current_pos
        min_len = min(len(actual_path), len(expected_p))
        deviation = torch.norm(actual_path[:min_len] - expected_p[:min_len], dim=-1, keepdim=True)
        
        # A) Squared Binding Loss: Makes hitting obstacles exponentially more painful
        binding_loss = torch.mean((deviation**2) / (sigma_t[0, idx, :min_len] + 1e-6))
        
        # B) Quadratic Agency Loss: Punishes large sigmas aggressively
        agency_loss = torch.mean(sigma_t[0, idx]**2)
        
        # C) Path Smoothness (Curvature Penalty)
        knots = mu_t[0, idx]
        if len(knots) > 2:
            diffs = knots[1:] - knots[:-1]
            if len(diffs) > 1:
                accel = diffs[1:] - diffs[:-1]
                smoothness_loss = torch.mean(accel**2)
            else:
                smoothness_loss = torch.tensor(0.0, device=knots.device)
        else:
            smoothness_loss = torch.tensor(0.0, device=knots.device)
        
        # Cb) Path Length Penalty (Energy Efficiency)
        if len(knots) > 1:
            segment_lengths = torch.norm(knots[1:] - knots[:-1], dim=-1)
            total_path_length = torch.sum(segment_lengths)
            straight_line_dist = torch.norm(intent)
            path_length_loss = total_path_length / (straight_line_dist + 1e-6)
        else:
            path_length_loss = torch.tensor(0.0, device=knots.device)
        
        # Cc) Curvature Penalty (Energy for Turning)
        if len(knots) > 2:
            vec1 = knots[1:-1] - knots[:-2]  # Segments ending at interior points
            vec2 = knots[2:] - knots[1:-1]   # Segments starting at interior points
            vec1_norm = vec1 / (torch.norm(vec1, dim=-1, keepdim=True) + 1e-6)
            vec2_norm = vec2 / (torch.norm(vec2, dim=-1, keepdim=True) + 1e-6)
            cos_angles = torch.sum(vec1_norm * vec2_norm, dim=-1)
            curvature_penalty = torch.mean(1.0 - cos_angles)
        else:
            curvature_penalty = torch.tensor(0.0, device=knots.device)
        
        # D) Goal Progress
        progress_reward = torch.norm(intent) - torch.norm(intent - (actual_path[-1] - current_pos.squeeze()))
        
        # Rebalanced Total Loss with complexity penalties
        total_loss = (-dist.log_prob(idx) * progress_reward 
                     + 5.0 * binding_loss 
                     + 1.5 * agency_loss 
                     + 0.5 * smoothness_loss
                     + 0.3 * path_length_loss
                     + 0.2 * curvature_penalty)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()


# ==========================================
# 3. SIMULATION WRAPPER (FIXED)
# ==========================================

class SimulationWrapper:
    def __init__(self, obstacles):
        self.obstacles = obstacles
        self.debug_tubes = []
    
    def execute(self, mu_t, sigma_t, current_pos):
        """
        Execute a tube in the world, checking for obstacles and binding violations.
        
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
        
        # Ensure sigma_t is 1D
        if sigma_t.dim() == 0:
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
            if sigma_t.dim() == 0:
                affordance_r = sigma_t.item()
            else:
                affordance_r = sigma_t[t_idx].item() if hasattr(sigma_t[t_idx], 'item') else float(sigma_t[t_idx])
            
            # Start with expected position
            actual_p = expected_p.clone()
            
            # Check for obstacles
            for obs_p, obs_r in self.obstacles:
                obs_pos = torch.tensor(obs_p, dtype=torch.float32, device=actual_p.device)
                d = torch.norm(actual_p - obs_pos)
                if d < obs_r:
                    # Collision: push away from obstacle
                    vec = actual_p - obs_pos
                    if d > 1e-6:
                        actual_p = obs_pos + (vec / d) * obs_r
                    else:
                        # If exactly on obstacle, push in a random direction
                        actual_p = obs_pos + torch.tensor([obs_r, 0, 0], dtype=torch.float32, device=actual_p.device)
            
            # Binding check: if deviation exceeds sigma, stop early
            deviation = torch.norm(actual_p - expected_p)
            if deviation > affordance_r:
                # Binding broken, return path up to this point
                actual_path.append(actual_p.clone())
                break
            
            actual_path.append(actual_p.clone())
        
        return torch.stack(actual_path) 
        
    def run_episode(self, actor, start_pos, target_pos):
        """
        Run a full episode using the actor for evaluation.
        
        Args:
            actor: The Actor model to use
            start_pos: Starting position [x, y, z]
            target_pos: Target position [x, y, z]
        
        Returns:
            path: numpy array of actual path taken
        """
        current_pos = torch.tensor(start_pos, dtype=torch.float32).view(1, 3)
        target = torch.tensor(target_pos, dtype=torch.float32).view(1, 3)
        
        path_history = [current_pos.squeeze().numpy()]
        steps = 0
        max_steps = 40
        
        while steps < max_steps:
            rel_goal = target - current_pos
            obs = torch.randn(1, 10) 
            
            with torch.no_grad():
                logits, mu_t, sigma_t = actor(obs, rel_goal)
                
                # FIX: Use .item() to get a python integer, not a Tensor
                best_idx_tensor = actor.select_port(logits, mu_t, rel_goal)
                best_idx = best_idx_tensor.item()
                
                # Get the committed tube (T, 3)
                # We add current_pos to transform from relative -> global
                committed_mu = mu_t[0, best_idx] + current_pos
                committed_sigma = sigma_t[0, best_idx]
                
                # Save for plotting (take every 5th step to save memory)
                if steps % 5 == 0:
                    self.debug_tubes.append(committed_mu.numpy())

            pts = len(committed_mu)
            
            if pts < 5:
                break

            # EXECUTE COMMITMENT (Bind -> Contradict)
            binding_broken = False
            
            # Start from t=1 because t=0 is our current position
            for t in range(1, pts):
                expected_p = committed_mu[t]
                affordance_r = committed_sigma[t]
                
                # Move
                actual_p = expected_p.clone()
                
                # Simple Physics: Check for Obstacles
                for obs_p, obs_r in self.obstacles:
                    d = torch.norm(actual_p - torch.tensor(obs_p))
                    if d < obs_r:
                        # Simple bounce/block
                        vec = actual_p - torch.tensor(obs_p)
                        actual_p = torch.tensor(obs_p) + (vec / d) * obs_r
                
                # BINDING CHECK
                # If reality (actual_p) is too far from plan (expected_p)
                deviation = torch.norm(actual_p - expected_p)
                if deviation > affordance_r:
                    # Update state to where we actually ended up
                    current_pos = actual_p.view(1, 3)
                    path_history.append(current_pos.squeeze().numpy())
                    binding_broken = True
                    break # Break inner loop to Re-plan
                
                # If valid, commit to this step
                current_pos = actual_p.view(1, 3)
                path_history.append(current_pos.squeeze().numpy())
                
                if torch.norm(current_pos - target) < 1.0:
                    return np.array(path_history)

            steps += 1

        return np.array(path_history)

# ==========================================
# 4. TRAINING & LIVE PLOTTING
# ==========================================

class LivePlotter:
    def __init__(self, obstacles, target_pos):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.obstacles = obstacles
        self.target_pos = target_pos
        
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
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Live Training: Tube Warping Visualization')
        self.ax.legend()
        self.ax.set_xlim(-2, 12)
        self.ax.set_ylim(-2, 12)
        self.ax.set_zlim(-2, 2)
        
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
        self.ax.set_title(f'Episode {episode}, Step {step} | Avg σ: {avg_sigma:.3f}')
        
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
    
    def close(self):
        plt.ioff()
        plt.close(self.fig)


def train_actor(actor, episodes=50, plot_live=True):
    optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    obstacles = [([5.0, 5.0, 0.0], 2.5)]
    sim = SimulationWrapper(obstacles)
    target_pos = torch.tensor([10.0, 10.0, 0.0])
    
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
        "initial_target": target_pos.tolist(),
        "episodes_data": []
    }
    
    # Initialize live plotter
    plotter = None
    if plot_live:
        plotter = LivePlotter(obstacles, target_pos.numpy())
    
    print(f"Training {episodes} episodes... (Session: {session_timestamp})")

    for ep in range(episodes):
        current_pos = torch.zeros((1, 3))
        ep_loss = 0
        goals_reached = 0
        
        # Each episode is a sequence of situations
        for step in range(10):
            rel_goal = target_pos - current_pos
            obs = torch.randn(1, 10) 
            
            logits, mu_t, sigma_t = actor(obs, rel_goal)
            
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
            
            # 2. Calculate Contradiction (Binding Loss)
            # Squared deviation makes hitting obstacles exponentially more painful
            # than having a slightly wider tube
            expected_p = selected_tube + current_pos
            # Align lengths for comparison
            min_len = min(len(actual_p), len(expected_p))
            deviation = torch.norm(actual_p[:min_len] - expected_p[:min_len], dim=-1, keepdim=True)
            # Squared binding loss: makes large deviations much more expensive
            binding_loss = torch.mean((deviation**2) / (selected_sigma[:min_len] + 1e-6))
            
            # 3. Calculate Agency Loss (Regularizer)
            # Quadratic penalty: punishes large sigmas aggressively
            # Prevents actor from "buying" its way out with uncertainty
            agency_loss = torch.mean(selected_sigma**2)
            
            # 4. Path Smoothness (Curvature Penalty)
            # Penalize sharp 'jerks' in the knots to prevent erratic jumping
            # and encourage smooth, efficient splines
            knots = selected_tube
            if len(knots) > 2:
                diffs = knots[1:] - knots[:-1]
                if len(diffs) > 1:
                    accel = diffs[1:] - diffs[:-1]
                    smoothness_loss = torch.mean(accel**2)
                else:
                    smoothness_loss = torch.tensor(0.0, device=knots.device)
            else:
                smoothness_loss = torch.tensor(0.0, device=knots.device)
            
            # 4b. Path Length Penalty (Energy Efficiency)
            # Penalize total distance traveled - encourages shorter, more efficient paths
            if len(knots) > 1:
                segment_lengths = torch.norm(knots[1:] - knots[:-1], dim=-1)
                total_path_length = torch.sum(segment_lengths)
                # Normalize by straight-line distance to goal for fairness
                straight_line_dist = torch.norm(rel_goal)
                path_length_loss = total_path_length / (straight_line_dist + 1e-6)  # Efficiency ratio
            else:
                path_length_loss = torch.tensor(0.0, device=knots.device)
            
            # 4c. Curvature Penalty (Energy for Turning)
            # Penalize how much the path deviates from a straight line
            # Higher curvature = more energy needed to turn
            if len(knots) > 2:
                # Calculate angles between consecutive segments
                # vec1[i] = segment from knots[i] to knots[i+1]
                # vec2[i] = segment from knots[i+1] to knots[i+2]
                # We want to compare vec1[i] with vec2[i] (angle at knots[i+1])
                vec1 = knots[1:-1] - knots[:-2]  # (T-2, 3) segments ending at interior points
                vec2 = knots[2:] - knots[1:-1]   # (T-2, 3) segments starting at interior points
                # Normalize vectors
                vec1_norm = vec1 / (torch.norm(vec1, dim=-1, keepdim=True) + 1e-6)
                vec2_norm = vec2 / (torch.norm(vec2, dim=-1, keepdim=True) + 1e-6)
                # Dot product gives cosine of angle (1 = straight, 0 = 90deg turn, -1 = 180deg)
                cos_angles = torch.sum(vec1_norm * vec2_norm, dim=-1)
                # Penalize deviation from straight (1 - cos_angle, so 0 for straight, 2 for 180deg)
                curvature_penalty = torch.mean(1.0 - cos_angles)
            else:
                curvature_penalty = torch.tensor(0.0, device=knots.device)
            
            # 4d. Path Directness Loss (CRITICAL: Prevents looping arcs)
            # Penalize the interpolated tube for deviating from straight-line segments between knots
            # This encourages the actor to place knots such that the spline creates direct paths
            # Method: For each segment between consecutive knots, measure maximum deviation from straight line
            if len(selected_tube) > 1 and len(knots) > 1:
                directness_penalty = torch.tensor(0.0, device=selected_tube.device)
                n_knots = len(knots)
                n_interp = len(selected_tube)
                
                # Distribute interpolated points across knot segments
                # Each knot segment should have approximately n_interp / (n_knots - 1) points
                points_per_segment = max(1, n_interp // max(1, n_knots - 1))
                
                for i in range(n_knots - 1):
                    k_start = knots[i]
                    k_end = knots[i + 1]
                    
                    # Get interpolated points for this segment
                    interp_start_idx = min(i * points_per_segment, n_interp - 1)
                    interp_end_idx = min((i + 1) * points_per_segment, n_interp - 1)
                    
                    if interp_end_idx > interp_start_idx:
                        interp_segment = selected_tube[interp_start_idx:interp_end_idx + 1]
                        
                        # Compute straight-line path between knots
                        segment_vec = k_end - k_start
                        segment_len = torch.norm(segment_vec) + 1e-6
                        
                        # For each interpolated point, compute distance to straight line
                        deviations = []
                        segment_len_scalar = segment_len.item()  # Convert to Python float for clamp
                        for p in interp_segment:
                            # Vector from k_start to p
                            vec_to_p = p - k_start
                            # Project onto segment direction (normalized)
                            segment_dir = segment_vec / segment_len
                            proj_len = torch.dot(vec_to_p, segment_dir)
                            # Clamp to segment bounds [0, segment_len]
                            proj_len = torch.clamp(proj_len, 0.0, segment_len_scalar)
                            # Find closest point on straight line
                            closest_on_line = k_start + proj_len * segment_dir
                            # Distance from interpolated point to straight line
                            deviation = torch.norm(p - closest_on_line)
                            deviations.append(deviation)
                        
                        if len(deviations) > 0:
                            # Use maximum deviation (catches looping arcs)
                            max_deviation = torch.stack(deviations).max()
                            directness_penalty += max_deviation
                
                # Normalize by number of segments
                if n_knots > 1:
                    directness_penalty = directness_penalty / (n_knots - 1)
            else:
                directness_penalty = torch.tensor(0.0, device=selected_tube.device)
            
            # 5. Intent Loss (CRITICAL: Direct supervision to fix drift)
            # This is the "pull" that ensures tubes point toward the goal
            # The tube's endpoint (in global coordinates) should be close to the target
            tube_endpoint_global = (selected_tube[-1] + current_pos.squeeze()).unsqueeze(0)  # Ensure (1, 3) shape
            target_pos_expanded = target_pos.unsqueeze(0)  # Ensure (1, 3) shape
            intent_loss = F.mse_loss(tube_endpoint_global, target_pos_expanded)
            
            # 6. Reward (Progress to goal)
            # Distance-based progress: positive when moving closer
            current_dist = torch.norm(target_pos - current_pos)
            final_dist = torch.norm(target_pos - actual_p[-1])
            progress = current_dist - final_dist  # Positive when moving closer
            
            # Scale progress reward to be comparable
            initial_dist = torch.norm(target_pos)  # Distance from origin to goal
            progress_scaled = progress / (initial_dist + 1e-6)  # Normalize to [0, 1] range
            
            # Policy loss: negative log prob * reward (so we maximize reward)
            policy_loss = -m.log_prob(idx) * progress_scaled * 20.0  # Scaled progress reward
            
            # Total TA Loss - Rebalanced weights
            # Intent loss is critical to prevent drift - it directly supervises tube direction
            loss = (0.5 * intent_loss          # Direct supervision: tubes must point to goal
                   + policy_loss               # Progress reward for moving closer
                   + 5.0 * binding_loss        # Binding: stay in the tube
                   + 1.5 * agency_loss         # Agency: don't inflate sigma
                   + 0.5 * smoothness_loss     # Smoothness: avoid jerky paths
                   + 0.3 * path_length_loss    # Energy: minimize path length
                   + 0.2 * curvature_penalty   # Energy: minimize turning
                   + 2.0 * directness_penalty) # Directness: prevent looping arcs
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            current_pos = actual_p[-1].view(1, 3).detach()
            ep_loss += loss.item()
            
            # Update live plot
            if plotter is not None:
                plotter.update(selected_tube, selected_sigma, actual_p, current_pos, ep, step)
            
            # Check if goal is reached - if so, generate a new random goal
            if torch.norm(current_pos - target_pos) < 1.0:
                goals_reached += 1
                
                # Generate a new random goal (avoiding obstacles and current position)
                # Place goal in a reasonable range: [2, 12] for x, y; [-2, 2] for z
                new_goal = torch.tensor([
                    random.uniform(2.0, 12.0),
                    random.uniform(2.0, 12.0),
                    random.uniform(-2.0, 2.0)
                ])
                
                # Ensure new goal is not too close to current position or obstacles
                min_dist = 3.0
                while (torch.norm(new_goal - current_pos.squeeze()) < min_dist or
                       any(torch.norm(new_goal - torch.tensor(obs[0])) < obs[1] + 1.0 
                           for obs in obstacles)):
                    new_goal = torch.tensor([
                        random.uniform(2.0, 12.0),
                        random.uniform(2.0, 12.0),
                        random.uniform(-2.0, 2.0)
                    ])
                
                target_pos = new_goal
                
                # Update plotter with new goal
                if plotter is not None:
                    plotter.target_pos = target_pos.detach().numpy() if isinstance(target_pos, torch.Tensor) else target_pos
                    # Re-draw target marker
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
                # Optionally, you can break here if you want to end episode after each goal
                # break
                
        # Save episode data
        final_dist = torch.norm(target_pos - current_pos).item()
        episode_data = {
            "episode": ep,
            "loss": float(ep_loss),
            "final_distance": float(final_dist),
            "goals_reached": goals_reached,
            "final_position": current_pos.squeeze().tolist(),
            "final_target": target_pos.tolist()
        }
        training_data["episodes_data"].append(episode_data)
        
        # Print summary every 5 episodes
        if ep % 5 == 0 or ep == episodes - 1:
            print(f"Episode {ep}/{episodes-1} | Loss: {ep_loss:.4f} | "
                  f"Final Dist: {final_dist:.2f} | Goals: {goals_reached}")
    
    # Save training data to file
    with open(session_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"\nTraining complete. Session data saved to: {session_file}")
    
    if plotter is not None:
        print("Close the plot window to exit.")
        plt.ioff()
        plt.show()

# ==========================================
# 5. MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    actor = Actor(obs_dim=10, intent_dim=3, n_knots=6, interp_res=40)
    
    # Train the actor with live plotting
    train_actor(actor, episodes=50, plot_live=True)