import torch
import numpy as np

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
        
    def run_episode(self, inference_engine, actor, tkn_processor, start_pos, target_pos, max_observed_obstacles=10, latent_dim=64):
        """
        Run a full episode using the HybridInferenceEngine and Actor for evaluation.
        
        Args:
            inference_engine: The HybridInferenceEngine model to use
            actor: The Actor model to use
            tkn_processor: The TknProcessor for tokenizing observations
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
        
        # Reset tkn processor
        tkn_processor.reset_episode()
        
        while steps < max_steps:
            rel_goal = target - current_pos
            
            # Get raw observation
            raw_obs = self.get_raw_observation(current_pos, target, max_obstacles=max_observed_obstacles)
            
            with torch.no_grad():
                # Process through tkn
                tkn_output = tkn_processor.process_observation(
                    current_pos, raw_obs, self.obstacles
                )
                lattice_tokens = tkn_output["lattice_tokens"].unsqueeze(0)  # (1, num_heads)
                lattice_traits = tkn_output["lattice_traits"].unsqueeze(0)  # (1, num_heads, 2)
                rel_goal_tensor = tkn_output["rel_goal"].unsqueeze(0)  # (1, 3)
                
                # INFER: Convert tokens + traits + intent to latent situation
                x_n = inference_engine(lattice_tokens, lattice_traits, rel_goal_tensor, h_state)  # (1, latent_dim)
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