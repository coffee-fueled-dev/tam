import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. GEOMETRY UTILITIES: CATMULL-ROM SPLINE
# ==========================================

class CausalSpline:
    """
    Differentiable Catmull-Rom Spline interpolator.
    Maps discrete knots to a continuous dense trajectory.
    """
    @staticmethod
    def interpolate(knots, sigmas, resolution=40):
        """
        Args:
            knots: (Batch, K, 3) 3D control points
            sigmas: (Batch, K, 1) Affordance radii
            resolution: Total number of points desired in the trajectory
        """
        B, K, D = knots.shape
        device = knots.device
        
        # Guard against short tubes
        if K < 3: 
            return knots, sigmas

        # 1. Ghost Knots for Boundary Conditions
        p0 = 2 * knots[:, 0:1] - knots[:, 1:2]
        pn = 2 * knots[:, -1:] - knots[:, -2:-1]
        padded_knots = torch.cat([p0, knots, pn], dim=1) # (B, K+2, 3)

        # 2. Time Steps Calculation
        num_segments = K - 1
        # Ensure at least 4 points per segment for smoothness
        points_per_seg = max(4, int(np.ceil(resolution / num_segments)))
        
        t = torch.linspace(0, 1, points_per_seg, device=device)
        t = t[:-1] # Remove last point to avoid duplicates
        
        # Reshape for broadcasting: (1, T_seg, 1)
        t1 = t.view(1, -1, 1)
        t2 = t1 ** 2
        t3 = t1 ** 3
        
        all_mu_segments = []
        all_sigma_segments = []

        for i in range(1, K):
            # Control points for segment i
            P0 = padded_knots[:, i-1].unsqueeze(1) # (B, 1, 3)
            P1 = padded_knots[:, i].unsqueeze(1)
            P2 = padded_knots[:, i+1].unsqueeze(1)
            P3 = padded_knots[:, i+2].unsqueeze(1)
            
            # Catmull-Rom Equation
            term0 = 2 * P1
            term1 = (-P0 + P2) * t1
            term2 = (2*P0 - 5*P1 + 4*P2 - P3) * t2
            term3 = (-P0 + 3*P1 - 3*P2 + P3) * t3
            
            segment_pos = 0.5 * (term0 + term1 + term2 + term3)
            all_mu_segments.append(segment_pos)
            
            # Linear Sigma Interpolation
            s_start = sigmas[:, i-1].unsqueeze(1)
            s_end = sigmas[:, i].unsqueeze(1)
            segment_sig = s_start + (s_end - s_start) * t1
            all_sigma_segments.append(segment_sig)

        # Append final endpoint
        all_mu_segments.append(knots[:, -1:].view(B, 1, 3))
        all_sigma_segments.append(sigmas[:, -1:].view(B, 1, 1))

        # Concatenate all segments
        mu_t = torch.cat(all_mu_segments, dim=1)
        sigma_t = torch.cat(all_sigma_segments, dim=1)
        
        return mu_t, sigma_t


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
        
        # Flatten for efficient interpolation
        knots_flat = knots_rel.view(B * self.n_ports, self.n_knots, 3)
        sigmas_flat = sigmas_raw.view(B * self.n_ports, self.n_knots, 1)
        
        # Interpolate
        mu_dense, sigma_dense = CausalSpline.interpolate(
            knots_flat, sigmas_flat, resolution=self.interp_res
        )
        
        # Reshape back to (B, M, T, 3)
        mu_t = mu_dense.view(B, self.n_ports, -1, 3)
        sigma_t = sigma_dense.view(B, self.n_ports, -1, 1)
        
        return logits, mu_t, sigma_t

    def select_port(self, logits, mu_t, intent_target):
        # Heuristic: Closest tube end-point to target, weighted by logits
        end_points = mu_t[:, :, -1, :] 
        dist = torch.norm(end_points - intent_target.unsqueeze(1), dim=-1)
        score = F.log_softmax(logits, dim=-1) - dist
        return torch.argmax(score, dim=-1)


# ==========================================
# 3. SIMULATION WRAPPER (FIXED)
# ==========================================

class SimulationWrapper:
    def __init__(self, actor):
        self.actor = actor
        self.debug_tubes = [] 
        
    def run_episode(self, start_pos, target_pos, obstacles):
        current_pos = torch.tensor(start_pos, dtype=torch.float32).view(1, 3)
        target = torch.tensor(target_pos, dtype=torch.float32).view(1, 3)
        
        path_history = [current_pos.squeeze().numpy()]
        steps = 0
        max_steps = 40
        
        print(f"--- New Episode: Goal {target_pos} ---")
        
        while steps < max_steps:
            rel_goal = target - current_pos
            obs = torch.randn(1, 10) 
            
            with torch.no_grad():
                logits, mu_t, sigma_t = self.actor(obs, rel_goal)
                
                # FIX: Use .item() to get a python integer, not a Tensor
                best_idx_tensor = self.actor.select_port(logits, mu_t, rel_goal)
                best_idx = best_idx_tensor.item()
                
                # Get the committed tube (T, 3)
                # We add current_pos to transform from relative -> global
                committed_mu = mu_t[0, best_idx] + current_pos
                committed_sigma = sigma_t[0, best_idx]
                
                # Save for plotting (take every 5th step to save memory)
                if steps % 5 == 0:
                    self.debug_tubes.append(committed_mu.numpy())

            # Debug output
            pts = len(committed_mu)
            print(f"Step {steps}: Bound to tube with {pts} pts.")
            
            if pts < 5:
                print("Error: Tube too short. Aborting.")
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
                for obs_p, obs_r in obstacles:
                    d = torch.norm(actual_p - torch.tensor(obs_p))
                    if d < obs_r:
                        # Simple bounce/block
                        print(f"  > HIT OBSTACLE at sub-step {t}")
                        vec = actual_p - torch.tensor(obs_p)
                        actual_p = torch.tensor(obs_p) + (vec / d) * obs_r
                
                # BINDING CHECK
                # If reality (actual_p) is too far from plan (expected_p)
                deviation = torch.norm(actual_p - expected_p)
                if deviation > affordance_r:
                    print(f"  > Binding Failed! Dev {deviation:.2f} > Sig {affordance_r.item():.2f}")
                    # Update state to where we actually ended up
                    current_pos = actual_p.view(1, 3)
                    path_history.append(current_pos.squeeze().numpy())
                    binding_broken = True
                    break # Break inner loop to Re-plan
                
                # If valid, commit to this step
                current_pos = actual_p.view(1, 3)
                path_history.append(current_pos.squeeze().numpy())
                
                if torch.norm(current_pos - target) < 1.0:
                    print("Goal Reached!")
                    return np.array(path_history)

            steps += 1

        return np.array(path_history)

# ==========================================
# 4. MAIN & PLOTTING
# ==========================================

if __name__ == "__main__":
    actor = Actor(obs_dim=10, intent_dim=3, n_knots=6, interp_res=40)
    sim = SimulationWrapper(actor)
    
    # Define Obstacles: (Pos, Radius)
    obstacles = [([5.0, 5.0, 0.0], 2.5)]
    
    # Run
    path = sim.run_episode([0,0,0], [10, 10, 0], obstacles)
    
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot Obstacle
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = 5 + 2.5 * np.outer(np.cos(u), np.sin(v))
    y = 5 + 2.5 * np.outer(np.sin(u), np.sin(v))
    z = 0 + 2.5 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='red', alpha=0.2, edgecolor='none')
    
    # 2. Plot Attempted Tubes (Green lines showing intent)
    for tube in sim.debug_tubes:
        ax.plot(tube[:,0], tube[:,1], tube[:,2], color='green', alpha=0.3, linestyle='--')

    # 3. Plot Actual Path (Blue solid line)
    if len(path) > 1:
        ax.plot(path[:,0], path[:,1], path[:,2], color='blue', linewidth=3, label='Robot Path')
        ax.scatter(path[-1,0], path[-1,1], path[-1,2], color='blue', s=60, label='End')
    
    ax.scatter(0, 0, 0, color='black', s=60, label='Start')

    ax.set_title("Trajectory Affordance: Navigation & Binding")
    ax.legend()
    plt.show()