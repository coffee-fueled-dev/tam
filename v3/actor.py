import torch
import torch.nn as nn
import torch.nn.functional as F
from v3.geometry import CausalSpline


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

