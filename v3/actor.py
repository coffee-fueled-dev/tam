import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from v3.geometry import CausalSpline


class Actor(nn.Module):
    """
    Fibration Actor: Basis-Projection Actor that outputs weights for basis functions.
    
    Instead of outputting raw coordinate offsets, this Actor outputs weights for a set
    of basis functions that describe the available "fiber" (affordance space). This
    implements the "Section Selection" logic for a fibration over the tkn base space.
    
    Key features:
    - Basis function weights instead of raw offsets
    - Per-dimension precision (sigma vector) for anisotropic affordance tubes
    - Dimension-agnostic: works with any state_dim
    """
    def __init__(self, latent_dim, state_dim, n_ports=4, n_knots=6, n_basis=8, interp_res=40):
        """
        Args:
            latent_dim: Dimension of latent situation space
            state_dim: Dimension of state space (e.g., 3 for 3D, 6 for 6D robotic arm)
            n_ports: Number of affordance ports to propose
            n_knots: Number of knots per tube
            n_basis: Number of basis functions for knot generation
            interp_res: Resolution for spline interpolation
        """
        super().__init__()
        self.n_ports = n_ports
        self.n_knots = n_knots
        self.n_basis = n_basis
        self.interp_res = interp_res
        self.state_dim = state_dim
        
        # Encoder: processes latent situation + intent
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim + state_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 256)
        )
        
        # Basis function generator: maps latent to basis weights
        # For each port, we generate weights for n_basis functions
        # Each basis function contributes to all n_knots * state_dim coordinates
        # Output: n_ports * (logit + n_basis + state_dim)
        #   - logit: port selection logit
        #   - n_basis: basis function weights
        #   - state_dim: per-dimension precision (sigma vector)
        self.port_head = nn.Linear(256, n_ports * (1 + n_basis + state_dim))
        
        # Basis functions: define a set of basis functions for knot generation
        # These represent the "Fibers" - geometric patterns the Actor can select from
        # We initialize with geometric patterns (smooth curves) rather than pure noise
        # Each basis function maps knot_index -> contribution to state_dim coordinates
        basis_init = torch.zeros(n_basis, n_knots, state_dim)
        
        # Initialize with smooth geometric patterns
        for b_idx in range(n_basis):
            # Create smooth patterns using sinusoidal and polynomial bases
            t = torch.linspace(0, 1, n_knots)
            
            # Different patterns for different basis functions
            if b_idx % 3 == 0:
                # Linear patterns in different directions
                dim = b_idx % state_dim
                basis_init[b_idx, :, dim] = t - 0.5  # Centered linear
            elif b_idx % 3 == 1:
                # Sinusoidal patterns
                freq = (b_idx // 3) + 1
                dim = b_idx % state_dim
                basis_init[b_idx, :, dim] = 0.3 * torch.sin(2 * np.pi * freq * t)
            else:
                # Polynomial patterns
                dim = b_idx % state_dim
                basis_init[b_idx, :, dim] = 0.2 * (t - 0.5) ** 2
        
        # Add small random noise for diversity
        basis_init = basis_init + torch.randn(n_basis, n_knots, state_dim) * 0.05
        
        self.basis_functions = nn.Parameter(basis_init)
        
        # Knot existence mask generator (optional, for variable-length knots)
        self.knot_existence_head = nn.Linear(256, n_ports * n_knots)
        
    def _generate_basis_functions(self):
        """
        Generate basis functions for knot construction.
        
        Returns:
            basis: (n_basis, n_knots, state_dim) tensor of basis function values
        """
        # Use learned basis functions
        return self.basis_functions
    
    def forward(self, latent_situation, intent, previous_velocity=None):
        """
        Generate affordance tubes using basis function projection.
        
        Args:
            latent_situation: (B, latent_dim) latent situation from InferenceEngine
            intent: (B, state_dim) intent/target direction tensor
            previous_velocity: Optional (B, state_dim) or (state_dim,) tensor for G1 continuity
        
        Returns:
            logits: (B, M) port selection logits
            mu_t: (B, M, T, state_dim) interpolated tube trajectories
            sigma_t: (B, M, T, state_dim) per-dimension precision (sigma vector)
            knot_mask: (B, M, K) knot existence mask
            knot_weights: (B, M, n_basis) basis function weights (for analysis)
        """
        B = latent_situation.size(0)
        situation = self.encoder(torch.cat([latent_situation, intent], dim=-1))
        
        # Generate port outputs
        raw_out = self.port_head(situation).view(B, self.n_ports, -1)
        logits = raw_out[:, :, 0]  # (B, M)
        
        # Extract basis weights and precision (sigma) vector
        basis_weights = raw_out[:, :, 1:1+self.n_basis]  # (B, M, n_basis)
        sigma_raw = raw_out[:, :, 1+self.n_basis:]  # (B, M, state_dim)
        
        # Normalize basis weights to prevent extreme values
        # This ensures the Actor selects from the fiber space smoothly
        basis_weights = torch.tanh(basis_weights)  # Constrain to [-1, 1]
        
        # Per-dimension precision: one sigma per dimension (anisotropic affordance tube)
        # Use exp for stability (log-space precision)
        # Each dimension gets its own precision, allowing hyperellipsoid affordance tubes
        sigmas = torch.exp(sigma_raw) + 0.1  # (B, M, state_dim) - per-dimension precision
        
        # Generate knot existence mask
        knot_existence_logits = self.knot_existence_head(situation).view(B, self.n_ports, self.n_knots)
        knot_mask_raw = torch.sigmoid(knot_existence_logits)  # (B, M, K)
        
        # Ensure first and last knots are always active
        device = knot_mask_raw.device
        ones_first = torch.ones(B, self.n_ports, 1, device=device)
        ones_last = torch.ones(B, self.n_ports, 1, device=device)
        knot_mask = torch.cat([ones_first, knot_mask_raw[:, :, 1:-1], ones_last], dim=-1)
        
        # Generate knots from basis functions
        # Basis functions: (n_basis, n_knots, state_dim)
        basis = self._generate_basis_functions()  # (n_basis, n_knots, state_dim)
        
        # Weighted combination: (B, M, n_basis) @ (n_basis, n_knots, state_dim)
        # Use einsum for efficient batched matrix multiplication
        # 'bmn, nkd -> bmkd' where b=batch, m=port, n=basis, k=knot, d=dim
        knots = torch.einsum('bmn, nkd -> bmkd', basis_weights, basis)  # (B, M, n_knots, state_dim)
        
        # Causal anchoring: ensure first knot is at origin
        knots = knots - knots[:, :, 0:1, :]
        
        # G1 CONTINUITY: Align first segment with previous velocity
        if previous_velocity is not None:
            if previous_velocity.dim() == 1:
                prev_vel = previous_velocity.unsqueeze(0)  # (1, state_dim)
            else:
                prev_vel = previous_velocity  # (B, state_dim) or (1, state_dim)
            
            if prev_vel.size(0) == 1 and B > 1:
                prev_vel = prev_vel.expand(B, -1)
            elif prev_vel.size(0) != B:
                prev_vel = prev_vel[:B]
            
            prev_vel_norm = prev_vel / (torch.norm(prev_vel, dim=-1, keepdim=True) + 1e-6)  # (B, state_dim)
            
            if self.n_knots > 1:
                first_segment = knots[:, :, 1:2, :]  # (B, M, 1, state_dim)
                first_segment_norm = first_segment / (torch.norm(first_segment, dim=-1, keepdim=True) + 1e-6)
                
                prev_vel_expanded = prev_vel_norm.view(B, 1, 1, self.state_dim)
                first_segment_mag = torch.norm(first_segment, dim=-1, keepdim=True)
                
                aligned_direction = 0.8 * prev_vel_expanded + 0.2 * first_segment_norm
                aligned_direction = aligned_direction / (torch.norm(aligned_direction, dim=-1, keepdim=True) + 1e-6)
                aligned_first_segment = aligned_direction * first_segment_mag
                
                knots = torch.cat([knots[:, :, :1, :], aligned_first_segment, knots[:, :, 2:, :]], dim=2)
        
        # INTENT BIAS: Push last knot toward intent
        intent_bias = intent.view(B, 1, self.state_dim)
        knots_last = knots[:, :, -1:, :] + intent_bias.unsqueeze(2) * 0.5
        knots = torch.cat([knots[:, :, :-1, :], knots_last], dim=2)
        
        # Interpolate knots to dense trajectories
        mu_dense_list = []
        sigma_dense_list = []
        
        for b in range(B):
            for m in range(self.n_ports):
                active_mask = knot_mask[b, m] > 0.5
                active_indices = torch.where(active_mask)[0]
                device = active_indices.device if len(active_indices) > 0 else knots.device
                
                if len(active_indices) == 0:
                    active_indices = torch.tensor([0, self.n_knots - 1], device=device)
                else:
                    if not torch.any(active_indices == 0):
                        active_indices = torch.cat([torch.tensor([0], device=device), active_indices])
                    if not torch.any(active_indices == self.n_knots - 1):
                        active_indices = torch.cat([active_indices, torch.tensor([self.n_knots - 1], device=device)])
                    active_indices = torch.unique(torch.sort(active_indices)[0])
                
                # Get active knots
                active_knots = knots[b, m, active_indices, :].unsqueeze(0)  # (1, K_active, state_dim)
                
                # Per-dimension sigma interpolation for true anisotropic affordance tubes
                # Each dimension maintains its own precision throughout the trajectory
                port_sigma = sigmas[b, m]  # (state_dim,) - per-dimension precision
                
                # Interpolate each dimension's sigma separately to preserve anisotropy
                # We'll interpolate sigma values at knot positions, then expand to full trajectory
                # For simplicity, we use constant sigma per dimension (can be improved with per-knot sigmas)
                # Create sigma tensor: (1, K_active, state_dim) - one sigma per dimension per knot
                active_sigmas = port_sigma.unsqueeze(0).unsqueeze(0).expand(1, len(active_indices), self.state_dim)  # (1, K_active, state_dim)
                
                # Interpolate trajectory (CausalSpline expects (B, K, 1) sigmas, so we'll handle per-dim separately)
                # For now, use mean sigma for spline interpolation, then expand per-dimension
                mean_sigma_per_knot = port_sigma.mean().unsqueeze(0).unsqueeze(0).expand(1, len(active_indices), 1)
                mu_dense_port, sigma_dense_mean = CausalSpline.interpolate(
                    active_knots, mean_sigma_per_knot, resolution=self.interp_res
                )
                
                # Expand sigma to per-dimension: maintain anisotropic precision
                # Each dimension gets its own sigma value, creating a hyperellipsoid affordance tube
                T_dense = mu_dense_port.shape[1]
                sigma_expanded = port_sigma.unsqueeze(0).expand(T_dense, self.state_dim)  # (T, state_dim)
                
                # Optionally: interpolate sigma per dimension if we had per-knot sigmas
                # For now, constant per-dimension sigma is simpler and works well
                
                mu_dense_list.append(mu_dense_port.squeeze(0))  # (T, state_dim)
                sigma_dense_list.append(sigma_expanded)  # (T, state_dim)
        
        # Pad to same length
        max_T = max(m.shape[0] for m in mu_dense_list)
        device = mu_dense_list[0].device
        
        mu_t_padded = []
        sigma_t_padded = []
        for mu, sigma in zip(mu_dense_list, sigma_dense_list):
            T = mu.shape[0]
            if T < max_T:
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
        mu_t = torch.stack(mu_t_padded).view(B, self.n_ports, max_T, self.state_dim)
        sigma_t = torch.stack(sigma_t_padded).view(B, self.n_ports, max_T, self.state_dim)
        
        return logits, mu_t, sigma_t, knot_mask, basis_weights

    def select_port(self, logits, mu_t, intent_target):
        """
        Select port based on closest tube endpoint to target, weighted by logits.
        
        Args:
            logits: (B, M) port logits
            mu_t: (B, M, T, state_dim) tube trajectories
            intent_target: (B, state_dim) or (state_dim,) target intent vector
            
        Returns:
            selected_indices: (B,) indices of selected ports
        """
        end_points = mu_t[:, :, -1, :]  # (B, M, state_dim)
        if intent_target.dim() == 1:
            intent_target = intent_target.unsqueeze(0)  # (1, state_dim)
        dist = torch.norm(end_points - intent_target.unsqueeze(1), dim=-1)  # (B, M)
        score = F.log_softmax(logits, dim=-1) - dist
        return torch.argmax(score, dim=-1)
