import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from v3.geometry import CausalSpline


class Actor(nn.Module):
    """
    Fibration Actor: Basis-Projection Actor with optional cross-attention.
    
    Supports both traditional MLP encoding and transformer-based cross-attention.
    When situation_sequence is provided, uses cross-attention to attend to dimension
    sequences from the transformer inference engine.
    
    Key features:
    - Basis function weights instead of raw offsets
    - Per-dimension precision (sigma vector) for anisotropic affordance tubes
    - Dimension-agnostic: infers state_dim from intent.shape
    - Optional cross-attention for transformer-based architecture
    """
    def __init__(self, latent_dim, n_ports=4, n_knots=6, n_basis=8, interp_res=40,
                 token_embed_dim=64, n_attention_heads=8):
        """
        Args:
            latent_dim: Dimension of latent situation space
            n_ports: Number of affordance ports to propose
            n_knots: Number of knots per tube
            n_basis: Number of basis functions for knot generation
            interp_res: Resolution for spline interpolation
            token_embed_dim: Embedding dimension from transformer
            n_attention_heads: Number of attention heads
        """
        super().__init__()
        self.n_ports = n_ports
        self.n_knots = n_knots
        self.n_basis = n_basis
        self.interp_res = interp_res
        self.token_embed_dim = token_embed_dim
        
        # Cross-attention: Actor queries situation transformer keys
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_embed_dim,
            num_heads=n_attention_heads,
            batch_first=True
        )
        
        # Intent projection: maps intent to query space (per-dimension)
        self.intent_proj = nn.Linear(1, token_embed_dim)
        
        # Situation projection: maps latent situation to key/value space
        self.situation_proj = nn.Linear(latent_dim, token_embed_dim)
        
        # Output projection: attended features + intent -> port outputs
        # Intent is included so model can learn to use it for port selection
        # (but we don't explicitly bias - model learns the association)
        self.port_head = nn.Sequential(
            nn.Linear(token_embed_dim + 1, 256),  # +1 for intent norm
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # Port output head: generates logits, basis weights, and sigmas
        max_supported_dim = 12
        self.port_output_head = nn.Linear(256, n_ports * (1 + n_basis + max_supported_dim))
        
        # Learnable intent bias: learns how much to bias knots toward goal
        # Takes intent distance and outputs a bias factor (0 = no bias, 1 = full bias)
        self.intent_bias_head = nn.Sequential(
            nn.Linear(1, 32),  # Intent distance -> hidden
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 1),  # -> bias factor
            nn.Sigmoid()  # Constrain to [0, 1]
        )
        
        # Basis functions: define a set of basis functions for knot generation
        # These represent the "Fibers" - geometric patterns the Actor can select from
        # We initialize with geometric patterns (smooth curves) rather than pure noise
        # Each basis function maps knot_index -> contribution to state_dim coordinates
        # Will be sliced to inferred state_dim in forward()
        basis_init = torch.zeros(n_basis, n_knots, max_supported_dim)
        
        # Initialize with smooth geometric patterns
        for b_idx in range(n_basis):
            # Create smooth patterns using sinusoidal and polynomial bases
            t = torch.linspace(0, 1, n_knots)
            
            # Different patterns for different basis functions
            if b_idx % 3 == 0:
                # Linear patterns in different directions
                dim = b_idx % max_supported_dim
                basis_init[b_idx, :, dim] = t - 0.5  # Centered linear
            elif b_idx % 3 == 1:
                # Sinusoidal patterns
                freq = (b_idx // 3) + 1
                dim = b_idx % max_supported_dim
                basis_init[b_idx, :, dim] = 0.3 * torch.sin(2 * np.pi * freq * t)
            else:
                # Polynomial patterns
                dim = b_idx % max_supported_dim
                basis_init[b_idx, :, dim] = 0.2 * (t - 0.5) ** 2
        
        # Add small random noise for diversity
        basis_init = basis_init + torch.randn(n_basis, n_knots, max_supported_dim) * 0.05
        
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
    
    def forward(self, latent_situation, intent, previous_velocity=None, situation_sequence=None,
                markov_lattice=None, current_pos=None):
        """
        Generate affordance tubes using basis function projection.
        
        Args:
            latent_situation: (B, latent_dim) latent situation from InferenceEngine
            intent: (B, state_dim) intent/target direction tensor - INFERS state_dim from this!
            previous_velocity: Optional (B, state_dim) or (state_dim,) tensor (kept for interface compatibility, not used)
            situation_sequence: (B, state_dim, token_embed_dim) sequence from transformer
                             Required for cross-attention
            markov_lattice: Optional MarkovLattice for look-ahead queries
            current_pos: Optional (state_dim,) current position for look-ahead queries
        
        Returns:
            logits: (B, M) port selection logits
            mu_t: (B, M, T, state_dim) interpolated tube trajectories
            sigma_t: (B, M, T, state_dim) per-dimension precision (sigma vector)
            knot_mask: (B, M, K) knot existence mask
            basis_weights: (B, M, n_basis) basis function weights (for analysis)
        """
        B = latent_situation.size(0)
        
        # INFER state_dim from intent shape
        state_dim = intent.shape[-1]
        
        # Cross-attention: Actor queries situation transformer keys
        # Project intent to queries (one per dimension)
        intent_queries = self.intent_proj(intent.unsqueeze(-1))  # (B, state_dim, token_embed_dim)
        
        # Cross-attend: queries attend to situation sequence
        attended, attention_weights = self.cross_attention(
            query=intent_queries,  # (B, state_dim, token_embed_dim)
            key=situation_sequence,  # (B, state_dim, token_embed_dim)
            value=situation_sequence  # (B, state_dim, token_embed_dim)
        )
        
        # Aggregate attended features (mean or weighted by attention)
        attended_feat = attended.mean(dim=1)  # (B, token_embed_dim)
        
        # Include intent norm in port selection (model learns to use it, but not explicitly biased)
        intent_norm = torch.norm(intent, dim=-1, keepdim=True)  # (B, 1)
        port_input = torch.cat([attended_feat, intent_norm], dim=-1)  # (B, token_embed_dim + 1)
        
        # Project through port head
        situation = self.port_head(port_input)  # (B, 256)
        
        # Generate port outputs
        raw_out = self.port_output_head(situation).view(B, self.n_ports, -1)
        logits = raw_out[:, :, 0]  # (B, M)
        
        # Extract basis weights and precision (sigma) vector
        basis_weights = raw_out[:, :, 1:1+self.n_basis]  # (B, M, n_basis)
        sigma_raw = raw_out[:, :, 1+self.n_basis:1+self.n_basis+state_dim]  # (B, M, state_dim) - slice to inferred dim
        
        # Normalize basis weights to prevent extreme values
        # This ensures the Actor selects from the fiber space smoothly
        basis_weights = torch.tanh(basis_weights)  # Constrain to [-1, 1]
        
        # Per-dimension precision: one sigma per dimension (anisotropic affordance tube)
        # Use exp for stability (log-space precision)
        # Each dimension gets its own precision, allowing hyperellipsoid affordance tubes
        sigmas = torch.exp(sigma_raw) + 0.1  # (B, M, state_dim) - per-dimension precision
        
        # Generate knot existence mask
        knot_existence_logits = self.knot_existence_head(situation).view(B, self.n_ports, self.n_knots)
        knot_mask = torch.sigmoid(knot_existence_logits)  # (B, M, K)
        
        # Note: No forced first/last knots - environment teaches through binding failures
        # If tube doesn't start at current position or reach goal, binding fails
        
        # Generate knots from basis functions
        # Basis functions: (n_basis, n_knots, max_supported_dim) - slice to inferred state_dim
        basis_full = self._generate_basis_functions()  # (n_basis, n_knots, max_supported_dim)
        basis = basis_full[:, :, :state_dim]  # (n_basis, n_knots, state_dim) - slice to inferred dim
        
        # Weighted combination: (B, M, n_basis) @ (n_basis, n_knots, state_dim)
        # Use einsum for efficient batched matrix multiplication
        # 'bmn, nkd -> bmkd' where b=batch, m=port, n=basis, k=knot, d=dim
        knots = torch.einsum('bmn, nkd -> bmkd', basis_weights, basis)  # (B, M, n_knots, state_dim)
        
        # Ensure knots are exactly (B, M, n_knots, state_dim)
        assert knots.shape[-1] == state_dim, f"knots last dim should be {state_dim}, got {knots.shape[-1]}"
        
        # LEARNABLE INTENT BIAS: Model learns how much to bias knots toward goal
        # This is learnable (not fixed) - model discovers optimal bias through training
        # Compute intent distance for bias factor
        intent_distance = torch.norm(intent, dim=-1, keepdim=True)  # (B, 1)
        
        # Learnable bias factor: model learns optimal bias based on distance
        # Close to goal → might bias more/less (model decides)
        # Far from goal → might bias more/less (model decides)
        intent_bias_factor = self.intent_bias_head(intent_distance)  # (B, 1)
        
        # Apply learnable bias to last knot: push toward intent
        # Model learns the optimal bias strength through intent_loss + binding failures
        intent_sliced = intent[:, :state_dim] if intent.shape[-1] > state_dim else intent  # (B, state_dim)
        intent_bias = intent_sliced.view(B, 1, state_dim)  # (B, 1, state_dim)
        
        # Apply learnable bias: knots_last = knots_last + intent_bias * learned_factor
        # The learned factor can be 0 (no bias) to 1 (full bias), or anywhere in between
        knots_last = knots[:, :, -1:, :] + intent_bias.unsqueeze(2) * intent_bias_factor.unsqueeze(-1)  # (B, M, 1, state_dim)
        knots = torch.cat([knots[:, :, :-1, :], knots_last], dim=2)
        
        # Note: No causal anchoring - environment teaches through binding failures
        # - If tube doesn't start at current position → binding failure
        # - Actor learns to start at origin naturally
        
        # Interpolate knots to dense trajectories
        mu_dense_list = []
        sigma_dense_list = []
        
        for b in range(B):
            for m in range(self.n_ports):
                active_mask = knot_mask[b, m] > 0.5
                active_indices = torch.where(active_mask)[0]
                device = active_indices.device if len(active_indices) > 0 else knots.device
                
                # Handle edge cases for interpolation (needs at least 2 knots)
                # If no knots active or only 1, this will cause binding failures
                # Environment teaches actor to use appropriate knots through binding feedback
                if len(active_indices) == 0:
                    # No knots active - use first and last as minimal fallback
                    # This creates a binding failure if tube doesn't start at origin or reach goal
                    active_indices = torch.tensor([0, self.n_knots - 1], device=device)
                elif len(active_indices) == 1:
                    # Only one knot - duplicate it for interpolation (creates zero-length tube)
                    # This will cause binding failure, teaching actor to use more knots
                    active_indices = torch.cat([active_indices, active_indices])
                else:
                    # Multiple knots - use as-is, let environment teach optimal configuration
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
                active_sigmas = port_sigma.unsqueeze(0).unsqueeze(0).expand(1, len(active_indices), state_dim)  # (1, K_active, state_dim)
                
                # Interpolate trajectory (CausalSpline expects (B, K, 1) sigmas, so we'll handle per-dim separately)
                # For now, use mean sigma for spline interpolation, then expand per-dimension
                mean_sigma_per_knot = port_sigma.mean().unsqueeze(0).unsqueeze(0).expand(1, len(active_indices), 1)
                mu_dense_port, sigma_dense_mean = CausalSpline.interpolate(
                    active_knots, mean_sigma_per_knot, resolution=self.interp_res
                )
                
                # Expand sigma to per-dimension: maintain anisotropic precision
                # Each dimension gets its own sigma value, creating a hyperellipsoid affordance tube
                T_dense = mu_dense_port.shape[1]
                sigma_expanded = port_sigma.unsqueeze(0).expand(T_dense, state_dim)  # (T, state_dim)
                
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
        mu_t = torch.stack(mu_t_padded).view(B, self.n_ports, max_T, state_dim)
        sigma_t = torch.stack(sigma_t_padded).view(B, self.n_ports, max_T, state_dim)
        
        # Note: Risk modulation removed - actor learns sigma directly from context
        # The hub graph learns risk patterns, and the transformer/actor learns to associate
        # patterns with appropriate sigma values through the situation_sequence attention.
        # The actor's sigma predictions already incorporate obstacle information via the
        # transformer's dimension-token sequence (which includes obstacle directions).
        # No post-hoc modulation needed - let gradients flow and the model learn.
        
        return logits, mu_t, sigma_t, knot_mask, basis_weights

    def compute_binding_loss(self, proposed_tube, actual_path, sigma_t, current_pos, knot_mask=None):
        """
        Compute principled TAM loss based on binding failure and trajectory geometry.
        
        This is the core TAM loss: binding succeeds when actual_path stays within
        the affordance cone (sigma-weighted tube), fails otherwise.
        
        Additionally computes geometry-based costs for path complexity:
        - Knot count: More control points = more commitments = more complexity
        
        All costs are dimension-agnostic and environment-agnostic.
        
        Args:
            proposed_tube: (T, state_dim) relative tube trajectory
            actual_path: (T_actual, state_dim) actual path taken from world
            sigma_t: (T, state_dim) per-dimension precision (cone width)
            current_pos: (state_dim,) or (1, state_dim) current position
            knot_mask: Optional (K,) tensor indicating active knots (for knot count)
            
        Returns:
            binding_loss: Scalar tensor - weighted deviation from affordance cone
            agency_cost: Scalar tensor - cone width cost (sigma^2)
            geometry_cost: Scalar tensor - knot count + direction change costs
        """
        # Ensure current_pos is (state_dim,)
        if current_pos.dim() > 1:
            current_pos = current_pos.squeeze()
        
        device = proposed_tube.device
        state_dim = proposed_tube.shape[-1]
        
        # Topological binding: verify actual_path stays within affordance cone
        min_len = min(len(actual_path), len(proposed_tube))
        if min_len == 0:
            return (torch.tensor(0.0, device=device), 
                   torch.tensor(0.0, device=device),
                   torch.tensor(0.0, device=device))
        
        # Expected path: proposed tube in global coordinates
        expected_p_slice = proposed_tube[:min_len] + current_pos  # (T, state_dim)
        actual_p_slice = actual_path[:min_len]  # (T, state_dim)
        
        # Per-dimension deviation from expected path
        deviation_per_dim = (actual_p_slice - expected_p_slice)  # (T, state_dim)
        
        # Weight by per-dimension precision (sigma)
        # Higher sigma = wider cone = more tolerance = lower penalty for deviation
        # Lower sigma = narrower cone = less tolerance = higher penalty for deviation
        # This implements the affordance cone: deviation weighted by cone width
        sigma_slice = sigma_t[:min_len]  # (T, state_dim)
        weighted_deviation = (deviation_per_dim**2) / (sigma_slice + 1e-6)  # (T, state_dim)
        
        # Binding loss: average weighted deviation (binding failure measure)
        binding_loss = torch.mean(weighted_deviation.sum(dim=-1))
        
        # Agency cost: cone width (sigma) - narrower cones = higher agency
        # This is dimension-agnostic: works for any state_dim
        agency_cost = torch.mean(sigma_t**2)  # Mean across all dimensions and time
        
        # Geometry cost removed - environment teaches knot count through binding failures
        # - Too many knots → more opportunities for binding failures → actor learns to use fewer
        # - Too few knots → can't navigate complex paths → actor learns to use more
        # - Optimal knot count emerges naturally from binding feedback
        geometry_cost = torch.tensor(0.0, device=device)
        
        return binding_loss, agency_cost, geometry_cost
    
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
