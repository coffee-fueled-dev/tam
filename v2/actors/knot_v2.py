import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple

def _lin_interp_knots_to_T(knots: torch.Tensor, T: int) -> torch.Tensor:
    """Differentiable linear interpolation from n_knots to T timesteps."""
    B, n_knots, d = knots.shape
    if n_knots == 1:
        return knots.expand(B, T, d)

    device, dtype = knots.device, knots.dtype
    knot_pos = torch.linspace(0.0, 1.0, n_knots, device=device, dtype=dtype)
    t_pos = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    # Find indices for interpolation
    idx = torch.searchsorted(knot_pos, t_pos, right=True) - 1
    idx = idx.clamp(0, n_knots - 2)

    i0, i1 = idx, idx + 1
    p0, p1 = knot_pos[i0], knot_pos[i1]

    # Calculate weights
    w = ((t_pos - p0) / (p1 - p0).clamp(min=1e-8)).view(1, T, 1).expand(B, T, d)
    
    # Gather and interpolate
    k0 = knots[:, i0, :]
    k1 = knots[:, i1, :]
    return (1.0 - w) * k0 + w * k1

class GeometricKnotActor(nn.Module):
    """
    A Rational Actor that represents agency as a geometric tube.
    Uses Hinge Loss to enforce the 'Binding' contract.
    """
    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        pred_dim: int = 2,
        T: int = 16,
        n_knots: int = 5,
        alpha_vol: float = 0.1,    # Pressure to shrink the tube (Agency)
        alpha_leak: float = 10.0,  # Penalty for trajectories exiting the tube
        k_sigma: float = 1.0,       # The 'Wall' of the tube (in sigma units)
        s0_emb_dim: int = 8        # Architectural bottleneck: obs embedding dimension
    ):
        super().__init__()
        self.T, self.n_knots = T, n_knots
        self.k_sigma = k_sigma
        self.alpha_vol, self.alpha_leak = alpha_vol, alpha_leak
        self.s0_emb_dim = s0_emb_dim
        self.obs_dim = obs_dim
        self.pred_dim = pred_dim
        self.z_dim = z_dim

        # Encoder MLP: maps FLATTENED trajectory delta to raw z
        # Takes full trajectory shape (T * pred_dim) to preserve temporal signature
        self.encoder_mlp = nn.Sequential(
            nn.Linear(T * pred_dim, 256),  # Full flattened trajectory
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim)
            # No activation - L2 normalization applied in encode()
        )

        # Architectural Bottleneck: Shallow obs encoder, Deep z encoder
        # Shallow obs encoder: single linear layer (8 units)
        self.obs_proj = nn.Linear(obs_dim, s0_emb_dim)
        
        # Deep z encoder: wide and deep MLP (512 -> 256 -> z_dim)
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # mu knots: Concatenate narrow obs (8) + wide z (256) -> Path Center
        self.mu_net = nn.Sequential(
            nn.Linear(s0_emb_dim + 256, 64),  # 8 (obs) + 256 (z) = 264 total
            nn.ReLU(),
            nn.Linear(64, n_knots * pred_dim)
        )
        
        # sigma knots: Choice ONLY -> Tube Width
        # This forces the agent to commit to a 'precision' regardless of the world state
        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_knots * pred_dim)
        )

    def encode(self, traj_flat: torch.Tensor) -> torch.Tensor:
        """Encode trajectory to latent space with L2 normalization."""
        raw_z = self.encoder_mlp(traj_flat)
        # L2 normalization projects onto unit hypersphere (avoids edge saturation)
        if raw_z.dim() == 1:
            return F.normalize(raw_z.unsqueeze(0), p=2, dim=1).squeeze(0)
        return F.normalize(raw_z, p=2, dim=1)

    def get_tube(self, s0: torch.Tensor, z: torch.Tensor, force_z_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get tube prediction from observation and latent commitment.
        
        Args:
            s0: Initial observation
            z: Latent commitment
            force_z_only: If True, zero out obs embedding to test z-dependency
        """
        # Handle z dimension: add batch dimension if needed
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]
        
        # Expand s0 to match batch of z candidates
        if s0.dim() == 1:
            s_expanded = s0.view(1, -1).expand(B, -1)
        elif s0.shape[0] == 1:
            s_expanded = s0.expand(B, -1)
        else:
            s_expanded = s0
        
        # Shallow obs encoder: single linear layer
        s_proj = self.obs_proj(s_expanded)
        
        # Zeroing diagnostic: replace obs embedding with zeros to test z-dependency
        if force_z_only:
            s_proj = torch.zeros_like(s_proj)
        
        # Deep z encoder: wide and deep MLP
        z_encoded = self.z_encoder(z)
        
        # Concatenate narrow obs (8) + wide z (256)
        h = torch.cat([s_proj, z_encoded], dim=-1)
        
        mu_k = self.mu_net(h).view(B, self.n_knots, -1)
        # Softplus ensures width is always positive but can shrink toward zero
        sigma_k = torch.nn.functional.softplus(self.sigma_net(z).view(B, self.n_knots, -1))
        
        mu = _lin_interp_knots_to_T(mu_k, self.T)
        sigma = _lin_interp_knots_to_T(sigma_k, self.T)
        return mu, sigma

    def train_step(self, s0: torch.Tensor, trajectory: torch.Tensor, z: torch.Tensor, optimizer: optim.Optimizer, force_z_only: bool = False):
        self.train()
        mu, sigma = self.get_tube(s0, z.unsqueeze(0), force_z_only=force_z_only)
        mu, sigma = mu.squeeze(0), sigma.squeeze(0)
        
        # --- 1. TIGHT CONTRACT: Sigma should match actual deviation ---
        dist = torch.abs(trajectory - mu)  # (T, pred_dim)
        
        # Leakage: Penalize if sigma is too small (the "Lie")
        leakage = torch.relu(dist - sigma * self.k_sigma).pow(2).mean()
        
        # Over-estimation: Penalize if sigma is too large (the "Lazy" factor)
        over_estimation = torch.relu(sigma - dist * 1.2).mean()
        
        # --- 2. AGENCY PRESSURE (Volume Minimization) ---
        volume = sigma.mean()
        
        # --- 3. START ANCHOR: Tube must begin at current state ---
        s0_pos = s0[:self.pred_dim]
        start_error = (mu[0] - s0_pos).pow(2).mean()
        
        # --- 4. ENDPOINT DIRECTION: Tube must go where trajectory went ---
        # This forces the tube network to respect z → direction mapping
        traj_direction = trajectory[-1] - trajectory[0]  # Where trajectory actually went
        tube_direction = mu[-1] - mu[0]  # Where tube predicts going
        # Cosine similarity: should be 1.0 if directions match
        direction_sim = F.cosine_similarity(
            traj_direction.unsqueeze(0), 
            tube_direction.unsqueeze(0)
        )
        direction_loss = (1.0 - direction_sim).mean()  # 0 if perfect match, 2 if opposite
        
        # --- 5. ENDPOINT DISTANCE: Tube endpoint should match trajectory endpoint ---
        endpoint_error = (mu[-1] - trajectory[-1]).pow(2).mean()
        
        # The 'Tight Contract': 
        loss = (
            self.alpha_leak * leakage + 
            0.5 * over_estimation + 
            self.alpha_vol * volume + 
            5.0 * start_error +
            2.0 * direction_loss +  # Force z → direction mapping
            1.0 * endpoint_error    # Endpoint accuracy
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            "loss": loss.item(), 
            "leak": leakage.item(), 
            "lazy": over_estimation.item(),
            "vol": volume.item(),
            "start_err": start_error.item(),
            "dir_loss": direction_loss.item(),
            "end_err": endpoint_error.item()
        }
    
    def supervised_contrastive_loss(self, z_batch: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
        """
        Supervised Contrastive Loss (SupCon) for mode separation.
        
        Args:
            z_batch: Batch of z embeddings (batch_size, z_dim)
            labels: Mode labels for each z (batch_size,)
            temperature: Temperature scaling parameter
        
        Returns:
            Contrastive loss value
        """
        if z_batch.shape[0] < 2:
            return torch.tensor(0.0, device=z_batch.device)
        
        # Normalize z to the hypersphere to stabilize distances
        z = F.normalize(z_batch, dim=1)
        
        # Calculate cosine similarity matrix
        logits = torch.div(torch.matmul(z, z.T), temperature)
        
        # Mask to identify same-mode pairs (positives)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove self-similarity (diagonal)
        mask = mask * (1 - torch.eye(z.shape[0], device=z.device))
        
        # Standard SupCon / InfoNCE logic:
        # We want to maximize the similarity of positive pairs
        # and minimize the similarity of negative pairs.
        exp_logits = torch.exp(logits) * (1 - torch.eye(z.shape[0], device=z.device))
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        
        # Mean log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mask_sum = mask_sum + (mask_sum == 0).float()  # Avoid division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        return -mean_log_prob_pos.mean()
    
    def train_encoder_batch(self, traj_flat_batch: torch.Tensor, z_batch: torch.Tensor, 
                            labels_batch: torch.Tensor, optimizer: optim.Optimizer, epoch: int = 0):
        """
        Train encoder on a full batch with contrastive loss.
        
        Args:
            traj_flat_batch: Flattened trajectories (batch_size, T*pred_dim)
            z_batch: Pre-computed z embeddings (batch_size, z_dim)
            labels_batch: Mode labels (batch_size,)
            optimizer: Encoder optimizer
            epoch: Current epoch for dynamic weighting
        """
        self.train()
        batch_size = traj_flat_batch.shape[0]
        
        # Forward pass: get z embeddings with gradients
        z = self.encode(traj_flat_batch)  # (batch_size, z_dim)
        
        # Dynamic loss weighting
        if epoch < 1000:
            recon_weight = 0.05
            intent_weight = 1.0
        else:
            progress = min(1.0, (epoch - 1000) / 500.0)
            recon_weight = 0.05 + 0.45 * progress
            intent_weight = 1.0 - 0.3 * progress
        
        # Reconstruction loss: predict trajectory from z (sample a few from batch)
        recon_loss = torch.tensor(0.0, device=z.device)
        for i in range(min(4, batch_size)):  # Sample up to 4 for efficiency
            dummy_s0 = torch.zeros(1, self.obs_dim, device=z.device)
            mu, sigma = self.get_tube(dummy_s0, z[i:i+1])
            traj_delta = traj_flat_batch[i].reshape(self.T, self.pred_dim)
            recon_loss = recon_loss + F.mse_loss(mu.squeeze(0), traj_delta)
        recon_loss = recon_loss / min(4, batch_size)
        
        # Supervised Contrastive Loss on FULL batch
        # Higher temperature (0.5) = softer contrastive (prevents saturation to corners)
        intent_loss = self.supervised_contrastive_loss(z, labels_batch, temperature=0.5)
        
        # With L2 normalization, z is on unit sphere - no need for diversity/L2 reg
        # For monitoring purposes, track z_std (for unit vectors, ~0.7 is well-spread)
        z_std = z.std()
        
        # Combined loss (simpler with L2 norm - no saturation issues)
        loss = recon_weight * recon_loss + intent_weight * intent_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            "loss": loss.item(),
            "recon": recon_loss.item(),
            "intent": intent_loss.item(),
            "z_std": z_std.item(),
            "recon_weight": recon_weight,
            "intent_weight": intent_weight,
        }
    
    def train_encoder_step(self, trajectory_delta: torch.Tensor, optimizer: optim.Optimizer, 
                          mode_label: int = None, z_batch: torch.Tensor = None, labels_batch: torch.Tensor = None,
                          epoch: int = 0, recon_weight: float = None, intent_weight: float = None):
        """
        Z-Warmup: Train encoder only to create stable latent mapping.
        Uses trajectory delta (traj - s0) to focus on movement shape, not absolute position.
        
        Args:
            trajectory_delta: Trajectory relative to start (traj - s0)
            optimizer: Optimizer for encoder
            mode_label: Ground truth mode label for contrastive loss (single sample)
            z_batch: Batch of z embeddings for contrastive loss (if available)
            labels_batch: Batch of mode labels for contrastive loss (if available)
            epoch: Current epoch for dynamic loss weighting
            recon_weight: Weight for reconstruction loss (if None, uses dynamic weighting)
            intent_weight: Weight for intent/contrastive loss (if None, uses dynamic weighting)
        """
        self.train()
        
        # FLATTEN the trajectory delta to preserve temporal signature
        # This allows the encoder to see the "shape" of the knot, not just its average
        traj_flat = trajectory_delta.reshape(-1)  # (T * pred_dim,)
        z = self.encode(traj_flat)
        
        # Delta-Traj Sanity Check: Verify trajectory_delta doesn't contain absolute coordinates
        traj_delta_abs_max = trajectory_delta.abs().max().item()
        
        # Reconstruction loss: predict trajectory from z
        dummy_s0 = torch.zeros(1, self.obs_dim, device=z.device)
        mu, sigma = self.get_tube(dummy_s0, z.unsqueeze(0))
        mu, sigma = mu.squeeze(0), sigma.squeeze(0)
        
        # Reconstruction loss: encoder should map to z that predicts trajectory well
        recon_loss = F.mse_loss(mu, trajectory_delta)
        
        # Dynamic loss weighting: KEEP RECON LOW for first 1000 epochs
        # Strategy: Force latent space organization before physics learning
        if recon_weight is None:
            if epoch < 1000:
                # Epoch 0-1000: Very low recon (0.05), High intent (1.0) - Force clustering first
                recon_weight = 0.05  # Minimal reconstruction pressure
                intent_weight_val = 1.0 if intent_weight is None else intent_weight
            else:
                # Epoch 1000+: Gradually increase recon weight
                progress = min(1.0, (epoch - 1000) / 500.0)
                recon_weight = 0.05 + 0.45 * progress  # Linear annealing: 0.05 -> 0.5
                intent_weight_val = 1.0 - 0.3 * progress if intent_weight is None else intent_weight  # 1.0 -> 0.7
        else:
            recon_weight = recon_weight
            intent_weight_val = intent_weight if intent_weight is not None else 1.0
        
        # Supervised Contrastive Loss: Use batch if available
        intent_loss = torch.tensor(0.0, device=z.device)
        if z_batch is not None and labels_batch is not None and len(z_batch) > 1:
            # Add current z to batch for contrastive loss
            z_batch_with_current = torch.cat([z_batch, z.unsqueeze(0)], dim=0)
            labels_batch_with_current = torch.cat([labels_batch, torch.tensor([mode_label], device=z.device)], dim=0)
            intent_loss = self.supervised_contrastive_loss(z_batch_with_current, labels_batch_with_current, temperature=0.1)
        
        # With L2 normalization, z is on unit sphere - diversity is natural
        z_std = z.std()
        
        # Combined loss (simpler with L2 norm)
        loss = recon_weight * recon_loss + intent_weight_val * intent_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return {
            "loss": loss.item(), 
            "recon": recon_loss.item(), 
            "intent": intent_loss.item() if isinstance(intent_loss, torch.Tensor) else intent_loss,
            "z_std": z_std.item(),
            "recon_weight": recon_weight,
            "intent_weight": intent_weight_val,
            "traj_delta_abs_max": traj_delta_abs_max
        }

    @torch.no_grad()
    def select_z_geometric(self, s0, trajectory_delta=None, intent_prototype=None, pop_size=128, n_iters=10, force_z_only=False):
        """
        CEM search for z that balances:
        1. Fit to intent (if provided)
        2. Minimal volume (Agency)
        3. Prior probability (staying near the encoder's hint)
        
        Args:
            s0: Initial observation
            trajectory_delta: Optional trajectory delta (traj - s0) for encoder hint (T, pred_dim)
            intent_prototype: Optional target trajectory
            pop_size: Population size for CEM (default: 128)
            n_iters: Number of CEM iterations (default: 10)
            force_z_only: If True, zero out obs embedding to test z-dependency
        """
        # Get encoder hint from trajectory delta if provided
        if trajectory_delta is not None:
            traj_flat = trajectory_delta.reshape(-1)  # (T * pred_dim,)
            z_hint = self.encode(traj_flat)
            # Start near hint with moderate exploration
            curr_mean = z_hint.clone()
            curr_std = torch.ones_like(z_hint) * 0.3
        else:
            # No hint: Start from origin with wide exploration
            # This allows CEM to discover intent through volume minimization
            curr_mean = torch.zeros(self.z_dim, device=s0.device)
            curr_std = torch.ones(self.z_dim, device=s0.device) * 0.7  # Wide exploration
        
        # Extract state position from observation (first pred_dim elements)
        s0_pos = s0[:self.pred_dim]
        
        for _ in range(n_iters):
            candidates = curr_mean + torch.randn(pop_size, self.z_dim, device=s0.device) * curr_std
            # L2 normalize candidates to stay on unit sphere (matching encoder output)
            candidates = F.normalize(candidates, p=2, dim=1)
            mu, sigma = self.get_tube(s0.unsqueeze(0), candidates, force_z_only=force_z_only)
            
            # === TIGHT CONTRACT CEM SCORING ===
            
            # 1. START-POINT ANCHOR (Massive penalty for tubes that don't start at s0)
            # This is the most important contract: you must begin where you are
            start_error = torch.norm(mu[:, 0, :] - s0_pos, dim=-1)
            start_penalty = -torch.pow(start_error, 2) * 100.0  # Massive penalty
            
            # 2. PROGRESS CHECK (Penalize "do-nothing" tubes)
            # A valid commitment must actually go somewhere
            displacement = torch.norm(mu[:, -1, :] - mu[:, 0, :], dim=-1)
            progress_reward = torch.log(displacement + 1e-6)  # Log scale rewards movement
            
            # 3. AGENCY (Volume) - Lower volume = more precise commitment
            # But don't let this dominate - precision without accuracy is useless
            agency_score = -sigma.mean(dim=(1,2)) * 0.3  # Reduced weight
            
            # 4. RELIABILITY (Tight tubes are more reliable if they're accurate)
            # Reward tubes where sigma is consistent (not wildly varying)
            sigma_consistency = -sigma.std(dim=(1,2))
            
            # If we have an intent (a target path), score fit to it
            intent_score = 0
            if intent_prototype is not None:
                intent_score = -torch.pow(mu - intent_prototype, 2).mean(dim=(1,2))
            
            # TOTAL: Start anchor dominates, then progress, then volume
            total_score = start_penalty + progress_reward * 3.0 + agency_score + sigma_consistency + (intent_score * 2.0)
            
            # Update CEM distribution (keep top 10% as elites)
            n_elites = max(8, pop_size // 10)
            elites = candidates[total_score.argsort(descending=True)[:n_elites]]
            curr_mean = elites.mean(dim=0)
            curr_std = elites.std(dim=0) + 1e-6  # Add small epsilon for stability
            
        return elites[0] # Return the most rational commitment
    
    def count_parameters(self):
        """Count parameters in obs vs z pathways for logging."""
        obs_params = sum(p.numel() for p in self.obs_proj.parameters())
        z_params = sum(p.numel() for p in self.z_encoder.parameters())
        total_params = obs_params + z_params
        return {
            "obs_params": obs_params,
            "z_params": z_params,
            "total_params": total_params,
            "z_to_obs_ratio": z_params / (obs_params + 1e-8)
        }