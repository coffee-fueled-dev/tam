import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Tuple, Optional


def _lin_interp_knots_to_T(knots: torch.Tensor, T: int) -> torch.Tensor:
    """Differentiable linear interpolation from n_knots to T timesteps."""
    B, n_knots, d = knots.shape
    if n_knots == 1:
        return knots.expand(B, T, d)

    device, dtype = knots.device, knots.dtype
    knot_pos = torch.linspace(0.0, 1.0, n_knots, device=device, dtype=dtype)
    t_pos = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)

    idx = torch.searchsorted(knot_pos, t_pos, right=True) - 1
    idx = idx.clamp(0, n_knots - 2)

    i0, i1 = idx, idx + 1
    p0, p1 = knot_pos[i0], knot_pos[i1]

    w = ((t_pos - p0) / (p1 - p0).clamp(min=1e-8)).view(1, T, 1).expand(B, T, d)

    k0 = knots[:, i0, :]
    k1 = knots[:, i1, :]
    return (1.0 - w) * k0 + w * k1

class FIFOQueue:
    """
    Simple MoCo-style feature queue for InfoNCE negatives.
    Stores normalized vectors.
    """
    def __init__(self, dim: int, size: int, device: torch.device):
        self.dim = dim
        self.size = int(size)
        self.device = device
        self.buf = torch.zeros(self.size, self.dim, device=self.device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor):
        # Clone and normalize to avoid any reference or in-place issues
        x_norm = F.normalize(x.detach().clone(), dim=-1)
        n = x_norm.shape[0]
        if n >= self.size:
            self.buf.copy_(x_norm[-self.size :].clone())
            self.ptr = 0
            self.full = True
            return
        end = self.ptr + n
        if end <= self.size:
            self.buf[self.ptr:end].copy_(x_norm.clone())
        else:
            first = self.size - self.ptr
            self.buf[self.ptr:].copy_(x_norm[:first].clone())
            self.buf[: end - self.size].copy_(x_norm[first:].clone())
        self.ptr = end % self.size
        if self.ptr == 0:
            self.full = True

    def get(self) -> torch.Tensor:
        # Return a clone to avoid in-place modification issues when used in computation graph
        if self.full:
            return self.buf.clone()
        return self.buf[: self.ptr].clone()


class Actor(nn.Module):
    """
    Binding-Failure-Driven Geometric Knot Actor.

    Core philosophy: The actor learns from **binding failure** - when reality exits
    the predicted affordance tube. This replaces custom losses with a single
    geometric principle: the tube must contain reality.

    Learning signal:
      1) Binding Loss: Penalizes when actual trajectory exits the tube boundary.
      2) Volume Loss: Rewards narrow tubes (high agency) - creates pressure to commit.
      
    The tension: Too narrow → binding failures. Too wide → low agency.
    The ideal: A tube *just wide enough* to contain reality.

    Architecture:
      - Multimodal proposal q(z|s0) with M heads on unit sphere.
      - Winner-take-all (WTA) training: best head (lowest binding failure) gets updated.
      - Necessity-gated repulsion: heads diversify only when binding failures differ across heads.
      - InfoNCE encoder for unsupervised z-space structure.
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int,
        pred_dim: int = 2,
        T: int = 16,
        n_knots: int = 5,

        # Contract weights
        alpha_vol: float = 0.1,
        alpha_leak: float = 10.0,
        k_sigma: float = 1.0,
        sigma_min: float = 0.05,  # Minimum sigma to prevent NLL from going unbounded negative

        # Bottleneck - auto-scales with pred_dim if None
        s0_emb_dim: int = None,

        # Multimodal proposal (ports)
        M: int = 8,                 # number of proposal heads (constant)
        L_refine: int = 2,          # refine top-L heads with short CEM

        # Necessity-gated repulsion
        regret_delta: float = 0.05, # gate threshold on regret proxy
        repel_tau: float = 0.25,    # cosine similarity margin; smaller -> more separation
        repel_weight: float = 0.2,  # repulsion strength when gate is on
        repel_min: float = 0.05,    # always-on minimum repulsion (prevents early collapse)

        # Encoder self-supervision
        info_nce_temp: float = 0.2,
        queue_size: int = 1024,
        aug_noise_std: float = 0.01,
        aug_time_crop_min_frac: float = 0.6,  # crop length fraction range [min, 1]
        aug_time_jitter: int = 2,             # +/- jitter in timesteps

        # CEM defaults for inference
        cem_pop_size: int = 96,
        cem_iters: int = 4,
        cem_init_std: float = 0.25,
        cem_elite_frac: float = 0.1,

        # Optional reseed
        reseed_enabled: bool = False,
        reseed_patience: int = 200,
        reseed_min_usage: float = 0.02,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.pred_dim = pred_dim
        self.z_dim = z_dim
        self.T, self.n_knots = T, n_knots

        self.alpha_vol = alpha_vol
        self.alpha_leak = alpha_leak
        self.k_sigma = k_sigma
        self.sigma_min = float(sigma_min)
        
        # Auto-scale s0_emb_dim with dimension for better high-d scaling
        if s0_emb_dim is None:
            s0_emb_dim = min(128, max(32, pred_dim * 2))
        self.s0_emb_dim = s0_emb_dim

        # Multimodal proposal params
        self.M = int(M)
        self.L_refine = int(min(L_refine, M))
        self.regret_delta = float(regret_delta)
        self.repel_tau = float(repel_tau)
        self.repel_weight = float(repel_weight)
        self.repel_min = float(repel_min)

        # Encoder SSL params
        self.info_nce_temp = float(info_nce_temp)
        self.aug_noise_std = float(aug_noise_std)
        self.aug_time_crop_min_frac = float(aug_time_crop_min_frac)
        self.aug_time_jitter = int(aug_time_jitter)

        # CEM params
        self.cem_pop_size = int(cem_pop_size)
        self.cem_iters = int(cem_iters)
        self.cem_init_std = float(cem_init_std)
        self.cem_elite_frac = float(cem_elite_frac)

        # Optional reseed
        self.reseed_enabled = bool(reseed_enabled)
        self.reseed_patience = int(reseed_patience)
        self.reseed_min_usage = float(reseed_min_usage)
        self.register_buffer("head_usage_ema", torch.ones(self.M) / self.M)
        self.register_buffer("steps_since_reseed", torch.zeros(1, dtype=torch.long))
        
        # Fixed integer knot times: situation-indexed (knot i = time t_i)
        # Ensures "knot i means time t_i" in a stable, causal way
        knot_times = torch.round(torch.linspace(0, self.T - 1, self.n_knots)).long()
        knot_times[0] = 0
        knot_times[-1] = self.T - 1
        # Ensure strictly increasing (handle edge cases)
        if torch.any(knot_times[1:] <= knot_times[:-1]):
            # Fallback: deterministic spacing
            knot_times = torch.linspace(0, self.T - 1, self.n_knots).floor().long()
            knot_times[0] = 0
            knot_times[-1] = self.T - 1
            # Remove duplicates if any
            knot_times = torch.unique_consecutive(knot_times)
        self.register_buffer("knot_times", knot_times)

        # --------- Encoder: trajectory_delta -> z (unit sphere) ---------
        enc_hidden = max(256, pred_dim * 8)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(T * pred_dim, enc_hidden),
            nn.LayerNorm(enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, enc_hidden // 2),
            nn.ReLU(),
            nn.Linear(enc_hidden // 2, z_dim),
        )

        # --------- Bottleneck encoders ---------
        self.obs_proj = nn.Linear(obs_dim, s0_emb_dim)
        z_enc_hidden = max(256, z_dim * 16)
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, z_enc_hidden),
            nn.LayerNorm(z_enc_hidden),
            nn.ReLU(),
            nn.Linear(z_enc_hidden, 256),
            nn.ReLU(),
        )

        # --------- Tube decoders (capacity scales with pred_dim) ---------
        dec_hidden = max(256, pred_dim * 4)
        self.mu_net = nn.Sequential(
            nn.Linear(s0_emb_dim + 256, dec_hidden),
            nn.LayerNorm(dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, n_knots * pred_dim),
        )

        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim + s0_emb_dim, dec_hidden),  # Now also conditioned on s0
            nn.LayerNorm(dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, n_knots * pred_dim),
        )

        # --------- Multimodal proposal q(z|s0) ---------
        router_hidden = max(128, s0_emb_dim * 2)
        self.z_router = nn.Sequential(
            nn.Linear(s0_emb_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.M * z_dim),
        )
        self.z_logits = nn.Sequential(
            nn.Linear(s0_emb_dim, router_hidden),
            nn.ReLU(),
            nn.Linear(router_hidden, self.M),
        )

        # InfoNCE queue is created lazily once we know device
        self._queue_size = int(queue_size)
        self._queue: Optional[FIFOQueue] = None

    # -----------------------------
    # Utilities
    # -----------------------------
    def _ensure_queue(self, device: torch.device):
        if self._queue is None or self._queue.device != device:
            self._queue = FIFOQueue(dim=self.z_dim, size=self._queue_size, device=device)

    def encode(self, traj_flat: torch.Tensor) -> torch.Tensor:
        """Encode trajectory delta (flattened) to latent z on unit hypersphere."""
        raw_z = self.encoder_mlp(traj_flat)
        if raw_z.dim() == 1:
            return F.normalize(raw_z.unsqueeze(0), p=2, dim=1).squeeze(0)
        return F.normalize(raw_z, p=2, dim=1)

    def _augment_traj_delta(self, traj_delta: torch.Tensor) -> torch.Tensor:
        """
        Structure-preserving augmentation for trajectory deltas.
        traj_delta: (T, pred_dim)
        """
        T, d = traj_delta.shape
        # Clone to avoid in-place operations on input tensor
        x = traj_delta.clone()

        # 1) time crop (keep contiguous window)
        if self.aug_time_crop_min_frac < 1.0:
            min_len = max(2, int(math.ceil(T * self.aug_time_crop_min_frac)))
            crop_len = torch.randint(low=min_len, high=T + 1, size=(1,), device=x.device).item()
            start = torch.randint(low=0, high=T - crop_len + 1, size=(1,), device=x.device).item()
            crop = x[start : start + crop_len].clone()  # Clone to avoid view issues

            # resample back to length T by linear interpolation in time
            t_old = torch.linspace(0, 1, crop_len, device=x.device, dtype=x.dtype)
            t_new = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype)
            # interpolate each dim (manual linear interpolation for compatibility)
            crop_resamp = []
            for j in range(d):
                # Manual linear interpolation
                values_old = crop[:, j]
                # Find indices for interpolation
                indices = torch.searchsorted(t_old, t_new, right=True) - 1
                indices = indices.clamp(0, crop_len - 2)
                i0, i1 = indices, indices + 1
                t0, t1 = t_old[i0], t_old[i1]
                w = (t_new - t0) / (t1 - t0).clamp(min=1e-8)
                v0, v1 = values_old[i0], values_old[i1]
                crop_resamp.append((1 - w) * v0 + w * v1)
            x = torch.stack(crop_resamp, dim=-1)

        # 2) time jitter (roll)
        if self.aug_time_jitter > 0 and T > 2:
            shift = torch.randint(-self.aug_time_jitter, self.aug_time_jitter + 1, (1,), device=x.device).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

        # 3) small gaussian noise
        if self.aug_noise_std > 0:
            x = x + torch.randn_like(x) * self.aug_noise_std

        return x

    def _info_nce_loss(self, z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        NT-Xent / InfoNCE with optional queue negatives.
        z1, z2: (B, D), already normalized or will be normalized here.
        """
        # Clone BEFORE normalizing to avoid in-place modification of computation graph tensors
        z1_norm = F.normalize(z1.clone(), dim=-1)
        z2_norm = F.normalize(z2.clone(), dim=-1)
        B, D = z1_norm.shape

        # positives: z1[i] with z2[i]
        pos = (z1_norm * z2_norm).sum(dim=-1, keepdim=True)  # (B, 1)

        # negatives: within-batch + queue
        logits_neg = []

        # within-batch negatives against z2
        sim_12 = z1_norm @ z2_norm.t()  # (B, B)
        mask = ~torch.eye(B, device=z1_norm.device, dtype=torch.bool)
        logits_neg.append(sim_12[mask].view(B, B - 1))

        # queue negatives (get queue before computing loss to avoid any issues)
        self._ensure_queue(z1_norm.device)
        q = self._queue.get()  # (Q, D) or empty
        if q.numel() > 0:
            logits_neg.append(z1_norm @ q.t())  # (B, Q)

        neg = torch.cat(logits_neg, dim=1) if len(logits_neg) > 0 else torch.empty(B, 0, device=z1_norm.device)

        # assemble logits: [pos | neg]
        logits = torch.cat([pos, neg], dim=1) / max(1e-8, temperature)
        labels = torch.zeros(B, dtype=torch.long, device=z1_norm.device)  # positive at index 0
        loss = F.cross_entropy(logits, labels)

        # update queue with z2 (stop-grad, clone to avoid any reference issues)
        # Do this AFTER computing loss to avoid any gradient issues
        with torch.no_grad():
            self._queue.enqueue(z2_norm.detach().clone())

        return loss

    # -----------------------------
    # Multimodal proposal q(z|s0)
    # -----------------------------
    def propose_z_modes(self, s0: torch.Tensor, force_z_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          z_modes: (B, M, z_dim) on unit sphere
          pi:      (B, M) mixture weights (softmax)
        """
        if s0.dim() == 1:
            s0 = s0.unsqueeze(0)
        B = s0.shape[0]

        s_proj = self.obs_proj(s0)  # (B, s0_emb_dim)
        if force_z_only:
            s_proj = torch.zeros_like(s_proj)

        z_raw = self.z_router(s_proj).view(B, self.M, self.z_dim)
        z_modes = F.normalize(z_raw, p=2, dim=-1)

        pi_logits = self.z_logits(s_proj)
        pi = F.softmax(pi_logits, dim=-1)

        return z_modes, pi

    # -----------------------------
    # Tube
    # -----------------------------
    def get_tube(self, s0: torch.Tensor, z: torch.Tensor, force_z_only: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        s0: (obs_dim,) or (B, obs_dim)
        z:  (z_dim,) or (B, z_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        B = z.shape[0]

        if s0.dim() == 1:
            s_expanded = s0.view(1, -1).expand(B, -1)
        elif s0.shape[0] == 1:
            s_expanded = s0.expand(B, -1)
        else:
            s_expanded = s0

        s_proj = self.obs_proj(s_expanded)
        if force_z_only:
            s_proj = torch.zeros_like(s_proj)

        z_encoded = self.z_encoder(z)
        h = torch.cat([s_proj, z_encoded], dim=-1)

        mu_k = self.mu_net(h).view(B, self.n_knots, -1)
        # sigma_net now takes z + s_proj for state-dependent uncertainty
        sigma_input = torch.cat([z, s_proj], dim=-1)
        sigma_k = F.softplus(self.sigma_net(sigma_input).view(B, self.n_knots, -1))

        mu = _lin_interp_knots_to_T(mu_k, self.T)
        sigma = _lin_interp_knots_to_T(sigma_k, self.T)
        
        # CRITICAL: Normalize mu so mu[0] = 0 always
        # This ensures mu is in "cumulative delta from current position" coordinates
        # Without this, the network output is in an arbitrary coordinate frame
        # and the action (mu[1] - mu[0]) would not match the tube direction
        mu = mu - mu[:, 0:1, :]  # Subtract mu[0] from all timesteps, now mu[:, 0, :] = 0
        
        return mu, sigma

    # -----------------------------
    # Binding Failure: The Core Learning Signal
    # -----------------------------
    def compute_binding_failure(
        self, 
        mu: torch.Tensor, 
        sigma: torch.Tensor, 
        actual_traj: torch.Tensor,
        temporal_discount: float = 0.9,
    ) -> Dict[str, torch.Tensor]:
        """
        Causal knot-space binding failure: situation-indexed knots with delta-based transitions.
        
        Knots are time-indexed: knot i = situation at time t_i (from self.knot_times).
        This makes the world's lag a structured contradiction at specific indices, enabling
        causal learning.
        
        Key components:
          1. Position binding: shape mismatch at knot positions
          2. Delta binding (primary): transition mismatch (causality/speed emerges naturally)
          3. Volume loss: agency pressure
        
        Args:
            mu: (T, pred_dim) - Predicted tube centerline
            sigma: (T, pred_dim) - Predicted tube width
            actual_traj: (T_actual, pred_dim) - What actually happened (may be shorter than T)
            temporal_discount: Weight earlier failures more heavily
            
        Returns:
            Dict with binding_loss, volume_loss, and diagnostics
        """
        T_pred, d = mu.shape
        T_actual = actual_traj.shape[0]
        eps = 1e-6
        
        # Extract knots: use arc-length normalized extraction for actual trajectory
        # This handles early cutoff better - knots represent % progress, not timestep index
        knot_indices = self.knot_times.clamp(max=T_actual - 1)
        K = len(knot_indices)
        
        # Predicted knots: use time-indexing (actor's plan is always full length)
        predicted_knots = mu[knot_indices]  # (K, d) - what actor intended at time t_i
        
        # Actual knots: use arc-length normalized extraction (handles cutoff/undershoot)
        # This makes speed limit / cutoff show up as shape mismatch, not indexing mismatch
        actual_knots = self._extract_knots_by_arc_length(
            actual_traj, K, self.knot_times
        )  # (K, d) - what world gave at progress fraction corresponding to t_i
        
        # Only clamp for numerical stability (tiny epsilon), not to prevent learning
        sigma_k = sigma[knot_indices].clamp(min=1e-6)  # (K, d) - tiny minimum for numerical stability
        
        # --- 1) Position residuals (shape mismatch) ---
        pos_res = actual_knots - predicted_knots  # (K, d)
        pos_dist = torch.norm(pos_res, dim=-1)  # (K,)
        
        sigma_k_scale = sigma_k.mean(dim=-1).clamp(min=eps)  # (K,)
        pos_norm = pos_dist / (sigma_k_scale * self.k_sigma + eps)
        pos_fail = torch.relu(pos_norm - 1.0)  # (K,)
        
        # Temporal weighting for positions
        w_pos = torch.pow(
            torch.tensor(temporal_discount, device=mu.device),
            torch.arange(K, device=mu.device, dtype=mu.dtype)
        )
        w_pos = w_pos / w_pos.sum()
        pos_loss = (pos_fail * w_pos).sum() * K
        
        # --- 2) Delta residuals (causality / transition mismatch) ---
        # This is PRIMARY: makes "speed" emerge without ever mentioning speed
        # If world clips motion, it shows as consistent shrinkage of ||ΔK_pred|| vs ||ΔK_actual||
        if K > 1:
            pred_dk = predicted_knots[1:] - predicted_knots[:-1]  # (K-1, d)
            act_dk = actual_knots[1:] - actual_knots[:-1]        # (K-1, d)
            dk_res = act_dk - pred_dk  # (K-1, d)
            dk_dist = torch.norm(dk_res, dim=-1)  # (K-1,)
            
            # Uncertainty for deltas: average endpoint sigmas
            sigma_dk = (sigma_k[1:] + sigma_k[:-1]).mean(dim=-1).clamp(min=eps)  # (K-1,)
            dk_norm = dk_dist / (sigma_dk * self.k_sigma + eps)
            dk_fail = torch.relu(dk_norm - 1.0)  # (K-1,)
            
            # Temporal weighting for deltas
            w_dk = torch.pow(
                torch.tensor(temporal_discount, device=mu.device),
                torch.arange(K - 1, device=mu.device, dtype=mu.dtype)
            )
            w_dk = w_dk / w_dk.sum()
            dk_loss = (dk_fail * w_dk).sum() * (K - 1)
        else:
            dk_loss = torch.tensor(0.0, device=mu.device)
        
        # --- 3) Combine: delta dominates (causality is primary) ---
        binding_loss = 0.25 * pos_loss + 1.0 * dk_loss
        
        # --- 4) Volume loss: Agency pressure ---
        # Use raw sigma for volume loss (let it learn naturally)
        volume_loss = torch.log(sigma + eps).sum(dim=-1).mean()
        
        # --- 5) Diagnostics ---
        predicted_length = self._compute_arc_length(mu)
        actual_length = self._compute_arc_length(actual_traj)
        length_ratio = actual_length / (predicted_length + eps)
        
        return {
            "binding_loss": binding_loss,
            "pos_loss": pos_loss,
            "dk_loss": dk_loss,
            "volume_loss": volume_loss,
            "length_ratio": length_ratio,
            "mean_knot_distance": pos_dist.mean() if K > 0 else torch.tensor(0.0, device=mu.device),
            "knot_residuals": pos_res,  # (K, d)
        }
    
    def _compute_arc_length(self, traj: torch.Tensor) -> torch.Tensor:
        """Compute total arc length of trajectory."""
        diffs = traj[1:] - traj[:-1]  # (T-1, d)
        segment_lengths = torch.norm(diffs, dim=-1)  # (T-1,)
        return segment_lengths.sum()
    
    def _extract_knots_by_arc_length(
        self, 
        traj: torch.Tensor, 
        n_knots: int,
        knot_times: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract knots from trajectory using arc-length normalized positions.
        This handles early cutoff better than pure time-indexing.
        
        Args:
            traj: (T_actual, d) actual trajectory
            n_knots: number of knots to extract
            knot_times: (n_knots,) desired time indices (for reference)
            
        Returns:
            knots: (n_knots, d) extracted knot positions
        """
        T_actual, d = traj.shape
        
        # Compute cumulative arc length
        diffs = traj[1:] - traj[:-1]  # (T_actual-1, d)
        segment_lengths = torch.norm(diffs, dim=-1)  # (T_actual-1,)
        cum_lengths = torch.cat([
            torch.tensor([0.0], device=traj.device),
            segment_lengths.cumsum(dim=0)
        ])  # (T_actual,)
        total_length = cum_lengths[-1]
        
        # If trajectory is too short or zero length, fall back to time-indexing
        if total_length < 1e-6 or T_actual < n_knots:
            # Fallback: use time-indexing with clamping
            valid_times = knot_times.clamp(max=T_actual - 1)
            return traj[valid_times]
        
        # Extract knots at arc-length fractions
        # Map knot times to arc-length fractions
        target_fractions = knot_times.float() / (self.T - 1)  # (n_knots,)
        target_lengths = target_fractions * total_length  # (n_knots,)
        
        # Find positions along trajectory for each target length
        knots = []
        for target_len in target_lengths:
            # Find the segment containing this arc length
            idx = torch.searchsorted(cum_lengths, target_len, right=True).clamp(0, T_actual - 2)
            idx = idx.item()
            
            # Interpolate within the segment
            if idx == 0:
                knots.append(traj[0])
            elif idx >= T_actual - 1:
                knots.append(traj[-1])
            else:
                len_start = cum_lengths[idx]
                len_end = cum_lengths[idx + 1]
                if len_end - len_start < 1e-8:
                    # Degenerate segment, use endpoint
                    knots.append(traj[idx])
                else:
                    alpha = (target_len - len_start) / (len_end - len_start)
                    knots.append((1 - alpha) * traj[idx] + alpha * traj[idx + 1])
        
        return torch.stack(knots, dim=0)  # (n_knots, d)
    

    # -----------------------------
    # Necessity-gated repulsion (ports diversify only when needed)
    # -----------------------------
    def repulsion_loss(self, z_modes: torch.Tensor) -> torch.Tensor:
        """
        z_modes: (M, z_dim) on unit sphere.
        L_repel = sum_{i<j} max(0, cos(z_i,z_j) - tau)
        """
        z = F.normalize(z_modes, dim=-1)
        sim = z @ z.t()  # (M, M)
        # upper triangle i<j
        iu = torch.triu_indices(self.M, self.M, offset=1, device=z.device)
        sims = sim[iu[0], iu[1]]
        return torch.relu(sims - self.repel_tau).mean()

    def _knot_times(self, T: int, device) -> torch.Tensor:
        # fixed knot times in [0, T-1]
        return torch.linspace(0, T - 1, self.n_knots, device=device).long()

    def _knot_nll(self, mu: torch.Tensor, sigma: torch.Tensor, x: torch.Tensor, knot_t: torch.Tensor) -> torch.Tensor:
        """
        mu,sigma,x: (T, d)
        knot_t: (K,)
        Returns scalar Gaussian NLL evaluated at knot times.
        
        Uses proper NLL with floor to prevent unbounded negative values.
        """
        eps = 1e-6
        mu_k = mu[knot_t]           # (K,d)
        x_k  = x[knot_t]            # (K,d)
        # Only clamp for numerical stability (tiny epsilon), not to prevent learning
        # Let sigma learn naturally - the log(2*pi) constant prevents unbounded negative NLL
        sig_k = sigma[knot_t].clamp(min=1e-6)  # Tiny minimum for numerical stability only
        # Don't clamp sigma here - let it learn naturally
        # Only clamp in the final loss computation to prevent numerical issues
        sig_k = sigma[knot_t].clamp(min=1e-6)  # Tiny minimum for numerical stability only

        r = x_k - mu_k              # (K,d)

        # diagonal Gaussian NLL (proper form with constant):
        # 0.5 * [ (r/sig)^2 + log(2*pi*sig^2) ]
        # = 0.5 * [ (r/sig)^2 + 2*log(sig) + log(2*pi) ]
        # We keep the log(2*pi) constant to prevent unbounded negative values
        log_2pi = math.log(2.0 * math.pi)
        nll = 0.5 * ((r / sig_k).pow(2) + 2.0 * sig_k.log() + log_2pi)
        return nll.mean() * self.n_knots

    def train_step_trajectory_jacobian(
        self,
        s0: torch.Tensor,
        actual_traj: torch.Tensor,
        intent_delta: torch.Tensor,
        optimizer: optim.Optimizer,
        alpha_bind: float = 1.0,
        alpha_terminal: float = 1.0,
        alpha_metabolic: float = 0.1,
        force_z_only: bool = False,
        z_executed: Optional[torch.Tensor] = None,  # If provided, train on this z (used for action)
    ) -> Dict[str, float]:

        self.train()
        assert s0.dim() == 1

        eps = 1e-6
        T = actual_traj.shape[0]
        knot_t = self._knot_times(T=self.T, device=actual_traj.device)
        knot_t_valid = knot_t[knot_t < T]
        if knot_t_valid.numel() == 0:
            knot_t_valid = torch.tensor([min(T - 1, self.T - 1)], device=actual_traj.device)

        # If z_executed is provided, train on that z (ensures consistency with action taken)
        # Otherwise, use multi-head proposal selection
        if z_executed is not None:
            # Train on the exact z that was used for action
            z_modes = z_executed.unsqueeze(0)  # (1, z_dim)
            mu_M, sigma_M = self.get_tube(s0.unsqueeze(0), z_modes, force_z_only=force_z_only)  # (1,T,d)
            effective_M = 1
        else:
            z_modes_B, _ = self.propose_z_modes(s0, force_z_only=force_z_only)
            z_modes = z_modes_B.squeeze(0)  # (M, z_dim)
            mu_M, sigma_M = self.get_tube(s0.unsqueeze(0), z_modes, force_z_only=force_z_only)  # (M,T,d)
            effective_M = self.M

        # --- score heads with the SAME objective we optimize ---
        losses = []
        terminals = []
        metab = []
        bind_nlls = []
        goal_dists = []
        sigma_mins = []
        sigma_means = []
        sigma_ends = []

        for m in range(effective_M):
            mu = mu_M[m]  # Already normalized in get_tube: mu[0] = 0
            sigma = sigma_M[m]
            
            # Don't clamp sigma during forward pass - let it learn naturally
            # Only use tiny minimum for numerical stability in division operations
            bind_nll = self._knot_nll(mu, sigma, actual_traj, knot_t_valid)

            # Start anchor: tube MUST start at current position (mu[0] = 0 in delta coordinates)
            # This is a soft loss to encourage the network to learn this directly
            mu_start = mu[0]  # (d,) predicted start position (should be 0 in delta coords)
            start_residual_sq = (mu_start ** 2).sum()  # Should be 0
            start_anchor_loss = start_residual_sq  # Direct MSE - no Huber needed (should be exactly 0)
            
            # Goal constraint: pure geometric constraint (not NLL with sigma)
            # This prevents sigma from being used as a lever to "score" the goal
            # Goal is a hard geometric requirement, not a probabilistic observation
            # IMPORTANT: Always use full-horizon endpoint for goal, not truncated by actual_traj length
            # (actual_traj may be short in online mode, but we still want full-horizon goal planning)
            mu_end = mu[self.T - 1]  # (d,) predicted FULL-HORIZON endpoint in delta coordinates
            
            # Goal loss: fixed-scale geometric constraint (Huber loss for robustness)
            goal_residual = mu_end - intent_delta  # (d,) residual to goal
            goal_dist_sq = (goal_residual ** 2).sum()
            # Use Huber loss: quadratic for small errors, linear for large (more robust than MSE)
            goal_dist = torch.sqrt(goal_dist_sq + 1e-8)
            huber_delta = 0.1  # Transition point
            goal_loss = torch.where(
                goal_dist < huber_delta,
                0.5 * goal_dist_sq,  # Quadratic for small errors
                huber_delta * (goal_dist - 0.5 * huber_delta)  # Linear for large errors
            )
            
            # Step binding: actual step vs predicted step at SAME timestep
            # This is the PRIMARY binding signal in online mode
            # mu is already normalized in get_tube (mu[0] = 0)
            step_binding_loss = torch.tensor(0.0, device=mu.device)
            if T > 1:
                # Compare each observed step (skip t=0 which is always 0)
                for t_obs in range(1, T):
                    if t_obs < self.T:  # Ensure we have a prediction for this time
                        actual_t = actual_traj[t_obs]
                        pred_t = mu[t_obs]
                        step_residual = actual_t - pred_t
                        step_dist_sq = (step_residual ** 2).sum()
                        step_dist = torch.sqrt(step_dist_sq + 1e-8)
                        # Huber loss for robustness
                        step_loss_t = torch.where(
                            step_dist < huber_delta,
                            0.5 * step_dist_sq,
                            huber_delta * (step_dist - 0.5 * huber_delta)
                        )
                        step_binding_loss = step_binding_loss + step_loss_t
                # Normalize by number of observed steps
                step_binding_loss = step_binding_loss / max(1, T - 1)
            
            # Endpoint binding (legacy, now mostly redundant with step_binding)
            t_actual_end = min(T - 1, self.T - 1)
            actual_end = actual_traj[-1]
            mu_at_actual_end = mu[t_actual_end]
            is_real_end = torch.norm(actual_end) > 1e-6 or T == 1
            if is_real_end:
                endpoint_residual = actual_end - mu_at_actual_end
                endpoint_dist_sq = (endpoint_residual ** 2).sum()
                endpoint_dist = torch.sqrt(endpoint_dist_sq + 1e-8)
                endpoint_loss = torch.where(
                    endpoint_dist < huber_delta,
                    0.5 * endpoint_dist_sq,
                    huber_delta * (endpoint_dist - 0.5 * huber_delta)
                )
            else:
                endpoint_loss = torch.tensor(0.0, device=mu.device)
            
            # Progress penalty: compare lengths over SAME time horizon
            # In online mode (T_actual << self.T), compare predicted[0:T_actual] to actual
            # This prevents unfair comparison of 1-step actual vs 20-step predicted
            actual_length = self._compute_arc_length(actual_traj)
            if actual_length > eps and T > 1:  # We actually moved and have >1 timestep
                # Compare only the predicted trajectory up to observed time
                mu_observed = mu[:T]  # Same horizon as actual
                predicted_length = self._compute_arc_length(mu_observed)
                if predicted_length > eps:
                    length_ratio = actual_length / predicted_length
                    progress_error = length_ratio - 1.0  # Should be ~0 if matched
                    progress_loss = progress_error.pow(2)
                else:
                    progress_loss = torch.tensor(0.0, device=mu.device)
            else:
                progress_loss = torch.tensor(0.0, device=mu.device)

            # metabolic: log arc length
            seg = mu[1:] - mu[:-1]
            path_len = torch.norm(seg, dim=-1).sum()
            metabolic = torch.log(path_len + 1.0)
            
            # Stall penalty: penalize zero-motion plans (prevents "stay still" attractor)
            # If the actor predicts no movement, it should be heavily penalized
            # This ensures the actor must make progress toward the goal
            stall_penalty = torch.exp(-path_len * 10.0)  # Large penalty when path_len ≈ 0, decays quickly
            
            # Minimum progress requirement: predicted path should be at least some fraction of distance to goal
            goal_distance = torch.norm(intent_delta).clamp(min=1e-6)
            min_progress_ratio = path_len / goal_distance  # Should be > 0.1 or so
            min_progress_penalty = torch.relu(0.1 - min_progress_ratio) * 10.0  # Penalize if < 10% of goal distance

            # Total loss: start anchor (hard constraint) + binding (world feedback) + 
            #            goal (geometric constraint) + step binding (online learning) + 
            #            progress (cutoff penalty) + metabolic (complexity)
            total = (
                10.0 * start_anchor_loss  # Strong weight - this is a hard constraint
                + alpha_bind * bind_nll  # Knot-based NLL (works when full traj observed)
                + alpha_bind * 2.0 * step_binding_loss  # Step binding (PRIMARY for online mode)
                + alpha_terminal * goal_loss  # Geometric constraint, not NLL
                + alpha_terminal * 0.3 * progress_loss  # Progress penalty (smaller weight)
                + alpha_metabolic * metabolic
                + 2.0 * stall_penalty  # Reduced from 5.0 - step binding should handle this
                + 1.0 * min_progress_penalty  # Reduced from 2.0
            )
            losses.append(total)
            terminals.append(goal_loss.detach())  # Goal geometric loss
            metab.append(metabolic.detach())
            bind_nlls.append(bind_nll.detach())
            goal_dists.append(goal_dist_sq.detach())
            # Track start anchor loss for diagnostics
            if m == 0:
                start_anchor_losses = []
                stall_penalties = []
                min_progress_penalties = []
                step_bind_losses = []
            start_anchor_losses.append(start_anchor_loss.detach())
            stall_penalties.append(stall_penalty.detach())
            min_progress_penalties.append(min_progress_penalty.detach())
            step_bind_losses.append(step_binding_loss.detach())
            # Track raw sigma (not clamped) for diagnostics - want to see if it's learning
            sigma_mins.append(sigma.min().detach())
            sigma_means.append(sigma.mean().detach())
            sigma_ends.append(sigma[t_actual_end].mean().detach())

        losses_t = torch.stack(losses)  # (effective_M,)
        m_star = torch.argmin(losses_t)

        # --- optional repulsion (only when using multiple proposal modes) ---
        regret_proxy = (torch.sort(losses_t).values[1] - torch.sort(losses_t).values[0]) if effective_M >= 2 else torch.tensor(0.0, device=losses_t.device)
        repel_gate = (regret_proxy > self.regret_delta).float() if effective_M >= 2 else torch.tensor(0.0, device=losses_t.device)
        repel = self.repulsion_loss(z_modes) if effective_M >= 2 else torch.tensor(0.0, device=losses_t.device)
        repel_coef = self.repel_min + self.repel_weight * repel_gate

        loss = losses_t[m_star] + repel_coef * repel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update head usage EMA (only when using multi-head proposal selection)
        if effective_M > 1:
            with torch.no_grad():
                onehot = torch.zeros(self.M, device=loss.device)
                onehot[m_star] = 1.0
                self.head_usage_ema.mul_(0.995).add_(0.005 * onehot)

        # Compute weighted contributions for diagnostics
        best_bind_nll = bind_nlls[m_star]
        best_goal_loss = terminals[m_star]  # This is now goal_loss (geometric)
        best_metab = metab[m_star]
        best_goal_dist_sq = goal_dists[m_star]
        best_start_anchor = start_anchor_losses[m_star] if 'start_anchor_losses' in locals() else torch.tensor(0.0, device=losses_t.device)
        
        weighted_bind = alpha_bind * best_bind_nll
        weighted_goal = alpha_terminal * best_goal_loss  # Goal geometric loss contribution
        weighted_metab = alpha_metabolic * best_metab
        weighted_start = 10.0 * best_start_anchor  # Start anchor contribution
        best_stall_penalty = stall_penalties[m_star] if 'stall_penalties' in locals() else torch.tensor(0.0, device=losses_t.device)
        best_min_progress_penalty = min_progress_penalties[m_star] if 'min_progress_penalties' in locals() else torch.tensor(0.0, device=losses_t.device)
        weighted_stall = 5.0 * best_stall_penalty
        weighted_min_progress = 2.0 * best_min_progress_penalty
        
        # Diagnostic: verify goal_dist is from predicted endpoint, not actual
        with torch.no_grad():
            mu_star = mu_M[m_star]
            actual_end = actual_traj[-1]
            mu_end_star = mu_star[knot_t_valid[-1]]
            endpoint_mismatch = ((mu_end_star - actual_end) ** 2).sum().item()
            # Also check start position
            mu_start_star = mu_star[0]
            start_mismatch = (mu_start_star ** 2).sum().item()
        
        best_step_bind = step_bind_losses[m_star] if 'step_bind_losses' in locals() else torch.tensor(0.0, device=losses_t.device)
        
        return {
            "loss": float(loss.item()),
            "best_head": int(m_star.item()),
            "best_total": float(losses_t[m_star].item()),
            "best_terminal_nll": float(best_goal_loss.item()),  # Now goal_loss (geometric)
            "best_metabolic": float(best_metab.item()),
            "best_binding_nll": float(best_bind_nll.item()),
            "step_binding_loss": float(best_step_bind.item()),  # Step binding (online mode)
            "goal_dist_sq": float(best_goal_dist_sq.item()),
            "goal_dist": float(torch.sqrt(best_goal_dist_sq).item()),
            "endpoint_mismatch": float(endpoint_mismatch),  # Diagnostic: should be > 0 if learning
            "start_anchor_loss": float(best_start_anchor.item()),  # Start position constraint
            "start_mismatch": float(start_mismatch),  # Diagnostic: ||mu[0]||^2 (should be ~0)
            "weighted_start": float(weighted_start.item()),  # Weighted start anchor contribution
            # Weighted contributions (diagnostics)
            "weighted_binding": float(weighted_bind.item()),
            "weighted_terminal": float(weighted_goal.item()),  # Goal geometric loss contribution
            "weighted_metabolic": float(weighted_metab.item()),
            "weighted_step_binding": float(alpha_bind * 2.0 * best_step_bind.item()),  # Step binding contribution
            # Sigma statistics (diagnostics)
            "sigma_min": float(sigma_mins[m_star].item()),
            "sigma_mean": float(sigma_means[m_star].item()),
            "sigma_end_mean": float(sigma_ends[m_star].item()),
            # Repulsion metrics
            "regret_proxy": float(regret_proxy.item()),
            "repel_gate": float(repel_gate.item()),
            "repel": float(repel.item()),
            "usage_entropy": float(self._usage_entropy().item()),
        }

    def _usage_entropy(self) -> torch.Tensor:
        p = self.head_usage_ema.clamp(min=1e-8)
        p = p / p.sum()
        return -(p * p.log()).sum()

    @torch.no_grad()
    def _maybe_reseed(self, z_modes: torch.Tensor, losses: torch.Tensor, regret_proxy: torch.Tensor):
        if not self.reseed_enabled:
            return
        self.steps_since_reseed += 1
        if int(self.steps_since_reseed.item()) < self.reseed_patience:
            return
        # only reseed when necessity seems present but heads are underused
        if regret_proxy.item() <= self.regret_delta:
            return

        usage = (self.head_usage_ema / (self.head_usage_ema.sum() + 1e-8)).cpu()
        worst = int(torch.argmin(self.head_usage_ema).item())
        if usage[worst].item() > self.reseed_min_usage:
            return

        # reseed worst head toward best-performing z (lowest loss)
        best = int(torch.argmin(losses).item())
        # perform a small random perturbation so it doesn't duplicate perfectly
        new_z = F.normalize(z_modes[best] + 0.05 * torch.randn_like(z_modes[best]), dim=-1)

        # overwrite router final layer bias chunk for that head minimally:
        # We can't directly set outputs, but we *can* keep a tiny learned nudge by storing a persistent
        # head-bias parameter. Simpler: do nothing structural and just reset usage timer.
        # If you want "true reseed", add a learnable per-head bias and set it here.
        # For now: reset usage EMA to encourage exploration.
        self.head_usage_ema[worst] = self.head_usage_ema.mean()
        self.steps_since_reseed.zero_()

    def adapt_sigma_from_binding(
        self,
        binding_terms: Dict[str, torch.Tensor],
        sigma: torch.Tensor,
        expand_rate: float = 0.1,
        shrink_rate: float = 0.05,
    ) -> torch.Tensor:
        """
        Adaptive sigma adjustment based on binding failure.
        
        If we exited → expand sigma (was too confident)
        If we stayed inside with room to spare → shrink sigma (gain agency)
        
        This can be used for online adaptation without full backprop.
        
        Args:
            binding_terms: Output from compute_binding_failure
            sigma: Current sigma (T, pred_dim)
            expand_rate: How much to expand on failure
            shrink_rate: How much to shrink on success
            
        Returns:
            Adjusted sigma suggestion
        """
        ever_exited = binding_terms["ever_exited"]
        max_divergence = binding_terms["max_divergence"]
        
        if ever_exited > 0.5:
            # Failure: expand sigma proportionally to how much we exceeded
            expansion = 1.0 + expand_rate * max_divergence
            return sigma * expansion.clamp(1.0, 2.0)
        else:
            # Success: shrink sigma (gain agency) but not too aggressively
            # Only shrink if we have significant margin (max_divergence < 0.5)
            if max_divergence < 0.5:
                shrinkage = 1.0 - shrink_rate * (1.0 - max_divergence)
                return sigma * shrinkage.clamp(0.5, 1.0)
        
        return sigma

    # -----------------------------
    # Unsupervised encoder training (InfoNCE)
    # -----------------------------
    def train_encoder_ssl_batch(
        self,
        traj_delta_batch: torch.Tensor,
        optimizer: optim.Optimizer,
    ) -> Dict[str, float]:
        """
        traj_delta_batch: (B, T, pred_dim) trajectory deltas
        Trains encoder_mlp only (but you can include others if you want).
        """
        self.train()
        B, T, d = traj_delta_batch.shape
        device = traj_delta_batch.device

        # make two augmented views
        v1 = []
        v2 = []
        for i in range(B):
            x = traj_delta_batch[i]
            v1.append(self._augment_traj_delta(x))
            v2.append(self._augment_traj_delta(x))
        v1 = torch.stack(v1, dim=0)  # (B,T,d)
        v2 = torch.stack(v2, dim=0)

        # Encode and clone immediately to avoid any in-place issues
        z1 = self.encode(v1.reshape(B, -1)).clone()
        z2 = self.encode(v2.reshape(B, -1)).clone()

        loss = self._info_nce_loss(z1, z2, temperature=self.info_nce_temp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return {
            "ssl_loss": float(loss.item()),
            "z_std": float(z1.std().item()),
            "queue_len": int(self._queue.get().shape[0]) if self._queue is not None else 0,
        }

    # -----------------------------
    # Inference: multimodal multi-start CEM (constant cost)
    # -----------------------------
    @torch.no_grad()
    def select_z_geometric_multimodal(
        self,
        s0: torch.Tensor,
        intent_delta: Optional[torch.Tensor] = None,
        trajectory_delta_hint: Optional[torch.Tensor] = None,
        pop_size: Optional[int] = None,
        n_iters: Optional[int] = None,
        force_z_only: bool = False,
        L_refine: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Intent-conditioned port selection.
        
        From the formulation: Actor chooses port that maximizes agency
        subject to binding success, conditioned on intent.
        
        Args:
            s0: Starting observation
            intent_delta: Goal in delta coordinates - determines which ports are afforded
            trajectory_delta_hint: Optional hint for ranking
            pop_size, n_iters: CEM parameters
            force_z_only: Zero out observation embedding
            L_refine: Number of top heads to refine with CEM
        """
        if s0.dim() != 1:
            raise ValueError("select_z_geometric_multimodal expects s0: (obs_dim,)")

        pop_size = int(pop_size or self.cem_pop_size)
        n_iters = int(n_iters or self.cem_iters)
        L = int(min(L_refine or self.L_refine, self.M))

        z_modes_B, pi = self.propose_z_modes(s0, force_z_only=force_z_only)  # (1,M,D)
        z_modes = z_modes_B.squeeze(0)  # (M,D)

        # Optional: hint vector from encoder
        hint_score = 0.0
        z_hint = None
        if trajectory_delta_hint is not None:
            z_hint = self.encode(trajectory_delta_hint.reshape(-1))
            hint_score = (z_modes @ z_hint).detach()

        # Intent-conditioned scoring: which ports are afforded for this intent?
        mean_scores = self._score_z_candidates(
            s0, z_modes, force_z_only=force_z_only, intent_delta=intent_delta
        )
        if z_hint is not None:
            mean_scores = mean_scores + 0.5 * hint_score

        top_idx = torch.topk(mean_scores, k=L, largest=True).indices

        best_z = z_modes[top_idx[0]]
        best_score = mean_scores[top_idx[0]].item()

        # Refine with intent-conditioned CEM
        for idx in top_idx:
            z0 = z_modes[idx]
            z_ref, score_ref = self._cem_refine_from_mean(
                s0, z0, pop_size=pop_size, n_iters=n_iters, 
                force_z_only=force_z_only, intent_delta=intent_delta
            )
            if score_ref > best_score:
                best_score = score_ref
                best_z = z_ref

        return best_z

    @torch.no_grad()
    def _score_z_candidates(
        self, 
        s0: torch.Tensor, 
        z: torch.Tensor, 
        force_z_only: bool = False,
        intent_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Intent-conditioned port scoring (higher is better).
        
        From the formulation: Actor chooses port that maximizes agency
        subject to binding success, conditioned on intent.
        
        A port is "afforded" for an intent only if it moves toward that intent.
        This is a SELECTION constraint, not a training signal.
        
        Args:
            s0: Starting observation
            z: (N, z_dim) candidate z values
            force_z_only: Zero out observation embedding
            intent_delta: Optional (pred_dim,) goal in delta coordinates
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        N = z.shape[0]
        d = self.pred_dim
        mu, sigma = self.get_tube(s0.unsqueeze(0), z, force_z_only=force_z_only)  # (N,T,d)

        # 1. Anchoring: tube should start at current position (delta = 0)
        start_error_sq = (mu[:, 0, :] ** 2).sum(dim=-1) / d
        start_penalty = -start_error_sq * 100.0

        # 2. Agency: prefer narrow tubes (high commitment)
        agency_score = -sigma.mean(dim=(1, 2)) * 0.5
        sigma_consistency = -sigma.std(dim=(1, 2)) * 0.2

        # 3. Intent-conditioned affordance: does this port serve the intent?
        intent_score = torch.zeros(N, device=z.device)
        
        if intent_delta is not None:
            # Endpoint proximity to intent (goal)
            endpoint = mu[:, -1, :]  # (N, d)
            endpoint_dist_sq = ((endpoint - intent_delta) ** 2).sum(dim=-1) / d
            
            # Strong preference for tubes that reach the goal
            # This is "is this port afforded for this intent?"
            intent_score = -endpoint_dist_sq * 50.0
            
            # Direction alignment: tube should point toward intent
            tube_direction = mu[:, -1, :] - mu[:, 0, :]  # (N, d)
            tube_norm = torch.norm(tube_direction, dim=-1, keepdim=True).clamp(min=1e-6)
            intent_norm = torch.norm(intent_delta).clamp(min=1e-6)
            
            direction_sim = (tube_direction * intent_delta).sum(dim=-1) / (tube_norm.squeeze() * intent_norm)
            intent_score = intent_score + direction_sim * 10.0
            
            # Progress reward: only if moving toward intent
            progress_toward_goal = torch.norm(tube_direction, dim=-1) * direction_sim.clamp(min=0)
            intent_score = intent_score + progress_toward_goal * 5.0

        return start_penalty + agency_score + sigma_consistency + intent_score

    @torch.no_grad()
    def _cem_refine_from_mean(
        self,
        s0: torch.Tensor,
        z_mean: torch.Tensor,
        pop_size: int,
        n_iters: int,
        force_z_only: bool = False,
        intent_delta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Intent-conditioned CEM refinement.
        Finds the z that maximizes agency while serving the intent.
        """
        curr_mean = F.normalize(z_mean, dim=-1)
        curr_std = torch.ones_like(curr_mean) * self.cem_init_std

        n_elites = max(8, int(pop_size * self.cem_elite_frac))

        best_z = curr_mean
        best_score = -1e9

        for _ in range(n_iters):
            cand = curr_mean + torch.randn(pop_size, self.z_dim, device=s0.device) * curr_std
            cand = F.normalize(cand, p=2, dim=1)

            scores = self._score_z_candidates(
                s0, cand, force_z_only=force_z_only, intent_delta=intent_delta
            )
            top = torch.topk(scores, k=n_elites, largest=True)
            elites = cand[top.indices]

            curr_mean = F.normalize(elites.mean(dim=0), dim=-1)
            curr_std = elites.std(dim=0) + 1e-6

            smax, imax = scores.max(dim=0)
            if smax.item() > best_score:
                best_score = smax.item()
                best_z = cand[imax]

        return best_z, float(best_score)

    # -----------------------------
    # Parameter counting (unchanged idea)
    # -----------------------------
    def count_parameters(self):
        obs_params = sum(p.numel() for p in self.obs_proj.parameters())
        z_params = sum(p.numel() for p in self.z_encoder.parameters())
        total_params = obs_params + z_params
        return {
            "obs_params": int(obs_params),
            "z_params": int(z_params),
            "total_params": int(total_params),
            "z_to_obs_ratio": float(z_params / (obs_params + 1e-8)),
        }