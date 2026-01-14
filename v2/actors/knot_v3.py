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


def _cosine_pairwise(z: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity for normalized vectors. z: (N, D) -> (N, N)."""
    z = F.normalize(z, dim=-1)
    return z @ z.t()


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
        x = F.normalize(x.detach(), dim=-1)
        n = x.shape[0]
        if n >= self.size:
            self.buf.copy_(x[-self.size :])
            self.ptr = 0
            self.full = True
            return
        end = self.ptr + n
        if end <= self.size:
            self.buf[self.ptr:end] = x
        else:
            first = self.size - self.ptr
            self.buf[self.ptr:] = x[:first]
            self.buf[: end - self.size] = x[first:]
        self.ptr = end % self.size
        if self.ptr == 0:
            self.full = True

    def get(self) -> torch.Tensor:
        return self.buf if self.full else self.buf[: self.ptr]


class Actor(nn.Module):
    """
    Unsupervised + online geometric knot actor with implicit latent ports.

    Core changes vs your original:
      1) Unsupervised encoder training via InfoNCE on trajectory-delta augmentations (no labels).
      2) State-conditioned multimodal proposal q(z|s0) with M fixed heads on the unit sphere.
      3) Tube training via winner-take-all (WTA) across heads (online specialization).
      4) Necessity-gated repulsion between heads (heads only diversify when regret indicates need).
      5) Inference: cheap scoring across heads -> refine top-L with short CEM (constant cost).
      6) Optional capacity recycling: reseed underused heads when necessary (still constant M).

    Fiber intuition:
      For a given situation s0, the router produces a *slice* through latent port space:
        {z_m(s0)}_{m=1..M}
      Each z_m induces a tube fiber in trajectory space via get_tube(s0, z_m).
      When a single fiber covers the world well (low regret), heads collapse.
      When environment demands multiple basins (high regret), heads repel and specialize.
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

        # Bottleneck
        s0_emb_dim: int = 8,

        # Multimodal proposal (ports)
        M: int = 8,                 # number of proposal heads (constant)
        L_refine: int = 2,          # refine top-L heads with short CEM

        # Necessity-gated repulsion
        regret_delta: float = 0.05, # gate threshold on regret proxy
        repel_tau: float = 0.25,    # cosine similarity margin; smaller -> more separation
        repel_weight: float = 0.2,  # repulsion strength when gate is on

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
        self.s0_emb_dim = s0_emb_dim

        # Multimodal proposal params
        self.M = int(M)
        self.L_refine = int(min(L_refine, M))
        self.regret_delta = float(regret_delta)
        self.repel_tau = float(repel_tau)
        self.repel_weight = float(repel_weight)

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

        # --------- Encoder: trajectory_delta -> z (unit sphere) ---------
        self.encoder_mlp = nn.Sequential(
            nn.Linear(T * pred_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, z_dim),
        )

        # --------- Bottleneck encoders ---------
        self.obs_proj = nn.Linear(obs_dim, s0_emb_dim)
        self.z_encoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        # --------- Tube decoders ---------
        self.mu_net = nn.Sequential(
            nn.Linear(s0_emb_dim + 256, 64),
            nn.ReLU(),
            nn.Linear(64, n_knots * pred_dim),
        )

        self.sigma_net = nn.Sequential(
            nn.Linear(z_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_knots * pred_dim),
        )

        # --------- Multimodal proposal q(z|s0) ---------
        self.z_router = nn.Sequential(
            nn.Linear(s0_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.M * z_dim),
        )
        self.z_logits = nn.Sequential(
            nn.Linear(s0_emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.M),
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
        x = traj_delta

        # 1) time crop (keep contiguous window)
        if self.aug_time_crop_min_frac < 1.0:
            min_len = max(2, int(math.ceil(T * self.aug_time_crop_min_frac)))
            crop_len = torch.randint(low=min_len, high=T + 1, size=(1,), device=x.device).item()
            start = torch.randint(low=0, high=T - crop_len + 1, size=(1,), device=x.device).item()
            crop = x[start : start + crop_len]

            # resample back to length T by linear interpolation in time
            t_old = torch.linspace(0, 1, crop_len, device=x.device, dtype=x.dtype)
            t_new = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype)
            # interpolate each dim
            crop_resamp = []
            for j in range(d):
                crop_resamp.append(torch.interp(t_new, t_old, crop[:, j]))
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
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        B, D = z1.shape

        # positives: z1[i] with z2[i]
        pos = (z1 * z2).sum(dim=-1, keepdim=True)  # (B, 1)

        # negatives: within-batch + queue
        logits_neg = []

        # within-batch negatives against z2
        sim_12 = z1 @ z2.t()  # (B, B)
        mask = ~torch.eye(B, device=z1.device, dtype=torch.bool)
        logits_neg.append(sim_12[mask].view(B, B - 1))

        # queue negatives
        self._ensure_queue(z1.device)
        q = self._queue.get()  # (Q, D) or empty
        if q.numel() > 0:
            logits_neg.append(z1 @ q.t())  # (B, Q)

        neg = torch.cat(logits_neg, dim=1) if len(logits_neg) > 0 else torch.empty(B, 0, device=z1.device)

        # assemble logits: [pos | neg]
        logits = torch.cat([pos, neg], dim=1) / max(1e-8, temperature)
        labels = torch.zeros(B, dtype=torch.long, device=z1.device)  # positive at index 0
        loss = F.cross_entropy(logits, labels)

        # update queue with z2 (stop-grad)
        with torch.no_grad():
            self._queue.enqueue(z2)

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
        sigma_k = F.softplus(self.sigma_net(z).view(B, self.n_knots, -1))

        mu = _lin_interp_knots_to_T(mu_k, self.T)
        sigma = _lin_interp_knots_to_T(sigma_k, self.T)
        return mu, sigma

    # -----------------------------
    # Contract loss (per-sample, differentiable)
    # -----------------------------
    def contract_terms(self, s0: torch.Tensor, trajectory: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        s0: (obs_dim,)
        trajectory: (T, pred_dim)  (usually trajectory delta)
        mu, sigma:  (T, pred_dim)
        """
        dist = torch.abs(trajectory - mu)

        leakage = torch.relu(dist - sigma * self.k_sigma).pow(2).mean()
        over_estimation = torch.relu(sigma - dist * 1.2).mean()
        volume = sigma.mean()

        s0_pos = s0[: self.pred_dim]
        start_error = (mu[0] - s0_pos).pow(2).mean()

        traj_direction = trajectory[-1] - trajectory[0]
        tube_direction = mu[-1] - mu[0]
        direction_sim = F.cosine_similarity(traj_direction.unsqueeze(0), tube_direction.unsqueeze(0))
        direction_loss = (1.0 - direction_sim).mean()

        endpoint_error = (mu[-1] - trajectory[-1]).pow(2).mean()

        return {
            "leak": leakage,
            "lazy": over_estimation,
            "vol": volume,
            "start_err": start_error,
            "dir_loss": direction_loss,
            "end_err": endpoint_error,
        }

    def contract_loss_from_terms(self, terms: Dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            self.alpha_leak * terms["leak"]
            + 0.5 * terms["lazy"]
            + self.alpha_vol * terms["vol"]
            + 5.0 * terms["start_err"]
            + 2.0 * terms["dir_loss"]
            + 1.0 * terms["end_err"]
        )

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

    # -----------------------------
    # Tube training: WTA across M heads (unsupervised + online)
    # -----------------------------
    def train_step_wta(
        self,
        s0: torch.Tensor,
        trajectory: torch.Tensor,
        optimizer: optim.Optimizer,
        force_z_only: bool = False,
    ) -> Dict[str, float]:
        """
        For one sample:
          - propose M z modes from s0
          - compute tube loss per head
          - pick best head (WTA)
          - backprop through winning head + gated repulsion if necessary

        trajectory is expected to be (T, pred_dim) (often trajectory delta).
        """
        self.train()
        if s0.dim() != 1:
            raise ValueError("train_step_wta expects a single sample s0: (obs_dim,)")

        z_modes_B, pi = self.propose_z_modes(s0, force_z_only=force_z_only)  # (1,M,D), (1,M)
        z_modes = z_modes_B.squeeze(0)  # (M, D)

        # Evaluate each head
        # vectorize: build tubes for all M at once
        mu_M, sigma_M = self.get_tube(s0.unsqueeze(0), z_modes, force_z_only=force_z_only)  # (M,T,d)
        losses = []
        leaks = []
        vols = []
        for m in range(self.M):
            terms = self.contract_terms(s0, trajectory, mu_M[m], sigma_M[m])
            Lm = self.contract_loss_from_terms(terms)
            losses.append(Lm)
            leaks.append(terms["leak"])
            vols.append(terms["vol"])

        losses_t = torch.stack(losses)  # (M,)
        leaks_t = torch.stack(leaks)
        vols_t = torch.stack(vols)

        m_star = torch.argmin(losses_t)
        best = losses_t[m_star]
        # regret proxy: gap between best and second best (>=0)
        sorted_losses, _ = torch.sort(losses_t)
        regret_proxy = (sorted_losses[1] - sorted_losses[0]) if self.M >= 2 else torch.tensor(0.0, device=best.device)

        # gated repulsion: only when regret indicates multi-basin necessity
        repel_gate = (regret_proxy > self.regret_delta).float()
        repel = self.repulsion_loss(z_modes) if self.M >= 2 else torch.tensor(0.0, device=best.device)

        # Total loss: WTA + (optional) repulsion
        loss = best + (self.repel_weight * repel_gate) * repel

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update head usage EMA
        with torch.no_grad():
            onehot = torch.zeros(self.M, device=loss.device)
            onehot[m_star] = 1.0
            self.head_usage_ema.mul_(0.995).add_(0.005 * onehot)

        # optional reseed (recycle capacity)
        self._maybe_reseed(z_modes.detach(), losses_t.detach(), regret_proxy.detach())

        return {
            "loss": float(loss.item()),
            "best_loss": float(best.item()),
            "best_head": int(m_star.item()),
            "regret_proxy": float(regret_proxy.item()),
            "repel_gate": float(repel_gate.item()),
            "repel": float(repel.item()),
            "best_leak": float(leaks_t[m_star].item()),
            "best_vol": float(vols_t[m_star].item()),
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

        z1 = self.encode(v1.reshape(B, -1))
        z2 = self.encode(v2.reshape(B, -1))

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
        trajectory_delta_hint: Optional[torch.Tensor] = None,
        pop_size: Optional[int] = None,
        n_iters: Optional[int] = None,
        force_z_only: bool = False,
        L_refine: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Two-stage inference:
          A) Score M router means cheaply (no CEM), pick top-L.
          B) Run short CEM from each selected mean, pick best resulting z.

        trajectory_delta_hint (optional) can be used to *bias* head ranking by similarity
        to encoder(trajectory_delta_hint), but remains unsupervised.
        """
        if s0.dim() != 1:
            raise ValueError("select_z_geometric_multimodal expects s0: (obs_dim,)")

        pop_size = int(pop_size or self.cem_pop_size)
        n_iters = int(n_iters or self.cem_iters)
        L = int(min(L_refine or self.L_refine, self.M))

        z_modes_B, pi = self.propose_z_modes(s0, force_z_only=force_z_only)  # (1,M,D)
        z_modes = z_modes_B.squeeze(0)  # (M,D)

        # Optional: hint vector from encoder, used only to rank means (not required).
        hint_score = 0.0
        z_hint = None
        if trajectory_delta_hint is not None:
            z_hint = self.encode(trajectory_delta_hint.reshape(-1))
            # cosine similarity to hint
            hint_score = (z_modes @ z_hint).detach()  # (M,)

        # Cheap energy scoring on means
        mean_scores = self._score_z_candidates(s0, z_modes, force_z_only=force_z_only)  # higher is better
        if z_hint is not None:
            mean_scores = mean_scores + 0.5 * hint_score  # small bias; keep geometry dominant

        top_idx = torch.topk(mean_scores, k=L, largest=True).indices

        best_z = z_modes[top_idx[0]]
        best_score = mean_scores[top_idx[0]].item()

        # refine each selected mean with short CEM
        for idx in top_idx:
            z0 = z_modes[idx]
            z_ref, score_ref = self._cem_refine_from_mean(
                s0, z0, pop_size=pop_size, n_iters=n_iters, force_z_only=force_z_only
            )
            if score_ref > best_score:
                best_score = score_ref
                best_z = z_ref

        return best_z

    @torch.no_grad()
    def _score_z_candidates(self, s0: torch.Tensor, z: torch.Tensor, force_z_only: bool = False) -> torch.Tensor:
        """
        Geometry-based candidate scoring (higher is better).
        Uses only tube properties + anchors, no labels.
        z: (N, z_dim)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        N = z.shape[0]
        mu, sigma = self.get_tube(s0.unsqueeze(0), z, force_z_only=force_z_only)  # (N,T,d)

        s0_pos = s0[: self.pred_dim]
        start_error = torch.norm(mu[:, 0, :] - s0_pos, dim=-1)
        start_penalty = - (start_error ** 2) * 100.0

        displacement = torch.norm(mu[:, -1, :] - mu[:, 0, :], dim=-1)
        progress_reward = torch.log(displacement + 1e-6) * 3.0

        agency_score = -sigma.mean(dim=(1, 2)) * 0.3
        sigma_consistency = -sigma.std(dim=(1, 2))

        return start_penalty + progress_reward + agency_score + sigma_consistency

    @torch.no_grad()
    def _cem_refine_from_mean(
        self,
        s0: torch.Tensor,
        z_mean: torch.Tensor,
        pop_size: int,
        n_iters: int,
        force_z_only: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        Short CEM around provided mean (unit vector). Returns best z and its score.
        """
        curr_mean = F.normalize(z_mean, dim=-1)
        curr_std = torch.ones_like(curr_mean) * self.cem_init_std

        n_elites = max(8, int(pop_size * self.cem_elite_frac))

        best_z = curr_mean
        best_score = -1e9

        for _ in range(n_iters):
            cand = curr_mean + torch.randn(pop_size, self.z_dim, device=s0.device) * curr_std
            cand = F.normalize(cand, p=2, dim=1)

            scores = self._score_z_candidates(s0, cand, force_z_only=force_z_only)
            top = torch.topk(scores, k=n_elites, largest=True)
            elites = cand[top.indices]

            # update
            curr_mean = F.normalize(elites.mean(dim=0), dim=-1)
            curr_std = elites.std(dim=0) + 1e-6

            # track best
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