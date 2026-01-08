
# Patch plan: z-memory replay buffer + memory regularizer + plots

## 0) High-level design choices (keep simple)
- **Memory is over z only** (no s0-conditioning yet).
- Use **kNN + Gaussian kernel** in z-space to estimate local “risk”.
- Add a **regularizer** that penalizes sampling z in high-risk neighborhoods.
- Add a **2D visualization** by projecting z’s with PCA (or random projection if you prefer).

This gives you “regions of z” behaving like soft ports, without IDs.

---

## 1) Actor: add a replay buffer of commitment outcomes

### 1.1 Add fields in `__init__`
Add hyperparams and buffers:

```python
from collections import deque

# ---- commitment memory (replay in z-space) ----
self.mem_size = 50_000
self.mem_min = 500          # don’t apply regularizer until enough data
self.mem_k = 64             # neighbors
self.mem_sigma = 1.0        # kernel width in z-space
self.w_mem = 0.02           # strength of memory regularizer (start small)
self.mem_detach_targets = True

self.mem = deque(maxlen=self.mem_size)  # store dicts per episode

# logging
self.history["mem_loss"] = []
self.history["mem_risk"] = []
self.history["mem_n"] = []
```

### 1.2 Define what you store per episode
After each `train_on_episode`, append:

- `z` (detached CPU tensor)
- `soft_bind_rate`
- `cone_vol_w` (or `sharp_log_vol` if you prefer)
- `E_T_train`
- `lambda_bind`, `lambda_T` (optional but useful)

Example:

```python
with torch.no_grad():
    self.mem.append({
        "z": z.detach().cpu().squeeze(0),  # [z_dim]
        "soft_bind": float(soft_bind_rate.item()),
        "cone_vol": float(cone_vol_w.item()),
        "E_T": float(E_T_train.item()),
        "lambda_bind": float(self.lambda_bind),
        "lambda_T": float(self.lambda_T),
        "Hr": int(Hr) if "Hr" in self.history else 0,
    })
```

(Keep it episode-level. Don’t store per-step trajectories.)

---

## 2) Actor: implement the memory “risk” estimator (kNN + kernel)

### 2.1 Add helper to fetch tensors from memory
Add a method:

```python
def _mem_tensors(self, device):
    # returns Z: [N,z_dim], and arrays for metrics
    zs = torch.stack([m["z"] for m in self.mem], dim=0).to(device)  # [N,z_dim]
    soft = torch.tensor([m["soft_bind"] for m in self.mem], device=device)
    cone = torch.tensor([m["cone_vol"] for m in self.mem], device=device)
    lam  = torch.tensor([m["lambda_bind"] for m in self.mem], device=device)
    return zs, soft, cone, lam
```

### 2.2 Define a scalar “risk” target per memory item
Keep it minimal and aligned with your story:

- low bind = bad
- high cone volume = bad (wide / vague)
- high lambda_bind = bad (had to pay a lot to make it reliable)

Define:

\[
r_i = a(1-\text{soft\_bind}_i) + b\log(\text{cone\_vol}_i+\epsilon) + c\lambda_{\text{bind},i}
\]

In code:

```python
def _mem_risk_targets(self, soft, cone, lam):
    eps = 1e-8
    a, b, c = 1.0, 0.2, 0.05  # start here; tune later
    return a * (1.0 - soft) + b * torch.log(cone + eps) + c * lam
```

### 2.3 kNN kernel risk estimate for current z
Add method:

```python
def memory_risk(self, z: torch.Tensor) -> torch.Tensor:
    """
    z: [1,z_dim]
    returns scalar estimated risk in neighborhood of z
    """
    if len(self.mem) < self.mem_min:
        return torch.tensor(0.0, device=z.device)

    Z, soft, cone, lam = self._mem_tensors(z.device)  # [N,z_dim], [N]
    r = self._mem_risk_targets(soft, cone, lam)       # [N]

    zq = z.squeeze(0)                                  # [z_dim]
    d2 = torch.sum((Z - zq) ** 2, dim=-1)             # [N]

    # pick k nearest
    k = min(self.mem_k, d2.numel())
    vals, idx = torch.topk(d2, k=k, largest=False)

    d2_k = vals
    r_k = r[idx]

    # gaussian kernel weights
    sigma2 = float(self.mem_sigma) ** 2
    w = torch.exp(-d2_k / (2.0 * sigma2))
    w = w / (w.sum() + 1e-8)

    # local expected risk
    return torch.sum(w * r_k)
```

Note: everything here is differentiable wrt `z` (but not wrt stored memory). That’s what you want.

---

## 3) Actor: add the memory regularizer to training

### 3.1 Where to add it
Inside `train_on_episode`, after you’ve computed the main loss terms (exp_nll, cone loss, etc.), before backward.

### 3.2 The regularizer
Compute:

```python
mem_risk = self.memory_risk(z)
mem_loss = self.w_mem * mem_risk
loss = loss + mem_loss
```

Recommended: **detach the memory targets** (they’re already detached because they’re stored as floats).

### 3.3 Log it
At end of episode:

```python
self.history["mem_loss"].append(float(mem_loss.detach().item()))
self.history["mem_risk"].append(float(mem_risk.detach().item()))
self.history["mem_n"].append(int(len(self.mem)))
```

---

## 4) Experiment: add an evaluation/visualization pass

You want a plot that answers:
- Are there *basins* / regions in z?
- Are those regions associated with reliability vs cone tightness?
- Does the actor *return* to the same regions?

### 4.1 Add a function to export memory to numpy
In experiments.py:

```python
def extract_memory(agent: Actor):
    zs = np.stack([m["z"].numpy() for m in agent.mem], axis=0)  # [N,z_dim]
    soft = np.array([m["soft_bind"] for m in agent.mem])
    cone = np.array([m["cone_vol"] for m in agent.mem])
    lam  = np.array([m["lambda_bind"] for m in agent.mem])
    return zs, soft, cone, lam
```

### 4.2 Project z to 2D (PCA)
Use numpy SVD PCA (no sklearn dependency):

```python
def pca2(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T  # [N,2]
```

### 4.3 Plot: z-map colored by outcomes
Add a 2x2 grid:

- colored by `soft_bind`
- colored by `log(cone_vol)`
- colored by `lambda_bind`
- colored by `mem_risk_target` (the same scalar you regularize)

```python
def plot_z_memory_map(agent: Actor, max_points=5000):
    zs, soft, cone, lam = extract_memory(agent)

    if zs.shape[0] == 0:
        print("No memory to plot.")
        return

    # subsample for speed
    N = zs.shape[0]
    if N > max_points:
        idx = np.random.choice(N, size=max_points, replace=False)
        zs, soft, cone, lam = zs[idx], soft[idx], cone[idx], lam[idx]

    Z2 = pca2(zs)

    risk = (1.0 - soft) + 0.2 * np.log(cone + 1e-8) + 0.05 * lam

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sc0 = axes[0,0].scatter(Z2[:,0], Z2[:,1], c=soft, s=10, alpha=0.5)
    axes[0,0].set_title("z-memory: soft_bind")
    plt.colorbar(sc0, ax=axes[0,0])

    sc1 = axes[0,1].scatter(Z2[:,0], Z2[:,1], c=np.log(cone + 1e-8), s=10, alpha=0.5)
    axes[0,1].set_title("z-memory: log cone_vol")
    plt.colorbar(sc1, ax=axes[0,1])

    sc2 = axes[1,0].scatter(Z2[:,0], Z2[:,1], c=lam, s=10, alpha=0.5)
    axes[1,0].set_title("z-memory: lambda_bind")
    plt.colorbar(sc2, ax=axes[1,0])

    sc3 = axes[1,1].scatter(Z2[:,0], Z2[:,1], c=risk, s=10, alpha=0.5)
    axes[1,1].set_title("z-memory: risk (regularizer target)")
    plt.colorbar(sc3, ax=axes[1,1])

    for ax in axes.ravel():
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    plt.tight_layout()
    plt.show()
```

### 4.4 Plot: “revisit” / path structure (optional but very telling)
If you also store a rolling “recent z’s” list during training, you can draw the last 1000 points in order as a faint line. That makes “attractors” visually obvious.

Minimal: during training keep `history["z_sample"]` as list of z vectors (or 2D PCA computed later). Then plot a time-colored trail.

---

## 5) Acceptance criteria (what success looks like)

After one run, you want to see:

1. **Non-uniform density** in PCA z-map (islands / ridges instead of a blob).
2. Those islands correlate with **high soft_bind** and/or **low lambda_bind**.
3. As training progresses, the agent’s sampled z’s **cluster more** and the memory regularizer **stops spiking** (meaning it learned to avoid known-bad neighborhoods).

If you get (1) but not (2), memory isn’t aligned with your reliability measure.
If you get (2) but not (1), z isn’t being reused (KL too high, or actor too stochastic).

---

## 6) One recommended default setting
Start with:
- `mem_min=500`
- `mem_k=64`
- `mem_sigma=1.0` (increase if z norms are large; decrease if everything looks “far”)
- `w_mem=0.02` (keep it small until the map looks structured)
