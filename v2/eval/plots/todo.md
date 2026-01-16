Below are a few **high-D friendly** visualizations that work well with your exact objects (`mu(t)∈R^d`, `sigma(t)∈R^d`, knots, and energy/score). I’ll reference your existing files:

- `geometry.py`: `plot_tube_overlay(...)` currently plots a few coordinates.
- `latent.py`: `plot_latent_scatter(...)` uses PCA/2D scatter.

---

## 1) Replace “plot dims” with a tube heatmap (time × dimension)

**Why it works in high-D:** you see _all_ coordinates, but in a compact way. The trick is to **sort dimensions** by something meaningful so the picture is interpretable.

### What to plot

- Heatmap of `mu` (centerline) and `sigma` (width) as two panels
- Sort dimensions by either:
  - overall magnitude `mean_t |mu[t, j]|`, or
  - variability `std_t mu[:, j]`, or
  - “importance” relative to energy (if you have it)

### Visual read

- Vertical structure → “which dims are involved”
- Horizontal structure → “when commitment shifts”
- High sigma bands → “slack / uncertainty directions”

**Add this function to `geometry.py`:**

```python
def plot_tube_heatmap(mu: np.ndarray, sigma: np.ndarray, output_path: Path,
                      sort_by: str = "mu_std", title: str = "Tube heatmap"):
    """
    High-D tube visualization: heatmaps of mu and sigma over (time x dim).
    sort_by: "mu_std" | "mu_abs_mean" | "sigma_mean"
    """
    import matplotlib.pyplot as plt
    T, d = mu.shape

    if sort_by == "mu_std":
        order = np.argsort(mu.std(axis=0))[::-1]
    elif sort_by == "mu_abs_mean":
        order = np.argsort(np.abs(mu).mean(axis=0))[::-1]
    elif sort_by == "sigma_mean":
        order = np.argsort(sigma.mean(axis=0))[::-1]
    else:
        order = np.arange(d)

    mu_s = mu[:, order].T        # (d, T)
    sig_s = sigma[:, order].T    # (d, T)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    im0 = axes[0].imshow(mu_s, aspect="auto")
    axes[0].set_title("mu (sorted dims)")
    axes[0].set_xlabel("time"); axes[0].set_ylabel("dimension (sorted)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(sig_s, aspect="auto")
    axes[1].set_title("sigma (sorted dims)")
    axes[1].set_xlabel("time")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
```

This alone usually beats “first 3 dims” by a mile.

---

## 2) Sliced projections (random directions) + “tube envelopes”

Instead of choosing coordinate axes, you choose **random unit vectors** \(u_k\) and project the entire tube:

- center: \( m_k(t) = u_k^\top \mu(t) \)
- width: \( w*k(t) \approx \sqrt{\sum_j (u*{k,j}^2 \sigma_j(t)^2)} \) (assuming diagonal sigma, which is what you have)

Then plot `m_k(t)` with `±w_k(t)` for **K directions** (say 8–16). This is “how a high-D tube looks from many viewpoints”.

### Why it’s interpretable

- If the tube is truly “geometric commitment”, you’ll see consistent structure across many slices.
- If commitments are coordinate artifacts, slices look noisy/inconsistent.

**Add to `geometry.py`:**

```python
def plot_sliced_tube(mu: np.ndarray, sigma: np.ndarray, output_path: Path,
                     K: int = 12, seed: int = 0, title: str = "Sliced tube projections"):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    T, d = mu.shape

    U = rng.normal(size=(K, d))
    U /= (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)  # (K,d)

    # projections
    m = mu @ U.T                         # (T,K)
    w = np.sqrt((sigma**2) @ (U**2).T)   # (T,K)

    fig, axes = plt.subplots(K, 1, figsize=(12, 2.2*K), sharex=True)
    if K == 1: axes = [axes]
    t = np.arange(T)

    for k in range(K):
        ax = axes[k]
        ax.plot(t, m[:, k], linewidth=1.8)
        ax.fill_between(t, m[:, k] - w[:, k], m[:, k] + w[:, k], alpha=0.2)
        ax.set_ylabel(f"slice {k}")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("time")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
```

This scales to `d=256+` cleanly.

---

## 3) High-D “knot” visualization: knot matrix SVD + heatmap

Your knots are _the_ commitment parameters. In high-D, treat them as a matrix:

- `mu_knots`: shape `(n_knots, d)`

Two views that work:

### A) Knot heatmap (knots × dim) with dim sorting

Same idea as tube heatmap, but focused on commitments rather than interpolation.

### B) Low-rank structure (SVD/PCA) on knots

Project knot vectors to 2–3 components and plot **knot index vs component value**.
That answers: _is the plan essentially low-rank, or does it truly use many degrees of freedom?_

If you want just one: do **SVD on mu_knots** and show first 3 components as lines over knot index.

---

## 4) Latent/port visualization that stays meaningful as z_dim grows

Your current `latent.py` does PCA/2D scatter. That becomes misleading in high-D because “clusters” are often projection artifacts.

Two upgrades that scale better:

### A) Cosine-similarity matrix + hierarchical ordering

You already have `plot_cosine_similarity_hist(...)`. The _matrix_ is more informative:

- Compute `C = z z^T` (cosine similarity)
- Order by hierarchical clustering / seriation
- Display as heatmap

You’ll immediately see:

- distinct head basins (block-diagonal structure)
- collapse (uniform bright block)
- diversity (multiple blocks)

### B) Sliced “energy vs direction” diagnostics

For a fixed `s0`, sample many `z` and compute `energy(z)` (or `-score(z)`).
Then show:

- histogram of energies
- scatter of energy vs angle in top 2 PCs **and** energy vs cosine similarity to each head

That last one is _great_ for “ports”: plot `cos(z, z_head_m)` on x, energy on y, for each m.

---

## 5) Scalar “commitment summaries” (the things your eye can track)

When `d` is big, you want a few scalars over time that are stable and comparable:

Good ones for your tube:

- **Center speed:** `||mu[t+1]-mu[t]||2`
- **Tube thickness:** `mean_j sigma[t,j]` or `||sigma[t]||2`
- **Relative thickness:** `||sigma[t]||2 / (||mu[t]||2 + eps)`
- **“Volume proxy”:** `mean_j log(sigma[t,j] + eps)` (tracks shrink/expand)

Plot those as time series (one figure). This gives a quick read: _tight early commitment vs late commitment_, etc.

---

## What I’d do to make your current two plots “work” in high-D

Given your existing code, the highest-leverage changes are:

1. Add `plot_tube_heatmap(...)` (time×dim)
2. Add `plot_sliced_tube(...)` (K random projections)
3. Add a `plot_cosine_similarity_matrix(...)` for latents / ports (clustered heatmap)

Those three cover:

- **tube geometry** in high-D
- **commitment directions** (via slices)
- **latent / head structure** without fragile 2D scatter
