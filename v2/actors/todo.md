Yeah — there are two separate things going on in what you shared:

1. **Why “gauge fixed ✓” in the plot but `gauge_fixed: false` in JSON/logs**
2. **Why this actor tends to “wash out” and fail to form strong attractors as \(d, K, z_dim\) grow**

I’ll hit both concretely.

---

## 1) The gauge-fixed discrepancy is real and it’s coming from mismatched success criteria

In your `test_reference_probe.py` you have **two different definitions** of “gauge fixed”:

### A) Plot criterion (in `plot_reference_probe_results`)

```py
Gauge fixed: ✓ if final_comp > 0.8 and final_probe < 0.2
```

Here `final_probe` is **the probe consistency loss** `L_probe` (a cosine _loss_, lower is better).

### B) Logged/JSON criterion (in `run_reference_probe_test`)

```py
'gauge_fixed': final_comp_sim > 0.8 and probe_agreement['probe_sim_AB'] > 0.8
```

Here you ignore `L_probe` entirely and instead threshold **probe cosine similarity** `probe_sim_AB` (higher is better).

So it’s totally possible to get:

- `L_probe ≈ 0.166` (looks “good enough” under the plot threshold `< 0.2`)
- while `probe_sim_AB ≈ 0.72` (fails the log threshold `> 0.8`)

Those are consistent because:

- cosine loss and cosine similarity are related by `loss = mean(1 - cos)` **but** you’re not using the same pairs/averaging:
  - `L_probe` averages **six directed constraints** (A→B, A→C, B→A, B→C, C→A, C→B)
  - `probe_sim_AB` checks only **A→B**

### Fix (recommended)

Pick **one** gauge-fixed definition and use it everywhere.

A robust definition for triadic gauge is:

- composition must be good
- and probe agreement must be good for **both** A→B and A→C (or min over all directed pairs)

Example:

```py
gauge_fixed = (
    final_comp_sim > 0.9 and
    min(probe_sim_AB, probe_sim_AC) > 0.9
)
```

or if you want to use loss:

```py
gauge_fixed = (final_comp_sim > 0.9 and final_probe_loss < 0.05)
```

Also: since `L_probe` is averaged over 6 maps, I’d _either_ log all 6 probe sims _or_ compute a min/mean across them and use that.

---

## 2) Why this actor struggles to scale: the big blockers (and how to change them)

From your plots (“variance between modes is very low”, tubes overlap ~100%) and your scaling table (“composition stays high but probe agreement collapses”), the main symptom is:

> **The actors aren’t producing stable, separated basins in z-space in the first place**, so probes aren’t reliable anchors.

Here are the most likely _mechanistic_ reasons in this Actor.

---

### (A) Your contract objective becomes exponentially hard in high dimension

This line is the killer:

```py
leakage = relu(|traj - mu| - sigma*k)^2 .mean()
```

Because you’re using **per-dimension bands** (an axis-aligned tube) and measuring violations per-dim, in high dimension you get the classic curse:

- even if each dimension individually fits “pretty well”, the chance that **all dims** are inside the tube at every time is low
- the easiest way out is: **increase sigma everywhere** and/or collapse mu toward an average → which is exactly what you’re observing

#### Concrete fix: use a _radial_ or _Mahalanobis_ constraint

Instead of per-dim `abs`, compute a normalized L2 residual:

```py
r2 = ((trajectory - mu) / (sigma + eps)).pow(2).sum(dim=-1)  # over dims
leak = relu(r2 - c).mean()
```

Where `c` is a chi-square-ish threshold (e.g. `c = pred_dim` or a quantile). This makes “coverage” scale sanely with dimension.

If you want to keep diagonal sigma, this is the single best upgrade.

---

### (B) Your loss weights are not dimension-normalized

Several terms scale with \(d\) implicitly:

- `start_err` uses mean over dims (ok-ish), but your **penalty weight** is fixed (`5.0`)
- `direction_loss` in high d becomes noisy (cosine similarities concentrate)
- in `_score_z_candidates`, `start_penalty = -(||mu0 - s0||^2)*100` grows with d (norm grows ~sqrt(d) → squared grows ~d)

So as `pred_dim` increases, your “energy landscape” can get dominated by a couple terms in unpredictable ways.

#### Concrete fix: normalize by dimension

Any place you use norms or squared norms, scale by `pred_dim`:

- use `||x||^2 / pred_dim`
- or `||x|| / sqrt(pred_dim)`

And re-tune constants once, not per-d.

---

### (C) Capacity bottleneck: your decoders are _way_ too small for large `pred_dim`

These two are suspicious:

```py
self.mu_net:  (s0_emb_dim + 256) -> 64 -> (n_knots * pred_dim)
self.sigma_net: z_dim -> 64 -> (n_knots * pred_dim)
```

If `pred_dim=50` and `n_knots=5`, outputs are 250 dims; that’s already large. If `pred_dim=8` it’s 40 dims — easy.

In high d you’re asking a 64-wide hidden layer to generate a high-dimensional structured object (a time-parameterized centerline and width field). It will underfit and regress-to-mean.

#### Concrete fix: scale width with pred_dim

At minimum:

- make those hidden layers 256 or 512
- add LayerNorm
- make mu_net deeper (2–3 hidden layers)

Example:

```py
hid = max(256, 4*pred_dim)
mu: (s0_emb+256)->hid->hid->(n_knots*pred_dim)
sigma: z_dim->hid->hid->(n_knots*pred_dim)
```

Also consider giving sigma_net access to _some_ s0 info in high-d (even a tiny conditioning) because “sigma depends only on z” can be too rigid when the environment’s uncertainty is strongly state-dependent.

---

### (D) WTA training + regret-gated repulsion is likely collapsing heads in high d

WTA has a known pathology: early lucky heads win, others starve → diversity dies.

Your repulsion gate:

```py
regret_proxy = loss2 - loss1
repel_gate = (regret_proxy > regret_delta)
```

In high d, losses tend to:

- be larger and noisier
- but also **more similar across heads** (because everything underfits similarly)

So `loss2-loss1` can easily be _small_, meaning the repulsion gate may rarely activate → heads collapse.

#### Concrete fixes (pick one)

1. **Soft-WTA / responsibilities**: backprop through top-k heads with weights `softmax(-loss/τ)`
2. Keep WTA, but **always apply a small repulsion** and anneal it:
   - early: strong repulsion (force exploration)
   - later: gated repulsion (allow collapse if truly unnecessary)
3. Use an explicit **head diversity prior** in router outputs (entropy regularizer on `pi`)

---

### (E) Your router depends on an 8D embedding (`s0_emb_dim=8`) even when the world is large

```py
obs_proj: obs_dim -> 8
z_router: 8 -> 64 -> M*z_dim
z_logits: 8 -> 64 -> M
```

In higher-dimensional environments, cramming state into 8 dims often forces the router to output nearly the same z-modes everywhere → again: attractor collapse.

#### Fix

Scale `s0_emb_dim` with d (or obs_dim). Rule of thumb:

- `s0_emb_dim = min(128, max(32, pred_dim*2))`

---

### (F) Your “data” is random rollouts during actor training in the probe tests

In `train_actor` inside the probe experiment, you use random actions:

```py
action = np.random.randn(...) * 0.3
```

If the environment’s multi-modality only shows up under purposeful behavior (or the “fork” depends on early commitment), random rollouts won’t reliably produce separable trajectory families. In higher d, random policies tend to produce “diffuse” clouds that look similar across modes.

So even a perfect learner would struggle to discover K basins.

#### Fix

Train actors on rollouts that _expose forks_:

- scripted “goal-seeking” rollouts per mode
- or teacher policies
- or at least biased exploration that reaches distinct basins

---

## What I would change first (minimal intervention, highest ROI)

If you want the smallest set of changes that most directly target “no attractors at high d / high K”, do this in order:

1. **Replace leakage with a Mahalanobis/radial residual** (fixes exponential pain with d)
2. **Increase mu/sigma decoder capacity** (64 → 256/512 + LayerNorm)
3. **Scale `s0_emb_dim` up with dimension**
4. **Switch WTA → soft-WTA(top-k)** or add a small always-on repulsion annealed down
5. **Train on data that actually contains K distinct futures** (not random rollouts)

Do those and you should see:

- higher inter-head diversity
- lower tube overlap across heads
- better probe stability
- and then probe-based gauge alignment will scale _much_ better without needing O(K²) probes.
