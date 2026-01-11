## ✅ COMPLETED: Homeostasis as _self-calibration_ via a commitment market (no fixed \(\rho\), no fixed \(\alpha/\beta\))

This is the most “TA-native” way to avoid choosing a bind target _and_ avoid hand-picking a cost ratio:

### Key idea

Let the agent maintain an internal scalar “commitment price” \(\lambda\) that adjusts based on realized failure frequency, but **without a predetermined target**. Instead, the update tries to equalize _marginal_ rates:

- If failures are frequent, failures become “expensive” → widen cones.
- If failures are rare, cones are “too loose” relative to what you can afford → tighten cones.

The operating point is the **fixed point** where the marginal benefit of tightening equals the marginal cost of added failures.

Concretely, train the cone by minimizing a “free energy”:

\[
L = L\_\mu \;+\; \alpha\,V(\Phi) \;+\; \lambda\,\text{Fail}
\]

and update \(\lambda\) with a _homeostatic rule_ that depends on **observed surprise of failure** rather than a target rate, e.g.:

- increase \(\lambda\) when failures occur in “easy” situations (unexpected failures)
- decrease \(\lambda\) when failures occur only in “hard” situations (expected failures)

To do this, you need a notion of “hardness”—which you can get from your world model or even from residual prediction error. That gives you a selection principle:

> only pay a lot to avoid failures that were preventable.

This produces an emergent bind rate that depends on environment structure, not a global \(\rho\).

**In your current identity-Infer toy**, a simple proxy for hardness is trajectory deviation from the mean: large residuals are “hard.” Then your \(\lambda\) update depends on residual-weighted failures.

Tube volume hitting the _floor_ means your current incentives make “tighten sigma” strictly better than “avoid failure” **in the gradients the model actually sees**. In other words, your failure term isn’t providing a strong (or smooth) opposing force as you shrink \(\sigma\), so optimization runs to the clamp.

There are a few very specific issues in the code that cause exactly this behavior, plus some fixes that preserve your “no fixed bind target” philosophy while producing a real equilibrium.

---

## 1) The biggest bug: your λ update ignores _how much_ you failed

You collapse failure to a boolean:

```py
failed = fail.item() > 0.5
```

So “fail=0.55” and “fail=1.0” are treated the same, and “fail=0.49” is treated as full success. That destroys the continuous feedback needed for homeostasis and makes λ dynamics coarse/noisy—often too weak to counter volume shrink.

### Fix

Use the **continuous fail rate** in both surprise and λ update:

- let `failed_amount = fail.item()` in \([0,1]\)
- weight updates by `failed_amount`

Example:

```py
fail_amt = float(fail.item())  # 0..1

surprise_fail = (-hardness)  # bigger when easy
delta = self.eta_lambda * fail_amt * (1.0 + max(0.0, surprise_fail))
```

And for success, scale by (1 - fail_amt):

```py
success_amt = 1.0 - fail_amt
delta = -self.eta_lambda * success_amt * (0.5 + max(0.0, -hardness) * 0.3)
```

This alone usually stops “slam to floor.”

---

## 2) Your failure penalty has terrible gradients (hard indicator bind)

Your bind/fail uses a hard box indicator:

```py
inside = (abs(err) < k*sigma).all(dim=-1).float()
```

That is **piecewise constant** w.r.t. \(\sigma\) and \(\mu\) almost everywhere → the model gets almost no gradient signal from `fail` until it crosses the boundary, then it jumps.

So the optimizer mostly sees:

- strong gradient from `alpha_vol * log(sigma)` pushing down
- weak/zero gradient from `fail` pushing up

=> volume collapses to clamp.

### Fix: use a differentiable “soft bind” for training

Keep hard bind for reporting, but train with a smooth proxy. A simple one:

Per timestep, per dim margin:
\[
m*{t,d} = k\sigma*{t,d} - |e*{t,d}|
\]
Soft-inside probability:
\[
p*{t,d} = \sigma(\gamma m*{t,d})
\]
Soft inside timestep:
\[
p_t = \prod_d p*{t,d}
\]
Soft bind:
\[
\text{bind}_\text{soft} = \frac{1}{T}\sum_t p_t
\]
Soft fail = \(1-\text{bind}_\text{soft}\)

Choose \(\gamma\sim 10\)–50 depending on scale.

This makes failure pressure continuous and lets an actual equilibrium form.

---

## 3) Log-volume is unbounded below + you clamp sigma too low

You’re minimizing:

```py
log_vol = log(sigma).mean()
```

If \(\sigma\to 0\), log-vol \(\to -\infty\). That’s _fine_ only if some other term prevents it. Right now, your other term (`fail`) often can’t, for gradient reasons.

### Fix options (pick 1–2)

**(A) Put a soft barrier near the floor**  
Even if you clamp for numerical stability, add a barrier so the optimizer “feels” it:

\[
L*{\text{barrier}} = \mathbb{E}[\mathrm{softplus}(\log\sigma*{\min} - \log\sigma)]
\]

**(B) Use volume relative to a baseline**  
Instead of absolute \(\log\sigma\), minimize deviation from a learned/reference scale:

\[
L*V = \mathbb{E}\big[\log(\sigma/\sigma*{\text{ref}})\big]
\]
where \(\sigma\_{\text{ref}}\) can be an EMA of residuals or a per-rule baseline. This makes “optimal” mean something in environment units.

**(C) Penalize _changes_ in volume (homeostasis prior)**  
Add a stabilization term:
\[
\eta \, (\log V*t - \log V*{t-1})^2
\]
Not necessary, but helps prevent drift.

---

## 4) Your “hardness” is coupled to μ accuracy, not environment uncertainty

Right now, hardness is residual \(|\tau-\mu|\). As training improves \(\mu\), hardness drops _everywhere_, making failures look “more surprising,” pushing λ up, pushing tubes tighter… until you hit the floor.

So the system can converge to “I got good at μ, so now I can always tighten,” even in noisy regimes, because hardness is tracking _model error_ not _irreducible noise_.

### Fix: define hardness in a way that tracks aleatoric difficulty

Since you have an environment with genuine stochasticity (rule 3), you can estimate irreducible difficulty by:

**(A) Ensemble/MC residual variance**
Sample multiple z’s (or dropout) and look at variability of predicted μ, or compare residuals across multiple episodes with the same condition.

**(B) Per-rule residual baseline**
Since rule is in observation, maintain separate running stats per rule (or per cluster) so “hard” isn’t washed out by easy cases.

Minimal change: keep running residual stats **per rule**:

- `running_residual_mean[rule]`
- `running_residual_std[rule]`

Then hardness is relative to the appropriate regime.

**(C) Use “normalized error”**
Hardness based on ratio:
\[
h = \frac{\mathbb{E}|e|}{\mathbb{E}\sigma} \quad\text{or}\quad \frac{\mathbb{E}|e|}{\text{EMA}(\mathbb{E}|e|)}
\]
So if sigma collapses, hardness rises, driving λ up.

That creates negative feedback.

---

## 5) Your λ dynamics are not tied to a stationary equilibrium condition

Right now λ is updated by heuristics that don’t correspond to optimizing a well-defined saddle point. That’s okay—but if you want real homeostasis, you need a **fixed point condition** like:

> expected marginal penalty from tightening = expected marginal penalty from failures.

A simple way to get that without a fixed \(\rho\):

### Use dual ascent on an _endogenous_ constraint: “calibrated failure surprise should be ~0”

Define a surprise-weighted constraint:
\[
g = \mathbb{E}[ \text{fail} \cdot \text{surprise} ]
\]
and update:
\[
\lambda \leftarrow \max(0, \lambda + \eta \, g)
\]

Interpretation:

- if you often fail in “easy” situations (positive surprise), λ increases
- if failures are mostly unsurprising, λ decreases

But crucially: this uses continuous fail and provides a fixed point at \(g\approx 0\), not at an arbitrary \(\rho\).

---

## 6) The most effective minimal patch (my recommendation)

If you want the smallest set of changes that usually fixes “volume hits floor”:

1. Replace hard fail with **soft fail** (differentiable)
2. Use continuous fail amount in λ update
3. Compute hardness per-rule (or normalize hardness by sigma)

That’s it.

### What you should see after these changes

- easy rules → small sigma, high bind
- noisy rule → larger sigma, lower bind
- log_vol stabilizes above the clamp
- λ stabilizes (no drift to min or max)

---

## 7) Concrete pseudocode changes (drop-in)

### Soft fail

```py
def compute_soft_fail(mu, sigma, traj, k=2.0, gamma=25.0):
    # margin: positive means inside
    margin = k * sigma - torch.abs(traj - mu)     # [T, D]
    p_dim = torch.sigmoid(gamma * margin)         # [T, D]
    p_t = torch.prod(p_dim, dim=-1)               # [T]
    bind_soft = p_t.mean()
    fail_soft = 1.0 - bind_soft
    return bind_soft, fail_soft
```

### λ update (continuous)

```py
fail_amt = float(fail_soft.item())     # 0..1
success_amt = 1.0 - fail_amt

# surprise for failures: -hardness (easy => big)
surp_fail = max(0.0, -hardness)

delta = self.eta_lambda * (fail_amt * (1.0 + surp_fail) - success_amt * 0.2)
self.lambda_fail = clip(self.lambda_fail + delta, 0.0, self.lambda_max)
```

### Hardness normalization (quick)

```py
# normalize by current sigma scale to create negative feedback
sigma_scale = float(out.sigma.mean().item())
res = float(residual.item())
hardness = (res / (sigma_scale + 1e-6))  # bigger if sigma too small
```

This makes “sigma too tight” automatically look “hard,” increasing λ.

---

## 8) If you _still_ want “no hyperparameters decide the point”

You still need _some_ selection principle. In your current formulation, \(\alpha*{\text{vol}}\) is already that principle. If \(\alpha*{\text{vol}}\) is high, the optimum moves toward smaller volumes; if λ isn’t allowed to dominate, you’ll hit the floor.

So even philosophically, you can’t escape:

- either specify a target (ρ), or
- specify costs (α vs failure price dynamics), or
- specify a budget.

What you _can_ do is make the selection principle **intrinsic** (e.g., "failures that were preventable are expensive"), which is what your surprise idea is aiming at—but it needs smooth fail + proper scaling to work.

---

## Implementation Results (2026-01-10)

All fixes implemented in `homeostasis_actor.py` and `train_homeostasis.py`.

### Key Changes Made:

1. **Soft fail** via `compute_soft_bind_fail()` with sigmoid on margin (γ=25)
2. **Continuous fail amount** in λ update (not boolean)
3. **Hardness = residual / σ** (creates negative feedback)

### Results:

```
Easy (rule 0): bind=1.000, σ=0.062, log_vol=-2.78
Hard (rule 3): bind=0.924, σ=0.160, log_vol=-1.85

✓ Sigma calibrated: hard 2.6x wider than easy
✓ Bind rate calibrated: hard < easy
✓ Volume NOT hitting floor
✓ λ stabilized at ~1.6
```

### Run command:

```bash
python3 train_homeostasis.py --name test --steps 10000
```
