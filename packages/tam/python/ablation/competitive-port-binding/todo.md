## Goal of the experiment

Create situations where **two futures are both plausible and equally predictable** (same achievable MSE), but **mutually incompatible** (a tight tube can’t cover both). Then test whether **competitive binding (CEM) chooses one future** (high agency, lower “coverage”) rather than **hedging** (low agency, higher coverage).

Success looks like:

- **CEM selects a low-volume tube** that matches _one_ mode
- the selected tube’s mean aligns with a specific mode
- the selected tube’s volume is much smaller than a “hedge tube” baseline
- selection is **consistent** (stable port-like attractors in z)

---

## 1) Environment design: bimodal but symmetric

You want bimodality without extra clues. Same `s0` should admit two equally likely futures.

### Minimal 2D trajectory generator (drop-in for your current env)

Given `s0=(x,y,goalx,goaly, rule_oh)`:

- Compute baseline straight-line trajectory (as you already do)
- Sample latent mode `m ∈ {+1, -1}` with 50/50
- Add mode-dependent deviation that is _large enough to be incompatible_ with a tight tube:

Example (curvature in opposite directions, same magnitude):

- `traj[t] = baseline[t] + m * bend(t) * n_perp`
- where `bend(t)` peaks mid-horizon and returns to zero at end, and `n_perp` is perpendicular to the baseline direction.

Key property:  
Both modes have identical noise distribution and identical marginal difficulty. Only sign differs.

### Make the modes truly incompatible

Choose bend amplitude `A` such that:

- If tube is “tight” (σ small), it can’t include both modes.
- If tube is “wide” enough to include both modes, volume penalty becomes costly.

Rule of thumb: pick `A` so that at mid-horizon, the two modes are separated by ~`6*k_sigma*σ_min` (so no overlap unless you widen).

---

## 2) Model variants to compare

You want 3 conditions:

### (A) No-competition (posterior sample)

Sample `z ~ q(z|s0)` once → tube.  
This is your “default binding.”

Expected behavior: tends to hedge if training pushes it that way.

### (B) Competitive binding (CEM)

Search in z-space for the best commitment.

Expected behavior: picks a single mode (one branch).

### (C) Hedge baseline (explicit mixture / wide-tube)

A “hedge policy” that is allowed to cover both modes cheaply, so you can show why your agent _isn’t_ doing that.

Two good hedge baselines:

- **Wide-tube baseline:** force σ to be large enough to cover both modes at each t (or add a term rewarding coverage).
- **Mixture-of-tubes baseline:** two heads producing (μ₁,σ₁) and (μ₂,σ₂) but you _don’t_ bind; you accept union. This is the “belief model” baseline.

This contrast is important: it shows “belief can represent both, commitment cannot.”

---

## 3) Training objective: make “both are equally predictive” true

You need to avoid accidentally making one mode easier.

### Key trick: train only on conditional likelihood, not “mode identification”

Do _not_ feed the mode `m` to the model.
Keep `s0` identical across modes.

Use your usual fit loss (`NLL` or MSE) and your volume/failure terms.

But you should add one additional constraint:

### “Mean should match one mode” pressure must be absent during training

If you train with MSE to the _observed_ trajectory, the mean network will “average” the modes (predict the midline), which is exactly the hedging failure mode we want to surface.

So you need a likelihood that makes multimodality not collapse the mean.

Two options:

**Option 1 (simplest, fits your current setup):**
Keep μ as-is, but during evaluation measure _mode alignment_ rather than μ-MSE. You accept that μ will be midline under passive binding, and show CEM “escapes” midline by selecting a z that shifts μ toward one mode.

This requires μ actually depends on z enough to shift.

**Option 2 (more correct):**
Make μ_net output a _residual_ added to baseline, and let z control that residual strongly. Then competitive binding can pick residual sign.

I’d recommend Option 2 if you can do it quickly:

- baseline is deterministic straight-line
- μ predicts deviation field
- z picks which deviation branch

That makes the “two futures equally predictable” claim _structural_.

---

## 4) The binding score for CEM (what it should optimize)

You want a score that trades:

- **Agency:** low volume
- **Reliability / risk:** avoid likely failure
- **Intent:** choose a coherent mode, not midline

Here’s a clean score:

\[
J(z) = -\alpha \log V(\sigma(z)) \;-\; \lambda \,\hat{p}_\text{fail}(s0,z)\;-\;\gamma\,H_\text{mode}(s0,z)
\]

Where:

- `log V` = mean log σ (what you already track)
- `p_fail` = risk critic prediction (what you already have)
- `H_mode` = _mode entropy proxy_ — encourages committing to one branch

How to define `H_mode` without access to true mode?
Use a geometric proxy:

### Mode entropy proxy (very practical)

Pick a diagnostic time `t* = T//2` where modes are maximally separated.

Let `d(z) = signed_perp_deviation(μ[t*])` relative to baseline direction.

Then:

- midline has `d≈0` (hedge)
- committed modes have `|d| large`

So define:
\[
H\_\text{mode} = \exp(-|d(z)|/s)
\]
This penalizes staying near midline.

Important: this does _not_ leak the true mode. It only rewards “choosing a side.”

If you don’t want extra terms, you can also rely purely on volume + risk and measure mode commitment in evaluation. But adding a weak mode-commit term makes the phenomenon easier to elicit.

---

## 5) Evaluation metrics (make the claim falsifiable)

You need metrics that separate:

- “good prediction”
- “good coverage”
- “commitment”

### Core metrics (per episode)

1. **Volume:** `log_vol`
2. **Bind rate / fail:** as you already compute
3. **Mode commitment index (MCI):**

   - compute mid-horizon signed deviation `d`
   - define `MCI = |d| / A` (normalized 0..1)
   - hedge → ~0, commit → ~1

4. **Which mode was chosen (classification):**
   Since you know `m` in the generator, check sign agreement:
   - `mode_match = sign(d) == m` (when |d| above threshold)

This gives a crisp behavioral statement:

- CEM should have high MCI and high mode_match rate.
- Random/posterior sample should have low MCI (midline).
- Hedge baseline has low MCI but high bind.

### Aggregate plots

- Scatter: (bind vs log_vol) colored by MCI
- Histogram of d for each selection method (should be bimodal under CEM)
- z-space clustering: z\* colored by sign(d) (ports emerge as two attractors)

---

## 6) Expected outcomes (what would “win” look like?)

**If TAM commitment works:**

- CEM: two clusters in z\*, high |d|, low volume, bind acceptable
- Random: single blob in z\*, d near 0, higher bind but higher volume (or low agency)
- Hedge baseline: d near 0, highest bind, worst agency (largest volume)

**If it fails (useful failure modes):**

- μ doesn’t move with z → CEM can’t choose a mode (fix: strengthen μ dependence on z)
- risk critic dominates → always chooses wide safe tubes (fix: raise α or reduce λ)
- training collapses to always predict one mode → environment asymmetry (fix: enforce symmetry, shuffle sign)

---

## 7) Implementation steps (ordered so you can iterate fast)

### Step 0: Add bimodal rule in env

- Add `mode ∈ {+1,-1}` sampled after reset
- Generate two trajectories for same s0 by flipping mode

### Step 1: Add diagnostic `d(z)` function

- Compute baseline direction
- Compute signed perpendicular displacement at t\*

### Step 2: Verify “incompatible futures”

Plot the two modes for the same s0 and ensure separation at t\*.

### Step 3: Make sure μ depends on z enough

Before any training changes:

- sample multiple z values
- plot μ trajectories
- confirm d(z) sweeps both signs

If not: change μ parametrization so z can flip deviation sign.

### Step 4: Train tube + risk critic as usual

Don’t change selection yet. Just get stable tubes + usable risk.

### Step 5: Turn on competitive binding at evaluation only

Run CEM at inference-time and measure MCI + mode_match.

### Step 6: Only if needed, add weak “choose a side” term

Add `-γ * exp(-|d|/s)` to CEM score (evaluation-time first, then training-time if you like).

---

## 8) What to write down as the experimental claim

If the results match expectation, your claim becomes:

> In a bimodal world where two incompatible futures are equally predictable from the same situation, competitive binding selects a low-volume commitment aligned with one mode, rather than expanding its acceptance set to cover both. This separates _belief_ (representing both) from _commitment_ (excluding one).

That’s the TAM “missing layer” sentence.
