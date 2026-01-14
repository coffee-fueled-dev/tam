Your Phase 0 diagnostics are internally consistent and they basically say one thing:

**the environment currently makes “hedging” (one tube covering multiple futures) cheap, so there’s no selection pressure for ports to emerge, and therefore CEM has nothing to exploit.** [oai_citation:0‡phase0_diagnostics.json](sediment://file_00000000d38871fdbfce9807255eab1d) [oai_citation:1‡summary.json](sediment://file_00000000504c71fd8162c65781a72213)

### What the numbers mean (and why you see “no basins”)

From `phase0_diagnostics.json`:

- **Actual tube fails ~32%**, but **2× wider tube fails ~1.3%** [oai_citation:2‡phase0_diagnostics.json](sediment://file_00000000d38871fdbfce9807255eab1d)  
  → widening σ is an _extremely effective_ hack: you buy near-perfect reliability for a small volume increase.

- **CEM score gap ~0.0024** and **score range ~0.0055** across 64 z samples [oai_citation:3‡phase0_diagnostics.json](sediment://file_00000000d38871fdbfce9807255eab1d)  
  → almost all candidate z’s look equally good under the current scoring/objective, so CEM can’t “choose” anything meaningful.

- Your run summary shows **z→mode accuracy ~0.515 (chance)** and **silhouette ~0** [oai_citation:4‡summary.json](sediment://file_00000000504c71fd8162c65781a72213)  
  → z isn’t encoding mode, and the latent isn’t clustered.

So yes: you’re reading the plots right. “Ports don’t emerge” here because **there is no forced commitment point**—the agent can do fine by inflating σ and staying ambiguous.

---

## Phase 1 goal (what you should change)

Phase 1 is: **make ambiguity expensive in a way that can’t be patched by “σ everywhere.”**

That means you need _at least one_ of these to be true:

1. **Worst-case matters**: one violation anywhere is costly (not average bind).
2. **Volume scales with dimension/time**: widening “a bit everywhere” is very expensive.
3. **Modes are mutually exclusive in a local window**: there is no tube that can cover both after gating without paying a huge σ cost.

Right now you have the opposite: average bind + mean log σ + easy widen → hedge wins.

---

## Concrete TODO list (Phase 1)

### A) Make hedging actually cost something (objective)

1. **Switch volume penalty from mean(log σ) to sum(log σ)**

- You already added `use_global_vol=True`. Make sure it’s literally:
  - `log_vol = torch.log(sigma).sum()` (or sum over (t,d))
  - not mean.
- This makes “inflate everywhere” scale with **T×d**, which is what you want.

2. **Change bind/fail to “max-timestep” rather than mean-timestep**

- Your diagnostics currently treat bind as an average over time (and hard inside-all-dims at each t, then mean over t).
- Replace the training fail proxy with a soft-max over time, e.g.:
  - compute per-timestep violation margin
  - `fail_soft = sigmoid(max_fail_gamma * max_t(violation_t))`  
    or use a smooth approximation: `softmax`/`logsumexp` across t.

3. **Lower k_sigma**

- If `k_sigma` is too generous, widening σ buys too much safety.
- Reduce from e.g. 1.5 → 1.0 (or even 0.8) in the _bind definition used for loss_.

4. **Add σ-floor + σ-smoothness only after you’ve made volume global**

- Smoothness helps avoid the “point-tube” pathology, but don’t use it to stabilize hedging.
- Apply it as: penalize `||Δ_t log σ||^2` so tubes remain tubes, not blobs.

---

### B) Make the environment have a real “commitment bottleneck”

5. **Strengthen early divergence _and_ enforce irreversibility**
   Right now you’ve increased early divergence (good), but your trajectories still look like they can “swing back” and overlap later.

Add one of these minimal irreversibility mechanisms:

- **Mode-specific one-way drift during gating**: during `t < t_gate`, dynamics include a strong mode-dependent affine push that cannot be undone later.
- **Latch variable**: once the system has moved past a boundary (hyperplane), dynamics change so returning is hard/impossible.
- **Local obstacle / forbidden slab** (very effective): if mode0 goes through region A and mode1 through region B, then a single tube covering both must inflate massively to include both corridors.

6. **Make success _not always 1.0_**
   Your `final_cem_success` and `final_random_success` are both 1.0 [oai_citation:5‡summary.json](sediment://file_00000000504c71fd8162c65781a72213), which means your “success” metric is currently too easy.

- Tighten `success_threshold` substantially (and/or scale it _down_ with √d).
- Or define success as: **end within ε AND bind ≥ τ** (so you can’t “succeed” while ignoring the tube).

---

### C) Give CEM a lever (selection signal)

7. **Ensure `score_commitment` is aligned with what you want**
   If you want “choose a mode,” your score must punish hedging.

A safe Phase 1 score is:

- `score = -NLL(trajectory | μ,σ) - α * global_log_vol - β * predicted_fail`

But in your “pure TAM” training, you aren’t using CEM at all during training, so **the encoder may collapse to a unimodal posterior** and all z samples become equivalent.

So:

8. **Train with “competitive sampling” (minimal)**
   Without adding labels:

- sample K candidate z’s each step
- pick the best one **under the same score CEM uses**
- do the gradient step on that z (“winner-take-gradient”)

This is the smallest change that creates pressure for **z to become a decision variable** rather than noise.

---

## What “done” looks like for Phase 1

Re-run Phase 0 diagnostics after these changes, and you want:

- **Hedge no longer cheap**: 2× σ should _not_ drop fail to ~0.01. It should stay meaningfully high (or volume should explode).
- **CEM opportunity increases**: score_range should jump from ~0.005 → something like ~0.1+ (order-of-magnitude).
- **CEM advantage appears**: `cem_success > random_success` and/or `cem_log_vol < random_log_vol` at same success.
- **z-space metrics improve**: silhouette > 0, z→mode accuracy > chance.
