## Patch A — Add knobs (disable + gating modes)

### A1) Add config flags + hyperparams (actor.py `__init__`)

Add:

- `reasoning_mode: str = "fixed"` # `"off" | "fixed" | "gated" | "dynamic"`
- `freeze_sigma_refine: bool = True`
- `c_improve: float = 1.0` # exchange rate: NLL improvement “pays for” Hr
- `improve_detach: bool = True` # prevent hacking improve via baseline coupling
- `gate_kind: str = "nll0"` # `"nll0" | "mem_risk" | "volatility" | "combo"`
- `gate_thresh: float = ...` # scalar threshold
- `gate_kappa: float = ...` # softness if using sigmoid gating
- `Hr_default: int = 0`
- `Hr_max: int = self.max_refine_steps`

Also add history keys:

- `"NLL0"`, `"NLLr"`, `"improve"`, `"ponder_reason_raw"`, `"ponder_reason_eff"`, `"gate_score"`, `"gate_on"`

**Done when:** you can run with `reasoning_mode="off"` and code never calls `infer_tube(...)` refinement (only `_tube_init` / `_tube_traj`).

---

## Patch B — Compute NLL0 vs NLLr (progress metric) correctly

### B1) Implement a helper that evaluates expected NLL under a given tube (actor.py)

Add a utility method:

- `_expected_nll(mu_knots, logsig_knots, stop_logit, y, T, detach_w: bool=True) -> (exp_nll, w)`

Use your existing:

- `_tube_traj(...)`
- `gaussian_nll(...)`
- `truncated_geometric_weights(...)`

Compute:

- `p_stop = sigmoid(stop_logit)`
- `w, _ = truncated_geometric_weights(p_stop, T)`
- `nll_t = gaussian_nll(mu, log_var, y).mean(-1)`
- `exp_nll = (w * nll_t).sum()`

**Done when:** you can call it for both Hr=0 and Hr>0 and get two scalars.

### B2) In `train_on_episode`, compute baseline and refined expected NLL

You already compute something similar for delta_nll; make it canonical:

- Baseline tube: `(mu0_knots, logsig0_knots, stop0_logit) = _tube_init(...)`
- `NLL0 = expected_nll(baseline)`
- Refined tube: depending on reasoning_mode, produce `(mur_knots, logsigr_knots, stopr_logit)`
- `NLLr = expected_nll(refined)`

Then:

- `improve = relu(NLL0_detached - NLLr)` (or raw `(NLL0 - NLLr)`; I’d start with relu)
- If `improve_detach`: `NLL0_detached = NLL0.detach()`

Log `NLL0, NLLr, improve`.

**Done when:** “improve” is positive exactly when refinement _actually reduces expected NLL_.

---

## Patch C — Fix the compute incentive (pay-for-progress ponder)

### C1) Replace reasoning ponder cost term

Right now you do:

- `ponder_reason = lambda_r * E_Hr_train`

Replace with:

- `ponder_reason_raw = self.lambda_r * E_Hr_train`
- `ponder_reason_eff = self.lambda_r * relu(E_Hr_train - self.c_improve * improve.detach_if_config)`

Notes:

- Detach `improve` here (always) so the model can’t increase improve by making baseline worse or by weird gradients.
- Keep logging both `ponder_reason_raw` and `ponder_reason_eff`.

Then set:

- `ponder = ponder_commit + ponder_reason_eff`

**Done when:** if refinement doesn’t improve NLL, extra Hr is strictly penalized; if it improves enough, penalty goes to ~0.

### C2) Update λ_r dual update to match new objective

You can keep the same controller update (`E_Hr_train - target_Hr`) **or** switch to an “unpaid compute” target:

- Option 1 (minimal change): keep as-is.
- Option 2 (better): drive λ_r using **effective unpaid compute**:
  - `unpaid = relu(E_Hr_train - c_improve * improve.detach())`
  - `lambda_r += lr * (unpaid - target_unpaid)` where `target_unpaid ~ small` (like 0.5)

Start with Option 1 for stability.

**Done when:** λ_r doesn’t blow up just because the agent uses Hr in episodes where it’s actually improving fit.

---

## Patch D — Prevent σ-gaming during refinement

### D1) Freeze σ during refinement (recommended diagnostic)

Modify `infer_tube(...)` loop:

- still compute `delta_mu, delta_logsig, delta_stop = refiner(...)`
- but if `freeze_sigma_refine`:
  - `delta_logsig = 0` (or skip applying it)

So:

- `logsig_knots` stays from `_tube_init` (or still clamped once)

**Done when:** performance difference between Hr=0 and Hr>0 can’t be explained by “σ got smaller”.

### D2) (Optional later) softer alternative: penalize Δlogσ without improvement

If you later want σ refinement back, add:

- `sigma_shrink = relu(-(logsig_r - logsig_0)).mean()` # only counts shrink
- `sigma_game_pen = w_sigma_game * relu(sigma_shrink - k * improve.detach())`

But don’t do this until you’ve run the freeze experiment.

---

## Patch E — Add a gating policy for when to think

### E1) Implement `gate_score(...)` in Actor

Inputs you already have per-episode:

- `volatility` from env info (you compute it in eval; add to train loop info too)
- `mem_risk(z)` (already exists)
- `NLL0` (computed in Patch B)

Implement something like:

- `score =`
  - `"nll0"`: `NLL0.detach()`
  - `"mem_risk"`: `mem_risk(z).detach()`
  - `"volatility"`: `torch.tensor(volatility)`
  - `"combo"`: weighted sum of normalized versions

Then decide gate:

- hard gate: `gate_on = score > gate_thresh`
- soft gate: `gate_on_prob = sigmoid((score - gate_thresh)/gate_kappa)`

Log `gate_score` and `gate_on`.

**Done when:** you can see a histogram of gate_score and verify it correlates with “hard episodes”.

### E2) Choose Hr using the gate

In `train_on_episode` (or earlier where you sample Hr):

If `reasoning_mode == "off"`:

- `Hr = 0`

If `"fixed"`:

- keep your existing sampling (or fixed Hr)

If `"gated"`:

- `Hr = Hr_default` when gate off
- `Hr = sample_reasoning_steps(...)` or `Hr_max` when gate on
  - simplest: `Hr = Hr_max` when on (you can anneal later)

If `"dynamic"`:

- your existing `infer_tube_dynamic(...)` path

**Done when:** average Hr drops a lot, but you still see high Hr on high gate_score episodes.

---

## Patch F — Update experiments + eval to compare modes

### F1) Add sweep in experiments.py

Run at least:

- `reasoning_mode="off"`
- `reasoning_mode="fixed"` (current baseline)
- `reasoning_mode="gated"` with `Hr_default=0`, `Hr_max=max_refine_steps`

Keep evaluation exactly as you already do (Hr=1 vs Hr=max is still useful), but add “mode” label in run_dir.

**Done when:** you can line up dashboards across runs and answer:

- Does gated reasoning improve J_mean on fault-heavy episodes?
- Does it improve calibration (z-score hist moves closer to N(0,1)) _without_ just shrinking σ?

### F2) Add 2 tiny new plots to the dashboard

1. `improve` vs `E_Hr_train` scatter (or binned):
   - proves “compute is being spent where it buys fit”
2. `gate_score` over training (and fraction gated-on)

**Done when:** you can visually confirm the incentive fix is working (high Hr coincides with high improve).

---

## Minimal implementation order (fastest path)

1. **A (knob)** + **B (NLL0/NLLr/improve)**
2. **C (pay-for-progress ponder)**
3. **D (freeze σ)**
4. **E (gating)**
5. **F (mode comparisons + 2 extra plots)**

---

## Quick “success criteria” checklist

- With `reasoning_mode="off"`, you get a stable baseline and no regressions.
- With `freeze_sigma_refine=True`, any Hr benefit must show up as **lower |err| and better J**, not just lower σ.
- `improve` is near 0 on easy episodes; positive on fault/switch episodes.
- In gated mode: average Hr stays low, but “hard episode” performance improves.
