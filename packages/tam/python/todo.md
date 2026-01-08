## Patch 0 — Add run directory + saving utilities (experiments.py)

### 0.1 Imports
Add at top:
- `import os, json, time`
- `from pathlib import Path`
- (optional) `import hashlib` if you want deterministic run IDs

### 0.2 Run directory helper
Add:

- `def make_run_dir(base="runs", tag="tam") -> Path:`
  - timestamp like `YYYYMMDD_HHMMSS`
  - create `runs/{tag}_{timestamp}/`
  - inside create subfolders:
    - `fig/` (pngs)
    - `data/` (npz/json)

### 0.3 Figure saving helper
Add:

- `def save_fig(fig, path: Path, dpi=150):`
  - `fig.savefig(path, dpi=dpi, bbox_inches="tight")`
  - `plt.close(fig)` to prevent leaks

### 0.4 Config dump
At run start, write:
- `config.json` containing:
  - seed/train_steps/eval_every/eval_episodes/maxH
  - env params
  - actor params you passed (you can manually build this dict in `main()`)

---

## Patch 1 — Define a performance metric: episode control cost J (experiments.py)

You need a control objective for evaluation that is **independent of tube self-prediction**.

### 1.1 Add cost function
Add:

- `def episode_control_cost(states, actions, dt=0.05, w_theta=1.0, w_omega=0.1, w_act=0.01):`
  - states shape `[T+1,2]`, actions `[T,1]`
  - `theta = states[1:,0]`, `omega = states[1:,1]`, `a = actions[:,0]`
  - cost per step: `w_theta*theta**2 + w_omega*omega**2 + w_act*a**2`
  - return:
    - `J_sum = cost.sum()` (or `.mean()`; pick one and stick to it)
    - `J_mean = cost.mean()`
    - optionally return `cost_t` for debugging

**Decision:** prefer `J_mean` so horizon variation doesn’t trivially dominate.

### 1.2 Extend EvalSnapshot to store J
Modify `EvalSnapshot` to include:
- `mean_J: float`
- optionally `J_points: np.ndarray` (per-episode) if you want scatter later

### 1.3 Compute J in evaluate_agent
Inside evaluation loop (after rollout):
- compute `J_mean = episode_control_cost(state_seq, actions, dt=env.dt)["J_mean"]`
- accumulate `J_sum += J_mean`
- store per-episode J into pareto row if you want (optional)

Add to snapshot:
- `mean_J = J_sum / n_episodes`

---

## Patch 2 — Add dashboard data extraction (evaluate_agent)

The dashboard needs two more evaluation-time aggregates:

### 2.1 Add “σ vs |err|” aggregates
During evaluation for each episode, after you compute `mu, log_var`:

Compute weighted summaries (using `w`):
- `sigma = exp(0.5*log_var)` → `[T,2]`
- `err = abs(y - mu)` → `[T,2]`
- weighted mean per dim:
  - `sigma_w = (w[:,None]*sigma).sum(0)`
  - `err_w   = (w[:,None]*err).sum(0)`
- accumulate sums across episodes:
  - `sigma_w_sum += sigma_w.cpu().numpy()`
  - `err_w_sum += err_w.cpu().numpy()`

At end:
- `mean_sigma_w = sigma_w_sum / n_episodes` (2-vector)
- `mean_err_w = err_w_sum / n_episodes`

Store into snapshot as:
- `mean_sigma_w: np.ndarray` (shape (2,))
- `mean_err_w: np.ndarray`

### 2.2 Collect z-scores for histogram
Also compute standardized residuals:
- `r = (y - mu) / (sigma + 1e-8)` → `[T,2]`
- flatten and append to list (cap size so memory doesn’t explode):
  - e.g. keep up to `max_r = 20000` samples per snapshot
  - randomly subsample `r.flatten()` if too large

Store into snapshot:
- `z_scores: np.ndarray` (1D array, bounded size)

---

## Patch 3 — Implement the 3-plot dashboard function (experiments.py)

Add:

`def plot_performance_dashboard(snapshots_lo, snapshots_hi, run_dir: Path, prefix="perf_dashboard")`

It will generate **three figures**, each comparing Hr=1 vs Hr=max across snapshots.

### 3.1 Plot A: J vs training step (performance curve)
- x: `snap.step`
- y_lo: `snap.mean_J` from `snapshots_lo`
- y_hi: `snap.mean_J` from `snapshots_hi`
- include legend: “Hr=1”, “Hr=max”
- title: `Control performance (lower is better): J_mean`

Save:
- `fig/perf_J_curve.png`

### 3.2 Plot B: weighted σ vs weighted |err| per dimension
For the **last snapshot** of each setting (or plot trajectories over snapshots if you prefer):

Option 1 (simple + informative):
- bar or line plot with two panels:
  - panel 1: theta: `mean_sigma_w[0]` vs `mean_err_w[0]`
  - panel 2: omega: `mean_sigma_w[1]` vs `mean_err_w[1]`
- do this for Hr=1 and Hr=max (two colors or two line styles)

Save:
- `fig/perf_sigma_vs_error_last.png`

Option 2 (over snapshots):
- plot `mean_sigma_w_dim` and `mean_err_w_dim` vs step, two lines each (σ and |err|), for theta and omega in separate figs.

(Plan recommendation: implement Option 1 first.)

### 3.3 Plot C: z-score histogram at 3 checkpoints
Pick checkpoints:
- first snapshot, middle snapshot, last snapshot
Do for Hr=1 and Hr=max separately (either two subplots or overlay).

Recommended:
- 2 rows × 3 cols:
  - row 1: Hr=1 histograms at [early, mid, late]
  - row 2: Hr=max histograms at [early, mid, late]
- fixed xlim like `[-5, 5]`

Save:
- `fig/perf_zscore_hists.png`

---

## Patch 4 — Save ALL figures + data to run directory (main)

### 4.1 Change plot functions to return fig objects
Right now your plot functions call `plt.show()` and discard figs.

Update each plotting function signature to accept:
- `save_path: Optional[Path] = None`
- `show: bool = True`

Inside:
- create `fig = plt.figure(...)` or `fig, ax = plt.subplots(...)`
- if `save_path`: `save_fig(fig, save_path)`
- if `show`: `plt.show()` else `plt.close(fig)`

Apply to:
- `plot_training_overview`
- `plot_dual_phase_portrait`
- `plot_eval_snapshots`
- `plot_z_memory_map`
- new `plot_performance_dashboard`

### 4.2 Save evaluation + history data
At end of `main()`:
- save `agent.history` → `data/history.npz` or `history.json`
  - npz is easiest: convert lists to numpy arrays
- save eval snapshots:
  - `data/eval_hr1.npz`
  - `data/eval_hrmax.npz`
  - include: step, mean_J, mean_sharp_log_vol, mean_volatility, etc.
- save memory (if present):
  - `data/memory.npz` with `zs, soft, cone, lam, risk`

### 4.3 Main flow
In `main()`:
1) `run_dir = make_run_dir(tag="tam_hidden_fault")`
2) train
3) generate + save:
   - `training_overview.png`
   - `dual_phase.png`
   - `eval_snapshots.png`
   - `z_memory.png`
   - **new**: `perf_J_curve.png`, `perf_sigma_vs_error_last.png`, `perf_zscore_hists.png`
4) dump `config.json`, `history.npz`, `eval_hr*.npz`

---

## Patch 5 — Small consistency fix: always evaluate using refinement for vis
You already updated `tube_predictions_for_episode` to use refinement. Make sure:
- `evaluate_agent()` calls `episode_metrics(..., Hr_eval=...)`
- `episode_metrics()` calls `tube_predictions_for_episode(..., Hr_eval=...)`
That ensures the σ/err and z-score plots reflect the same semantics as your Pareto + calibration plots.

---

## Suggested file outputs (runs/<run>/fig)
- `training_overview.png`
- `dual_phase_portrait.png`
- `eval_calibration_pareto_horizon.png` (your current eval plots)
- `z_memory_map.png`
- `perf_J_curve.png`
- `perf_sigma_vs_error_last.png`
- `perf_zscore_hists.png`

And (runs/<run>/data)
- `history.npz`
- `eval_hr1.npz`
- `eval_hrmax.npz`
- `memory.npz` (if present)
- `config.json`
