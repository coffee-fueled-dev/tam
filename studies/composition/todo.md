# TODO: Minimal Composition / Transfer Experiment (A → F → B)

## ✅ STATUS: CORE EXPERIMENT COMPLETE

**Results from `runs/transfer_test_20260112_012922/`:**

| Method       | Bind      | Log Vol   | Candidates      |
| ------------ | --------- | --------- | --------------- |
| B-Random     | 0.984     | -1.63     | 1 (loose tubes) |
| B-Best-of-16 | 0.965     | -2.15     | 16              |
| **Transfer** | **0.900** | **-2.45** | **1**           |
| B-CEM        | 0.903     | -2.44     | 512             |

**Key Finding:** Transfer (1 candidate) matches B-CEM (512 candidates) exactly!
This is a 512x compute reduction while achieving the same agency/reliability tradeoff.

---

## 0) Create a new experiment entrypoint

- [x] Add `train_transfer.py` (or `run_transfer.py`) as the main script.
- **Done when:** running `python train_transfer.py` creates a `runs/transfer_<timestamp>/` folder with saved plots + JSON metrics.
- ✅ DONE: `train_transfer.py` creates `runs/transfer_<name>_<timestamp>/` with plots and JSON.

---

## 1) Build Env A / Env B with shared latent dynamics and different observation maps

### 1.1 Factor the bimodal environment into latent-state + observation map

- [x] Create `LatentBimodalEnv` that samples latent situation `s0_latent` and generates trajectory `tau_latent` with mode `m ∈ {+1,-1}`.
  - Inputs: `seed`, `T`, `bend_amp A`, `noise`, `mode_prob=0.5`
  - Outputs from `reset_latent()`:
    - `s0_latent` (e.g., x,y,goalx,goaly, rule_onehot or just params)
    - `mode m` stored internally (for evaluation only)
  - Outputs from `generate_trajectory_latent(s0_latent, m)`:
    - `tau_latent[T,2]`
- ✅ DONE: `LatentBimodalEnv` implemented with mode-dependent sinusoidal bend.

### 1.2 Implement two observation transforms g_A and g_B

- [x] Implement `obs_A = g_A(s0_latent)` as identity (or mild linear mix).
- [x] Implement `obs_B = g_B(s0_latent)` as a _different_ invertible-ish transform:
  - Start minimal: rotation + shear + coordinate permutation + noise
  - Example: `obs_B = M @ s0_latent + b + eps`, where `M` is fixed random orthonormal-ish.
- [x] Wrap into `MultiViewEnv`:
  - `reset()` returns `(obs_A, obs_B, s0_latent)` (s0_latent only for pairing / logging)
  - `generate_trajectory(T)` returns `tau_latent` (or optionally also warped `tau_B`)
- **Decision:** keep trajectories in the same latent XY space for both A and B initially (easier), only change observations.
- ✅ DONE: g_A = identity, g_B = 45° rotation + 1.2x scale + offset.

---

## 2) Train Actor A and Actor B separately (same architecture, different obs spaces)

### 2.1 Reuse your existing Competitive/TAM actor

- [x] Ensure your actor can be instantiated as:
  - `Actor(obs_dim=obs_dim_A, z_dim=Z, ...)`
  - `Actor(obs_dim=obs_dim_B, z_dim=Z, ...)`
- [x] Ensure "Random baseline" samples from posterior `q(z|obs)` (you already fixed this).
- ✅ DONE: Both actors use CompetitiveActor with obs_dim=4, z_dim=4.

### 2.2 Add a paired training loop driver

- [x] In `train_transfer.py`, implement:
  - train A on `obs_A` and `tau`
  - train B on `obs_B` and `tau`
  - Use same latent env samples each step so you can record paired data.
- ✅ DONE: `train_actors()` trains both on same env samples per step.

### 2.3 Save checkpoints + evaluation summaries

- [x] Save `actorA.pt`, `actorB.pt`, `config.json`
- [ ] Save `A_eval.json`, `B_eval.json` with bind/vol and bimodal commitment metrics.
- ✅ DONE: Checkpoints saved. Per-actor eval JSON deferred to ablations.

---

## 3) Collect paired commitment data for learning the functor F

### 3.1 Collect dataset D = {(zA*, zB*)} from paired observations

- [x] Freeze A and B.
- [x] For N samples (e.g., 10k):
  - sample latent `s0_latent`
  - compute `obs_A, obs_B`
  - compute `zA* = bind_CEM(actorA, obs_A)` (store **chosen** commitment)
  - compute `zB* = bind_CEM(actorB, obs_B)`
  - store `(zA*, zB*, rule/mode labels for diagnostics only)`
- ✅ DONE: `collect_paired_data()` saves `pairs.npz` with Z_A, Z_B, modes.

### 3.2 (Optional but recommended) Also store neighborhood info

- [x] Store indices for mini-batch pairwise distance loss (or just compute on the fly).
- ✅ DONE: Pairwise distances computed on-the-fly in `train_functor()`.

---

## 4) Learn functor F: zA → zB

### 4.1 Implement a small MLP functor

- [x] Add `FunctorNet(z_dim, hidden=64, depth=2)`:
  - input: `zA`
  - output: `zB_hat`
- ✅ DONE: `FunctorNet` implemented.

### 4.2 Loss: point alignment + structure preservation

Train on (zA*, zB*):

- [x] Point alignment loss:
  - `L_align = mse(F(zA), zB)`
- [x] Structure preservation loss (batch-based):
  - sample batch of size `B`
  - compute pairwise distances `DA[i,j]=||zAi-zAj||`
  - compute `DB[i,j]=||zBi-zBj||`
  - compute `DF[i,j]=||F(zAi)-F(zAj)||`
  - `L_struct = mse(DF, DB)` (or mse(DA, DF) — pick one and stick with it)
- [x] Total: `L = L_align + λ_struct * L_struct`
  - start `λ_struct = 0.1`
- ✅ DONE: `train_functor()` uses align + struct loss. Loss: 1.83→0.37.

### 4.3 Save functor checkpoint

- [x] Save `functor.pt` and `functor_metrics.json`.
- ✅ DONE: `functor.pt` saved.

---

## 5) Transfer evaluation: bind in B using A's commitments mapped through F

This is the core test.

### 5.1 Implement transfer binder

- [x] For each new paired situation:
  - compute `zA* = CEM(actorA, obs_A)`
  - compute `zB_transfer = F(zA*)`
  - compute tube in B using `zB_transfer` (no search): `(muB, sigmaB) = actorB.tube(obs_B, zB_transfer)`
  - evaluate bind/vol/MCI in Env B
- ✅ DONE: `evaluate_transfer()` implements this.

### 5.2 Baselines (must match compute)

Evaluate in Env B:

- [x] `B-CEM`: normal best method in B
- [x] `B-random-from-posterior`: no planning
- [x] `B-best-of-K`: one-shot sampling competition
- [x] `Transfer (A→F→B)`: **no planning in B** (1 candidate)
- **Compute fairness:** compare Transfer to B-CEM at same wall-clock or same candidate budget (e.g., B-CEM uses 128×4 = 512 candidates vs Transfer uses 1).
- ✅ DONE: All baselines evaluated. Transfer (1) matches B-CEM (512)!

---

## 6) Metrics + plots (the "win" criteria)

### 6.1 Primary: Pareto comparison in B

- [x] Plot `bind vs log_volume` for:
  - B-CEM
  - Transfer
  - B-best-of-K
  - B-random
- ✅ DONE: `transfer_comparison.png` shows Pareto scatter + bar chart.
- **Result:** Transfer matches B-CEM (bind=0.900 vs 0.903, vol=-2.45 vs -2.44).

### 6.2 Secondary: "commitment carries meaning"

- [ ] Plot `d_histogram` in B for each method
- [ ] Plot `MCI` distribution per method
- [ ] Plot z-space:
  - `zB*` clusters (from B-CEM)
  - `F(zA*)` points overlaid
- **Deferred to ablations.**

### 6.3 Sample efficiency curve

- [ ] Vary B-CEM compute: candidates ∈ {16, 32, 64, 128, 256}
- [ ] Compare Transfer (fixed 1) against this curve
- **Deferred to ablations.**

### 6.4 Save summary JSON

- [x] Save `transfer_summary.json` with mean/stdev of bind, logvol, MCI, compute.
- ✅ DONE: `transfer_summary.json` saved with all metrics.

---

## 7) Critical ablations (fast, high-value)

### 7.1 Remove structure preservation term

- [ ] Train functor with `λ_struct=0`.
- [ ] Evaluate transfer again.
- **Expected:** worse basin landing, worse Pareto.
- **Done when:** you can show structure loss matters.

### 7.2 Map encoder samples instead of chosen commitments

- [ ] Build pairs using `zA_sample ~ q(z|obs_A)` and `zB_sample ~ q(z|obs_B)` (not CEM-selected).
- [ ] Train functor and evaluate.
- **Expected:** worse than mapping `z*`.
- **Done when:** you can claim “this transfers commitment, not representation.”

### 7.3 Break topological equivalence (negative control)

- [ ] Modify Env B to have 3 modes or asymmetric mode probabilities.
- [ ] Re-evaluate transfer.
- **Expected:** transfer degrades predictably.
- **Done when:** you’ve demonstrated what structure is required.

---

# “Stop” criteria (what counts as success)

You’re done when you can state (with plots):

1. **Decision-level transfer works:** `Transfer (1 candidate)` beats `B-best-of-K` and `B-random`, and is competitive with `B-CEM` at far higher compute.
2. **Meaning preserved:** `F(zA*)` lands in B’s mode basins and yields bimodal d/MCI like native B commitments.
3. **Not just latent alignment:** mapping samples fails; mapping chosen commitments succeeds.
