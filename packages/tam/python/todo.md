Below is a **concrete, implementation-ready TODO list**, ordered so that each block produces interpretable signal before you invest in the next one. Nothing here requires architectural changes beyond what you already have — this is about **making cross-environment reuse measurable and real**.

---

# Cross-Environment Commitment Reuse — Concrete TODO List

## PHASE 0 — Hygiene (do this first)

Goal: ensure transfer results are statistically meaningful and comparable.

- [ ] **Multi-seed evaluation**

  - Run each `(source_env, target_env, reuse_mode)` with **N ≥ 10 seeds**
  - Store per-seed outcomes, not just means

- [ ] **Confidence intervals**

  - Report mean ± std **and** bootstrap 95% CI for:
    - outcome (−J or success)
    - coverage @ k\*
    - bind success
  - Add error bars to reuse-gain plots

- [ ] **Paired comparison**

  - For each seed, compute:
    ```
    Δ_memory = outcome(memory) − outcome(native)
    Δ_proto  = outcome(prototype) − outcome(native)
    ```
  - Plot paired scatter or bar with CI
  - This removes environment variance from the comparison

- [ ] **Evaluation parity**
  - Ensure eval uses identical settings across reuse modes:
    - same horizon sampling
    - same action noise (ideally 0)
    - same `Hr_eval ∈ {0, 1, max}` (separate runs)

---

## PHASE 1 — Diagnose z portability (before “fixing” transfer)

Goal: verify whether **z has shared semantics across envs at all**.

- [ ] **Cone-summary portability test**

  - Sample a fixed set of z’s from source memory (e.g. 50)
  - For each z:
    - Compute `(C_source(z), H_source(z))`
    - Compute `(C_target(z), H_target(z))` using target tube
  - Plot:
    - `C_source vs C_target`
    - `H_source vs H_target`
  - If correlation ≈ 0 → z is not portable → reuse via distance is meaningless

- [ ] **Order preservation test**
  - Rank z’s by sharpness / horizon in source
  - Measure rank correlation in target
  - This is the minimal requirement for reuse

---

## PHASE 2 — Replace “nearest-z” with real reuse

Goal: test **concept reuse**, not coordinate coincidence.

### 2A. Behavioral retrieval (recommended first)

This is the **simplest meaningful reuse**.

- [ ] From source env, store memory tuples:

  ```
  (z, mean_outcome, bind_rate, cone_volume, E[T])
  ```

- [ ] In target env:

  - Sample K candidate z’s from source memory (e.g. K=10)
  - For each candidate z:
    - Evaluate _cheap proxy_ in target:
      - predicted `NLL0`
      - predicted cone volume
      - predicted p_stop / E[T]
  - Select best z by:
    ```
    score = −NLL0 − α*cone_vol + β*E[T]
    ```

- [ ] Compare against:

  - native z
  - nearest-neighbor z
  - random z from source memory

- [ ] Plot:
  - outcome vs retrieval score
  - fraction of episodes where retrieval beats native

---

## PHASE 3 — Prototype reuse done correctly

Goal: test whether **basins** (not points) transfer.

- [ ] Extract prototypes via **behavioral clustering**, not KMeans on z:

  - cluster by `(cone_volume, E[T], bind_rate)` or outcome
  - choose representative z per basin

- [ ] During transfer:

  - Evaluate each prototype z with proxy score (as above)
  - Select best prototype _per episode_

- [ ] Compare:

  - best-of-K prototype
  - nearest-prototype
  - native z

- [ ] Add ablation:
  - number of prototypes K ∈ {1, 4, 8}

---

## PHASE 4 — Add minimal adaptation (only if reuse shows promise)

Goal: make reuse plausible when raw z is misaligned.

### 4A. Learned z-adapter (small, contained)

- [ ] Train adapter:
  ```
  z_target = g(z_source)
  ```
- [ ] Objective:
  - match **cone summaries** `(C, H)` between envs
  - NOT raw z distance
- [ ] Freeze main actor/tube
- [ ] Evaluate:
  - memory + adapter vs memory alone

### 4B. Environment embedding (optional, stronger)

- [ ] Add env embedding `e`
  - actor: `q(z | s0, e)`
  - tube: `tube(s0, z, e)`
- [ ] Train on mixed envs
- [ ] Test zero-shot on held-out env seeds
- [ ] Measure whether same z regions are reused across e

---

## PHASE 5 — Canonical transfer plots (keep these, drop the rest)

These are the **only plots you should rely on** for transfer.

- [ ] **Transfer gain (paired)**

  - y = outcome(reuse) − outcome(native)
  - x = seed
  - CI bands

- [ ] **Cone-summary portability**

  - `(C_source, H_source)` vs `(C_target, H_target)`

- [ ] **Retrieval effectiveness**

  - histogram of proxy scores
  - win-rate vs native

- [ ] **Reuse breakdown**
  - stacked bar: fraction of episodes where:
    - native wins
    - reuse wins
    - tie

---

## PHASE 6 — Sanity checks (non-negotiable)

- [ ] Shuffled memory control (should destroy reuse benefit)
- [ ] Random z baseline
- [ ] Freeze vs unfreeze actor comparison
- [ ] Disable reasoning vs enable reasoning (transfer often depends on it)

---

## Summary Decision Gates

Only proceed if:

- **Phase 1** shows partial portability of `(C,H)`
- **Phase 2** beats native in ≥1 env with CI
- **Phase 3** outperforms nearest-z

If none of these hold, the conclusion is **not that TAM failed**, but that:

> z currently encodes _environment-specific_ commitments, not transferable concepts — which tells you exactly what to fix next.
