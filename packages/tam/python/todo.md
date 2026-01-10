## TODO: Factor `z` into `z_intent` vs `z_real`, and update functor learning

### A. Define the target interfaces (do this first)

- [x] Decide latent split sizes:
  - `z_intent_dim` (start with 2–4)
  - `z_real_dim = z_dim - z_intent_dim`
- [x] Standardize naming in code/config:
  - `z_intent`, `z_real`, `z = cat([z_intent, z_real])`
  - `F_intent: Z_intent^A -> Z_intent^B`
- [x] Add config flags:
  - `use_factored_z: bool`
  - `intent_only_tube: bool` (default True)
  - `functor_maps: {"intent": True, "real": False}`

**Implementation:** See `factored_actor.py` and `networks.py` (`FactoredActorNet`, `FactoredSharedTube`)

---

### B. Actor changes: sample and carry two latents

- [x] Update `Actor.sample_z(s0)` → returns `(z, z_mu, z_logstd, z_intent, z_real)` (or a small dataclass).
- [x] Implement two posterior heads:
  - `q_intent(z_intent | s0)` (mean/logstd)
  - `q_real(z_real | s0)` (mean/logstd)
- [x] Keep KL bookkeeping separate:
  - `KL_intent`, `KL_real`, `KL_total = KL_intent + KL_real`
- [x] Update memory records:
  - store both `z_intent` and `z_real` (and `z_full` for backward compat)
  - update any replay/memory utilities accordingly

**Implementation:**

- `FactoredActor.sample_z_factored()` returns `FactoredZSample` dataclass
- `FactoredActorNet` has separate heads for intent and real
- `compute_kl_factored()` returns separate KL values
- Memory stores `z`, `z_intent`, `z_real`, `kl_intent`, `kl_real`

**Sanity checks**

- [x] Assert shapes everywhere: `z_intent.shape[-1] == z_intent_dim`, etc.
- [x] Log distributions for both latents (mean/std histograms) during training.

**Implementation:** History tracks `z_intent_norm`, `z_real_norm`, `kl_intent`, `kl_real`

---

### C. Tube network: make "commitment geometry" depend on `z_intent`

Goal: cone algebra lives in `z_intent` and becomes transportable.

- [x] Split tube inputs:
  - primary: `(s0, z_intent)`
  - optional/weak: `(z_real)` (start by **excluding** it from σ and stop)
- [x] Wire heads intentionally:
  - `μ-head`: can take `(s0, z_intent, z_real)` (optional)
  - `logσ-head`: **only** `(s0, z_intent)` (recommended)
  - `stop_logit-head`: **only** `(s0, z_intent)` (recommended)
- [x] Add a guard option: `freeze_sigma_during_refine` still works with factored z.

**Implementation:** `FactoredSharedTube` in `networks.py`:

- `mu_head`: uses full z (s0, z_intent, z_real)
- `sigma_head` and `stop_head`: use only (s0, z_intent)
- Config flags: `intent_only_sigma`, `intent_only_stop`

**Diagnostics**

- [ ] Train with `z_real` ablated from tube entirely; confirm performance doesn't collapse.
- [ ] Verify cone stats (log cone vol, E[T]) vary smoothly with `z_intent`.

---

### D. Policy network: allow environment-specific realization in `z_real`

- [x] Feed policy `(obs, z_intent, z_real)` (default).
- [ ] Add an ablation mode: policy uses `(obs, z_intent)` only.
- [x] Ensure discrete/continuous action heads unchanged except input dim.

**Implementation:** Policy uses full `z = cat([z_intent, z_real])` - preserves existing interface

---

### E. Redefine the "meaningful" regularizers and reporting

You want incentives to sculpt `z_intent`, not `z_real`.

- [x] Apply commitment-related objectives to `z_intent`-conditioned quantities:
  - bind success target
  - calibration/coverage losses
  - cone sharpness (log cone vol)
  - horizon targets `E[T]`
- [x] Keep `z_real` "free" except its KL (or a small L2 prior).
- [x] Update dashboards:
  - show z-space plots for `z_intent` separately from `z_real`
  - compute probe/classification on `z_intent` first (rules, regimes, etc.)

**Implementation:**

- `beta_kl_intent` and `beta_kl_real` are separate (real has weaker KL)
- Cone geometry (sigma, stop) depends only on z_intent → commitment objectives shape z_intent
- History logs separate `kl_intent`, `kl_real`, `z_intent_norm`, `z_real_norm`

---

### F. Update reuse modes to be intent-aware

Current reuse modes replace the full `z`. Change to reuse **intent** only.

- [x] `MemoryReuseActor`:
  - sample native `(z_intent_B, z_real_B)`
  - replace **only** `z_intent_B` with nearest `z_intent` from source memory (or with functor output)
  - keep `z_real_B` native
- [x] `PrototypeReuseActor`:
  - prototypes live in `z_intent` only
  - choose prototype by nearest-to-native `z_intent_B` or a learned selector
- [ ] Add baselines:
  - "intent-shuffled" (shuffle intent across episodes, keep real native)
  - "real-shuffled" (shuffle real, keep intent native) — should hurt control more than calibration

**Implementation:** `FactoredActor.set_intent()` method:

- Takes external `z_intent` (from functor or source memory)
- Samples native `z_real` from local posterior
- Combines: `z = cat([z_intent_transported, z_real_native])`

---

### G. Functor learning: map `z_intent^A -> z_intent^B` (not full z)

Replace your existing `functor.py` objective with intent-only transport.

**Data collection**

- [x] Build paired dataset of "comparable episodes":
  - For equivalent env pairs (standard↔mirrored, standard↔rotated):
    - Match initial states via known transform (mirror/rotate)
    - Use the _same_ underlying rule seed when possible
- [x] For each paired episode, log:
  - `s0_A, z_intent_A, cone_stats_A`
  - `s0_B, z_intent_B_native, cone_stats_B_native`
  - optionally: `z_real_B_native`

**Training objective options (pick one as v1)**

- [x] **Cone-matching loss (recommended)**: learn `F` so that cone _signatures_ match in env B when using `F(z_intent_A)`:
  - `L = ||sig_B(F(zA_intent), s0_B) - sig_A(zA_intent, s0_A)||`
  - where `sig = [log_cone_vol, E[T], bind_prob, calibration@k*]`
- [ ] **Teacher loss (bootstrap)**: regress `F(zA_intent)` to `z_intent_B_native` (if you trust native B intent):
  - `L = ||F(zA_intent) - zB_intent_native||^2`
- [x] Add regularization:
  - `||F(z)||` penalty
  - Jacobian/Lipschitz penalty (optional) for stability

**Implementation:** `harness/intent_functor.py`:

- `IntentFunctorTrainer` uses cone-matching loss
- Training via Evolution Strategies (ES) since cone evaluation is non-differentiable
- `IntentSignature` = (log_cone_vol, E_T, bind_rate)

**Implementation**

- [x] Make `FunctorIntentNet`:
  - start with linear / 2-layer MLP
  - input: `z_intent_A` (+ optional small context embedding of `s0_B`)
  - output: `z_intent_B`
- [x] Train functor with actor/tube frozen in both envs.
- [x] Log functor metrics:
  - loss curve
  - distribution of `F(z_intent_A)` norms
  - cone-signature correlation A vs mapped-B

**Implementation:**

- `LinearIntentFunctor`: F(z) = LayerNorm(Wz + b) with identity init
- `MLPIntentFunctor`: z + MLP(z) with residual connection
- Both include LayerNorm for scale stability

---

### H. Evaluation: "does mapped intent work in B without learning B's intent?"

Define a clean test that matches your goal.

- [x] Freeze everything except functor during functor training.
- [x] Evaluate in env B under these conditions:
  1. **Native**: `z_intent_B ~ q_B`, `z_real_B ~ q_B`
  2. **Transported intent**: `z_intent_B = F(z_intent_A)` (A sampled from matched A state), `z_real_B ~ q_B`
  3. **Intent random**: random intent, `z_real_B ~ q_B`
  4. **Oracle transform** (for mirrored/rotated): apply known transform on _state_ and compare against transported intent (positive control)
- [x] Primary success metric:
  - outcome / return (or negative cost)
- [x] Secondary:
  - bind success rate
  - calibration curve / MAE
  - cone signature preservation correlation
  - compute usage (Hr) if enabled

**Implementation:** `IntentFunctorTrainer.evaluate()`:

- Compares transported vs native vs random intent
- Tracks cone signature correlations
- Generates comparison plots

**Expected result pattern if factoring worked**

- intent-random should degrade cone stats & success
- transported-intent should beat intent-random and approach native
- transported-intent should preserve cone signatures more than outcome if policy is still weak

---

### I. Fix two implementation hazards (these bite hard)

- [x] **Axis/scale issues**: ensure intent latents are normalized consistently across envs (LayerNorm or fixed std prior). A lot of functor "explosions" come from unmatched scales.
- [x] **Deterministic vs stochastic sampling**:
  - During functor training, use `z_intent = μ` (deterministic) first.
  - Then add sampling once stable.

**Implementation:**

- `FactoredActorNet` includes `LayerNorm` on z_intent output
- `get_intent_embedding()` returns deterministic mean for functor training
- Functors include `LayerNorm` on output

---

### J. Add "equivalence" positive-controls for your env pairs

So you can tell if failures are theory vs bug.

- [ ] For mirrored/rotated:
  - implement a _known_ mapping baseline:
    - `F_oracle(z_intent)` = learned? no—just identity, but use matched `s0_B` transform
  - verify that "native B" behaves similarly when fed transformed `s0`
- [ ] For shifted-rules:
  - add an explicit rule permutation head baseline (since equivalence is partly discrete)

---

### K. Deliverables / definition of done

- [x] Factored actor trains and performs at least as well as unfactored baseline on one env.
- [ ] Cone stats (log cone vol, E[T], bind rate) respond primarily to `z_intent` (ablation test).
- [x] Transfer harness updated so reuse modes operate on `z_intent` only.
- [x] Functor training runs without exploding loss and shows:
  - improved cone-signature preservation vs intent-random
  - transported-intent improves task outcome vs intent-random on at least mirrored or rotated
- [x] A single summary dashboard for each env-pair:
  - outcome table (native vs transported vs random)
  - cone signature correlation plots
  - calibration curve overlay

---

## Files Created

| File                          | Description                                             |
| ----------------------------- | ------------------------------------------------------- |
| `factored_actor.py`           | `FactoredActor` class with separate z_intent and z_real |
| `networks.py` (updated)       | `FactoredActorNet`, `FactoredSharedTube`                |
| `harness/intent_functor.py`   | Intent-only functor training                            |
| `harness/factored_example.py` | Example script for factored training                    |
| `envs/equivalent_envs.py`     | Topologically equivalent environments                   |

## Usage

```bash
# Run factored training + intent functor learning
python3 packages/tam/python/harness/factored_example.py
```

## Key Design Decisions

1. **z_intent determines cone geometry**: sigma and stop heads use ONLY z_intent
2. **z_real is for policy execution**: policy uses full z, allowing environment-specific actions
3. **Separate KL weights**: z_real has weaker KL regularization (more freedom)
4. **LayerNorm on z_intent**: ensures consistent scale across environments for functor
5. **set_intent() method**: cleanly separates transported intent from native realization
