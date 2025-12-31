# TAM Hyperparameters & Magic Numbers

Complete audit of all hardcoded values, assumptions, and configuration defaults in the geometric TAM implementation.

## Neural Network Architecture

### CommitmentNet (Distance Prediction)
**File:** `src/geometric/commitment.ts`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `hiddenSizes` | `[64, 32]` | Two-layer MLP for situation-dependent learning | ✅ Yes - depends on input dim |
| `learningRate` | `0.01` (default) / `0.001` (stable) | Adam optimizer step size | ✅ Yes - lower = more stable |
| `batchSize` | `32` (default) / `10` (responsive) | Training batch size | ✅ Yes - tradeoff: speed vs stability |
| `initialRadius` | `2.0` (unused legacy) | Deprecated - see biasInit | ❌ Legacy |
| `minRadius` | `0.01` | Floor to prevent numerical instability | ⚠️ Principled - numerical stability |
| `narrowStep` | `0.1` | Additive increase on success: `d_new = d + 0.1` | ✅ Yes - controls learning speed |
| `widenStep` | `0.2` | Additive decrease on failure: `d_new = d - (violation × 0.2)` | ✅ Yes - typically 2× narrowStep |
| **Initialization** | | | |
| `biasInitializer` | `-2.0` | softplus(-2) ≈ 0.13 for epistemic humility | ⚠️ Principled - wide initial cones |
| `kernelInitializer` | `randomNormal(mean=0, stddev=0.01)` | Small weights for gradual learning | ⚠️ Standard NN practice |
| `hiddenBiasInit` | `zeros` | Standard for ReLU | ⚠️ Standard NN practice |
| `hiddenKernelInit` | `heNormal` | Kaiming initialization for ReLU | ⚠️ Standard NN practice |
| **Output Clamp** | | | |
| `maxDistance` | `100` | Upper bound on predicted distance | ✅ Yes - prevents overflow |

**Key Insight:** Bias init of `-2.0` implements **epistemic humility** - system starts with wide cones (low commitment) and learns to narrow through experience.

---

### CausalNet (Trajectory Prediction)
**File:** `src/types.ts` → `defaultCausalNetConfig`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `hiddenSizes` | `[64, 32]` | Same as CommitmentNet for symmetry | ✅ Yes |
| `learningRate` | `0.01` | Adam optimizer step size | ✅ Yes |
| `batchSize` | `32` | Training batch size | ✅ Yes |

---

## Geometric Port Configuration

### Embeddings & Alignment
**File:** `src/geometric/port.ts`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `embeddingDim` | `8` (default) / `10` (experiment) | Dimension of port embeddings | ✅ Yes - match state dim |
| `embeddingLearningRate` | `0.01` | Gradient step for embedding updates | ✅ Yes |
| `embeddingInitRange` | `[-0.05, 0.05]` | `(Math.random() - 0.5) × 0.1` | ⚠️ Small random init |
| `embeddingDecreaseScale` | `0.5` | Asymmetric: decrease slower than increase | ✅ Yes - bias toward attending |
| **Softmax Temperature** | | | |
| `temperature` | `2.0` | Controls attention concentration | ✅ Yes - higher = more uniform |

**Softmax Temperature Scale:**
- `T = 1.0`: Standard softmax (can be too peaked)
- `T = 2.0`: Moderate differentiation (current)
- `T → ∞`: Uniform distribution

---

### Aperture & Alignment
**File:** `src/types.ts` → `defaultGeometricPortConfig`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `defaultAperture` | `1.0` | Base window size (intrinsic port capacity) | ✅ Yes - domain dependent |
| `minAlignmentThreshold` | `0.1` | Ports with α < 0.1 are inapplicable | ✅ Yes - determines when port applies |

**Cone Radius Formula:**
```
radius[i] = aperture × alignment[i] / (1 + distance)
```

---

## Refinement Control

### Binding Rate Policy
**File:** `src/types.ts` → `defaultGeometricPortConfig`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `equilibriumRate` | `0.95` (default) / `0.3` (experiment) | Target binding success rate | ✅ Yes - domain dependent |
| `bindingRateTolerance` | `0.05` (5%) | Accept ± tolerance around target | ✅ Yes - control stability |
| `bindingRateDecay` | `0.1` | EMA decay (effective ~10 sample window) | ✅ Yes - memory span |
| `bindingRateEMAInit` | `0.5` | Neutral prior for new ports | ⚠️ Principled - maximum entropy |

**Equilibrium Rate Interpretation:**
- `0.95`: Conservative (default) - only narrow when very confident
- `0.50`: Balanced - equal success/failure
- `0.30`: Aggressive (experiment) - explore more, commit less

---

### Agency Gradient Policy
**File:** `src/geometric/refinement.ts`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `trajectoryWindowSize` | `20` | Samples for gradient estimation | ✅ Yes |
| `dampingFactor` | `0.3` | Velocity threshold for damping | ✅ Yes - prevent oscillation |
| `significanceThreshold` | `0.05` (5%) | Minimum change to consider significant | ✅ Yes |
| `bindingRateThreshold` | `0.3` (30%) | Below this, widen aggressively | ✅ Yes |

---

## Proliferation Control

**File:** `src/types.ts` → `defaultGeometricPortConfig`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `enableProliferation` | `false` | Off by default | ✅ Feature flag |
| `proliferationAgencyThreshold` | `0.3` (30%) | Proliferate if agency consistently low | ✅ Yes |
| `proliferationMinSamples` | `20` | Need data before deciding | ✅ Yes - statistical confidence |
| `proliferationCooldown` | `50` | Samples between proliferations | ✅ Yes - prevent explosion |

---

## Binding Evaluation

### Fibration (Cone Binding)
**File:** `src/geometric/fibration.ts`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| **Numerical Stability** | | | |
| `minRadius` | `1e-8` | Prevent division by zero | ❌ Numerical stability |
| `minWeight` | `1e-8` | Prevent division by zero in alignment | ❌ Numerical stability |
| `minDiagonal` | `1e-8` | Cholesky solver stability | ❌ Numerical stability |
| **Binding Criterion** | | | |
| `normalizedDistance ≤ 1` | Success | Standard: inside unit sphere in normalized space | ⚠️ Principled |
| `normalizedDistance > 1` | Failure | Outside acceptance region | ⚠️ Principled |

**Key Implementation:** Uses **alignment-weighted mean radius** for reference:
```typescript
// CORRECT (current):
refRadius = Σ(alignment[i] × radius[i])

// WRONG (previous):
refRadius = Σ(radius[i]) / dim  // Noise dims drag it down!
```

---

## History Tracking

**File:** `src/geometric/history.ts`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| `familiarityThreshold` | `10` | Samples before situation is "familiar" | ✅ Yes - statistical confidence |
| `recentWindowMs` | `60000` (1 min) | Time window for "recent" | ✅ Yes - domain dependent |
| `agencyWindowSize` | `50` | Rolling window for agency | ✅ Yes |
| `bimodalRatioThreshold` | `0.25` (25%) | Within-cluster / total variance | ✅ Yes - sensitivity |
| `minFailuresForBimodal` | `10` | Need data for cluster detection | ✅ Yes |

---

## Experiment-Specific Constants

### Test Environment (09g)
**File:** `examples/09g-binding-rate-with-softmax.ts`

| Parameter | Value | Rationale | Tunable? |
|-----------|-------|-----------|----------|
| **Spring Dynamics** | | | |
| `k` (spring constant) | `1.0` | Restoring force | ✅ Yes - domain |
| `b` (damping) | `0.1` | Friction | ✅ Yes - domain |
| `dt` (time step) | `0.1` | Integration step | ✅ Yes - domain |
| **State Initialization** | | | |
| `stateRange` | `[-1, 1]` | `(Math.random() - 0.5) × 2` | ✅ Yes - domain |
| **Evaluation** | | | |
| `coverageThreshold` | `0.1` | Error < 0.1 counts as "bound" | ✅ Yes - domain dependent |
| `testSetSize` | `100` | States for evaluation | ✅ Yes - statistical |
| `totalSamples` | `5000` | Training duration | ✅ Yes - convergence |
| `checkpointEvery` | `100` | Logging frequency | ✅ Yes - monitoring |
| **Stability Detection** | | | |
| `varianceThreshold` | `0.01` | For "stable" distance | ✅ Yes - sensitivity |
| `stabilityWindow` | `5` | Last N checkpoints | ✅ Yes |
| `minStableDistance` | `0.1` | Above this is "not collapsed" | ✅ Yes |

---

## Debug Logging

**File:** `src/geometric/commitment.ts`

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `narrowLogProb` | `0.01` (1%) | Log NARROW training occasionally |
| `narrowDebugProb` | `0.02` (2%) | Detailed NARROW debug |
| `widenLogProb` | `0.01` (1%) | Log WIDEN training occasionally |

---

## Summary by Category

### 🔴 **Principled (Don't Change)**
- Numerical stability constants (`1e-8`)
- Epistemic humility initialization (`bias = -2.0`)
- Standard NN initialization (He, Xavier)
- Neutral priors (`0.5` for unknown binding rate)
- Binding criterion (normalized distance ≤ 1)

### 🟡 **Standard Practice (Change with Care)**
- Neural network architectures (`[64, 32]`)
- Learning rates (`0.01`, `0.001`)
- Batch sizes (`10`, `32`)
- Softmax temperature (`2.0`)

### 🟢 **Domain-Dependent (Expect to Tune)**
- `equilibriumRate` (0.3 for exploration, 0.95 for exploitation)
- `aperture` (intrinsic port capacity)
- `embeddingDim` (match state dimension)
- `narrowStep` / `widenStep` (learning speed)
- Coverage threshold (`0.1` for "good enough")
- Test environment parameters (spring constants, ranges)

### 🔵 **Feature Flags**
- `enableProliferation` (false by default)
- `useDynamicReference` (true by default)
- `enablePortFunctors` (false by default)

---

## Critical Ratios & Relationships

### Learning Rate Relationships
```
embeddingLearningRate : commitmentLearningRate : causalLearningRate
        0.01          :         0.001          :       0.01
```
**Current:** Embedding and causal learn faster than commitment (for stability)

### Step Size Relationships
```
widenStep = 2 × narrowStep
  0.2     =     2 × 0.1
```
**Rationale:** Widen faster than narrow (conservative narrowing)

### Temperature vs Attention Concentration
```
T = 0.5 → Very peaked (almost one-hot)
T = 1.0 → Standard softmax
T = 2.0 → Moderate (current)
T = 5.0 → Nearly uniform
```

### Equilibrium Rate vs Environment Noise
```
Low Noise (10%) → equilibriumRate = 0.95 (tight binding)
Med Noise (50%) → equilibriumRate = 0.70 (balanced)
High Noise (80%) → equilibriumRate = 0.30 (loose binding)
```

---

## Recommended Tuning Procedure

1. **Start Conservative:**
   - Use default configs
   - `equilibriumRate = 0.95`
   - `learningRate = 0.001`

2. **Match to Domain:**
   - Set `embeddingDim` = state dimension
   - Adjust `aperture` based on typical delta magnitudes
   - Set `coverageThreshold` based on acceptable error

3. **Tune Equilibrium:**
   - Observe natural binding rate
   - Set `equilibriumRate` to maintain exploration
   - Adjust `bindingRateTolerance` for stability

4. **Adjust Learning Speed:**
   - Increase `narrowStep` / `widenStep` if converging slowly
   - Decrease `learningRate` if oscillating
   - Increase `batchSize` for smoother updates

5. **Monitor:**
   - Check distance trajectory (should stabilize)
   - Check binding rate (should oscillate around target)
   - Check causal coverage (should approach 100%)

---

## Open Questions

1. **Temperature Selection:** Is `T = 2.0` optimal for all dimensions, or should it scale with `embeddingDim`?

2. **Step Size Scaling:** Should `narrowStep` scale with current distance? (Additive vs proportional)

3. **Architecture Depth:** Are two hidden layers `[64, 32]` sufficient for high-dimensional spaces?

4. **Batch Size Dynamics:** Should batch size adapt based on binding rate volatility?

5. **Initialization Variance:** Is `stddev = 0.01` for kernel init optimal, or should it depend on input dimension?

---

## Change Log

- **2025-12-31:** Initial audit
  - Identified 40+ hyperparameters
  - Categorized by tunability
  - Documented current experimental values
