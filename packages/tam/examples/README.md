# TAM Examples

Experiments demonstrating TAM's core capabilities, organized by complexity.

## Running Experiments

```bash
bun run examples/01-basic-training.ts
bun run examples/02-multi-horizon.ts
bun run examples/03-specialization.ts
bun run examples/04-encoder-learning.ts
```

## Experiment Progression

### 01. Basic Training
**Concepts**: Single-domain learning, grokking, calibration

**What it shows**:
- TAM learns simple dynamics from observations
- Grokking: Sudden generalization on held-out test data
- Agency correlates with accuracy when properly trained
- Coverage rate measures epistemic integrity

**Expected results**:
- Error < 0.05 after 1000 episodes
- Grokking around episode 200-500
- Agency > 90% with good coverage

**Run time**: ~2 minutes

---

### 02. Multi-Horizon Prediction
**Concepts**: Temporal abstraction, long-horizon planning

**What it shows**:
- Long-horizon rollouts fail with iterative 1-step (error compounds)
- Direct k-step prediction learns actual dynamics (no compounding)
- Different horizons = different specialists
- ~70% error reduction for 20-step predictions

**Expected results**:
- 1-step iterative: 0.49 error (20 steps)
- 20-step direct: 0.14 error (73% improvement!)

**Run time**: ~5 minutes

---

### 03. Multi-Mode Specialization
**Concepts**: Port proliferation, specialist discovery

**What it shows**:
- TAM discovers distinct behaviors automatically
- No manual mode switching needed
- Agency-based selection picks appropriate specialist
- Each specialist commits to narrow cones for its mode

**Expected results**:
- Discovers 4 specialist ports (left/right/up/down)
- High agency when mode is clear
- Good coverage (commitments honored)

**Run time**: ~2 minutes

---

### 04. Encoder Learning
**Concepts**: Feature discovery, end-to-end learning

**What it shows**:
- TAM learns encoders from raw features
- Automatically ignores noise, discovers relevant features
- Compares: static (all features) vs hand-crafted vs learnable

**Expected results**:
- Learnable encoder approaches hand-crafted performance
- Significantly better than static (which includes noise)

**Current limitation**: Uses temporal smoothness loss (TensorFlow.js lacks stopGradient)

**Run time**: ~3 minutes

---

## Harness Architecture

All experiments use a standardized harness (`./harness/`) that provides:

### Domain Interface
```typescript
interface Domain<S> {
  name: string;
  randomState(): S;
  simulate(state: S, steps?: number): S;
  embed(state: S): number[];
  embeddingDim: number;
}
```

### Standard Metrics
- **Prediction Error**: `||predicted - actual||`
- **Agency**: Commitment strength (1 - normalized cone volume)
- **Coverage Rate**: % of trajectories inside cones
- **Calibration**: Correlation between agency and accuracy

### Training Loop
```typescript
const result = await runExperiment("My Experiment", {
  domain: myDomain,
  episodes: 500,
  checkpointEvery: 100,
  testSize: 100,
});
```

Automatic:
- Checkpoint collection at intervals
- Test set evaluation (held-out)
- Grokking detection
- Calibration analysis
- Result comparison

---

## Available Domains

| Domain | Purpose | Dim | Complexity |
|--------|---------|-----|------------|
| `dampedSpring1D` | Basic dynamics | 2D | Simple |
| `driftModes` | Multi-mode behavior | 6D | Medium |
| `noisyPendulum` | Encoder learning | 4-10D | Medium |

Create custom domains by implementing the `Domain<S>` interface.

---

## Key Metrics Explained

### Agency ≠ Accuracy (but they correlate!)

**Agency**: How confident is the actor?
- Measured by cone volume
- High agency = narrow cones = strong commitments

**Accuracy**: How correct are predictions?
- Measured by prediction error
- Low error = good world model

**Coverage (Binding)**: Does reality honor commitments?
- Measured by % of trajectories inside cones
- High coverage = epistemic integrity

**Well-Calibrated Model**:
- High agency → Low error (confident = accurate)
- High agency → High coverage (commitments honored)
- Together: Confidence matches reality

---

## Legacy Experiments

Old experiments (in `experiments/` subfolder) are preserved for reference but use ad-hoc code:
- `predictive-accuracy.ts` - Early calibration analysis
- `complete-calibration.ts` - Full agency×accuracy×binding
- `port-functors.ts` - Functor discovery (composition)
- `homeostasis.ts` - Multi-actor coordination
- Others - Various specialized tests

These will be migrated to use the harness or removed.

---

## Next Steps

### Immediate
- [ ] Add 05: Compositional generalization (functors across domains)
- [ ] Test sample efficiency improvements
- [ ] Benchmark on standard RL environments

### Research
- [ ] Port to PyTorch (for true binding/agency loss)
- [ ] High-dimensional encoders (vision)
- [ ] Hierarchical planning with multi-horizon
- [ ] Continuous control (Neural ODE ports)

---

## Contributing New Experiments

1. Define your domain (implement `Domain<S>`)
2. Use `runExperiment()` from harness
3. Add custom callbacks if needed (encoder training, etc.)
4. Document: What it shows, expected results, run time
5. Keep experiments < 200 lines of code

Example:
```typescript
import { runExperiment, myDomain } from "./harness";

async function main() {
  const result = await runExperiment("My Experiment", {
    domain: myDomain,
    episodes: 500,
  });

  // Custom analysis
  console.log("Special metric:", computeSpecialMetric(result));

  result.bank?.dispose();
}

main().catch(console.error);
```

---

## Troubleshooting

**"Grokking not detected"**
- Try more episodes (1000+)
- Check domain complexity (may need more training)
- Verify test set is truly held-out

**"Poor coverage rate"**
- Model may be overconfident (high agency, low coverage)
- Increase training episodes
- Adjust specialist threshold

**"No specialization"**
- Lower `specialistThreshold` in bank config
- Ensure domain has distinct modes
- Verify sufficient training samples per mode

**"High error despite training"**
- Check domain is deterministic (or low noise)
- Verify embedding captures relevant features
- May need encoder learning (04)
