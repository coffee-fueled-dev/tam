# Homeostasis Study

Self-calibrating tube geometry: discovering optimal bind rates without a fixed target.

## Context in TAM

In the Trajectory-Affordance Model, an agent must balance two competing pressures:

- **Agency**: Tighter tubes (smaller σ) = stronger commitment = more agency
- **Reliability**: Wider tubes = fewer binding failures

The naive approach is to set a fixed bind target ρ (e.g., 85%). This study eliminates that hyperparameter entirely—the optimal bind rate **emerges** from environment structure.

## Key Insight

Replace the fixed target with a **self-calibrating λ** (failure price) that adjusts based on **surprise**:

- Failures in "easy" situations (low hardness) → surprising → increase λ → tighten tubes
- Failures in "hard" situations (high hardness) → expected → decrease λ → allow wider tubes

The equilibrium is the fixed point where marginal benefit of tightening equals marginal cost of failures.

## Files

| File       | Purpose                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| `actor.py` | Actor with self-calibrating homeostasis (soft fail, z-score normalization) |
| `train.py` | Training harness demonstrating emergent per-rule bind rates                |

## Expected Results

```
Easy (rule 0): bind=1.000, σ=0.058, log_vol=-2.86
Hard (rule 3): bind=0.913, σ=0.158, log_vol=-1.87

✓ Sigma calibrated: hard 2.7x wider than easy
✓ Bind rate calibrated: hard < easy
✓ Volume NOT hitting floor
✓ λ stabilized at ~1.5
```

The agent learns **different operating points per rule** without being told what they should be.

## Usage

```bash
python train.py --name zscore --steps 10000
```

## Relationship to Other Studies

- **tube_geometry/**: Provides the base architecture (TubeNet with σ from z only)
- **competitive-port-binding/**: Extends homeostasis with CEM-based deliberation over candidate commitments
