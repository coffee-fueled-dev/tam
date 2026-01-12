# Tube Geometry Study

Foundational study establishing how the latent commitment variable **z** can control tube (affordance cone) geometry.

## Context in TAM

In the Trajectory-Affordance Model, an agent commits to a "tube" in trajectory space—a probabilistic envelope around a predicted future. This study validates the core geometric mechanism:

- **Tube**: Gaussian envelope defined by μ(t) (mean trajectory) and σ(t) (width)
- **Binding**: Success when the actual trajectory stays within k·σ of μ
- **Agency**: Inversely proportional to tube volume (tighter tubes = more agency)

The key empirical finding: **σ should depend only on z, not on s₀**. This forces z to become the sole controller of uncertainty geometry, creating a clean separation between "what trajectory" (μ from s₀, z) and "how certain" (σ from z alone).

## Files

| File                 | Purpose                                                                        |
| -------------------- | ------------------------------------------------------------------------------ |
| `actor.py`           | **Core implementation**: TubeNet + Policy with the key insight (σ from z only) |
| `train_actor.py`     | Training harness for the Actor on a simple 2D navigation environment           |
| `visualize_tubes.py` | Visualization utilities (2D tubes, cone profiles, z-space, 3D views)           |

## Key Results

1. **z controls cone geometry**: Perturbing z dimensions produces predictable changes in tube volume
2. **Rule differentiation**: Different environment dynamics (rules 0-3) produce distinct z encodings
3. **Bind-volume tradeoff**: The model learns to balance tube tightness against binding reliability

## Downstream Studies

This study provides the foundation for:

- **homeostasis/**: Self-calibrating optimal bind rate discovery
- **competitive-port-binding/**: CEM-based deliberation over candidate commitments

## Usage

```bash
# Train the actor
python train_actor.py --name baseline --steps 4000

# Visualize a trained model
python visualize_tubes.py --model runs/<run_name>/model.pt
```
