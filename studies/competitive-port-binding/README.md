# Competitive Port Binding Study

CEM-based deliberation over candidate commitments: "thinking" before acting.

## Context in TAM

TAM claims agency is the capacity to commit to specific futures. But which commitment should the agent select? This study implements a **deliberation engine** that evaluates candidate z's before binding:

```
S(s₀, z) = -intent_proxy + α·agency - λ·risk
```

Where:
- **Intent proxy**: How well μ_T aligns with goal (available at bind time)
- **Agency**: Tube tightness (-log σ)
- **Risk**: Learned critic r_ψ(s₀, z) → p_fail

The agent uses Cross-Entropy Method (CEM) to search z-space, iteratively refining toward high-scoring commitments.

## Key Experiments

### 1. Binding Mode Comparison (`train.py`)

Compares four z-selection strategies:

| Mode | Description |
|------|-------------|
| Random | Sample one z ~ q(z\|s₀) |
| Best-of-K | Sample K, pick best score |
| CEM | Iterative refinement (4 rounds) |
| Oracle | Use true outcome (diagnostic upper bound) |

**Expected result**: CEM achieves tighter tubes than Random while maintaining acceptable bind rate—trading coverage for agency.

### 2. Bimodal Commitment (`train_bimodal.py`)

Tests the core TAM claim: in bimodal environments, the agent should **commit** to one future rather than **hedge** by expanding acceptance.

Environment: Same s₀ produces two incompatible trajectories (hidden mode ±1). A midline hedge covers both but sacrifices agency.

**Key metric**: Mode Commitment Index (MCI)
- MCI ≈ 0 → hedging (tube at midline)
- MCI ≈ 1 → committed to one mode

**Expected result**: CEM achieves high MCI with bimodal d(z) histogram; Random/Hedge stay near midline.

## Files

| File | Purpose |
|------|---------|
| `actor.py` | CompetitiveActor with CEM binding, RiskNet critic, scoring function |
| `train.py` | Binding mode comparison on 4-rule environment |
| `train_bimodal.py` | Bimodal commitment experiment (rejection of hedging) |

## Usage

```bash
# Standard competitive binding experiment
python train.py --name v2 --steps 10000 --cem-iters 4

# Bimodal commitment test
python train_bimodal.py --name test --steps 10000 --bend-amplitude 0.15
```

## Relationship to Other Studies

- **tube_geometry/**: Provides base TubeNet architecture
- **homeostasis/**: Provides self-calibrating λ dynamics (used here for tube training)
- This study adds the **deliberation layer**: choosing among commitments rather than sampling one
