# TAM Examples

Demonstrations of the Trajectory-Affordance Model's core capabilities.

## Core Examples (01-04)

### 01: Basic Training
Introduction to TAM fundamentals: binding, agency, and coverage.

### 02: Multi-Horizon Learning
Demonstrates temporal abstraction with specialists at different time scales.
- Shows how TAM naturally supports multi-scale prediction
- Direct 20-step prediction significantly outperforms chained 1-step

### 03: Specialization via Proliferation
Shows automatic discovery of distinct behavioral modes.
- No manual mode switching required
- Agency-based selection chooses appropriate specialists

### 04: Encoder Learning
Demonstrates learning state representations from raw observations.

## Advanced Properties (05, 07-09)

### 05: Compositional Learning
**Key Demo:** Bridging discontinuous dynamics through functor discovery.

**Domain:** Piecewise potential wells
- Left well: Oscillates around x=-1
- Right well: Oscillates around x=+1
- Gap: No restoring force (ballistic motion)

**Challenge:** Gap has fundamentally different dynamics than wells. Simple interpolation CANNOT work!

**Solution:** Online learning discovers functors that map between oscillatory ↔ ballistic regimes.

### 07: Double Grokking
**Key Demo:** Two distinct learning phases (dynamics + epistemics).

Shows TAM learns in two stages:
1. First grokking (~sample 100): Learns WHAT happens (dynamics)
2. Second grokking (~sample 250): Learns HOW SURE to be (epistemics)

Demonstrates separation of ontology (world model) from epistemology (confidence).

### 08: Agency Requires Causality
**Key Demo:** Permutation test showing agency depends on causal structure.

Compares two conditions:
- Causal: Correct before→after pairs → Agency jumps
- Permuted: Shuffled pairs → Agency stays flat

Proves TAM learns genuine causal structure, not just statistical patterns.

### 09: Dimensional Discovery
**Key Demo:** Discovering causal dimensions in high-dimensional noisy spaces.

**Environment:** 10D (2 causal + 8 noise)
- Dimensions 0-1: Spring dynamics (causal)
- Dimensions 2-9: Pure random noise

**Outcome:**
- 100% coverage on causal dimensions
- 0% coverage on noise dimensions  
- 97%+ agency after convergence

Demonstrates:
- Epistemic humility (starts with wide cones)
- Anisotropic attention (learns dimensional relevance)
- Homeostatic equilibrium (adapts to environment noise)

## Key Concepts

### Agency
Measure of commitment specificity. High agency = narrow cones = confident predictions.

### Coverage
Fraction of predictions that successfully bind (error < threshold).

### Binding
Checking if actual trajectory falls within predicted affordance cone.

### Grokking
Sharp phase transition where model suddenly "gets it" after gradual learning.

### Proliferation
Automatic creation of specialist ports for different modes/contexts.

### Functors
Learned transformations between different actors' state spaces.

### Epistemic Humility
Starting with maximally wide cones ("I don't know yet") and learning to narrow through experience.

## Running Examples

```bash
# Run any example
bun run examples/01-basic-training.ts
bun run examples/07-double-grokking.ts
bun run examples/09-dimensional-discovery.ts

# Results saved to examples/results/*.json
```

## For ARC-AGI

The most relevant examples for ARC-AGI are:
- **05-composition.ts**: Bridging discontinuous transformations
- **09-dimensional-discovery.ts**: Finding causal structure in noise
- **03-specialization.ts**: Discovering distinct behavioral modes

These demonstrate TAM's ability to discover structure, compose knowledge, and handle complex transformations—core requirements for ARC-AGI tasks.
