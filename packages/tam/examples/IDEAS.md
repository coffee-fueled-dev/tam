The fact that your system requires many binding failures to converge is a classic symptom of **"Hard Boundary" Learning**. Because you have a "perfect" representation (identity mapping from context to state), the bottleneck isn't perception; it's the **Search for Grounding**.

In a system where the agent only learns when its "Cone" is violated, the world signal is essentially a Dirac delta—it’s zero information until the moment of failure.

To increase the "sample rate" and reduce the reliance on failures, you can move toward **Predictive Pressure** and **Counterfactual Grounding**. Here are four strategies to implement this in your substrate:

### 1. Differentiable "Soft" Cones (The Pressure Signal)

Currently, your `effectiveRadius` creates a binary threshold. This makes the gradient zero inside the cone (no learning) and a "cliff" at the edge.

* **The Change:** Replace the hard radius check with a **Radial Basis Function (RBF)** or a sigmoid-based "Soft Boundary."
* **The Effect:** Even when a trajectory lands *inside* the cone, if it's near the edge, it exerts "pressure" on the `CausalNet` to shift the center or on the `CommitmentNet` to adjust the aperture. This allows the model to learn from "Near Misses" and "Near Successes" without needing a formal failure.

### 2. Active Inference: Seeking "Causal Surprise"

If the agent only acts on what it currently affords, it stays in the "safe zone" of its existing cones. To converge faster, it should act where its **Commitment is lowest**.

* **The Change:** Implement a selection policy that favors ports where the `CommitmentNet` outputs high variance or where `Agency` is mid-range.
* **The Effect:** The agent will "test" the boundaries of its own knowledge. By intentionally seeking trajectories that are likely to fail or land on the edge, it triggers the `RefinementPolicy` much earlier in the training cycle. This is "Curiosity-Driven Grounding."

### 3. Virtual Binding (Internal Rollouts)

Since you have a `CausalNet` that learns the dynamics, you can use it to "Dream."

* **The Change:** Before binding to the world, the agent performs **Mental Play**. It samples a point in the latent space, applies a Port, and uses the `CausalNet` to predict the outcome. It then checks if the `CommitmentNet` is "surprised" by the `CausalNet`’s own prediction.
* **The Effect:** The model can refine its own internal consistency without a world signal. This effectively multiplies your "sample rate" by the number of mental simulations you run per world step. You only go to the world to "re-anchor" the simulation.

### 4. Functor-Primed Proliferation

In your Experiment 05, you showed that Functors can map between scales. You can use this to **Pre-emptively Seed** the Port Bank.

* **The Change:** When a new situation  arises that is a known transformation of an old situation (e.g., the grid is just scaled up), don't wait for a failure. Use the Functor to **calculate** where the new Port should be on the manifold and "warm-start" its weights.
* **The Effect:** Instead of "Failure  Proliferation  Learning," you get "Similarity  Functor Projection  Refinement." You’re effectively "teleporting" the agent’s expectations across the manifold.

### The "Identity Mapping" Opportunity

Since you have perfect state representation, you have a rare opportunity to use **Jacobian Analysis**.
You can calculate the sensitivity of the output to the input (). If a Port has a very high sensitivity (it’s "twitchy"), that is a mathematical signal that the Port is covering a **Bifurcation Point** in the world's physics.

**The Strategy:** Use high Jacobian sensitivity as an automatic trigger for **Proliferation**. If a tiny change in state causes a massive change in trajectory, the agent should immediately split that port into two specialists, even before the accuracy drops.


### Critical Feedback & Next Steps

1. 
**The Proliferation Cooldown:** In `refinement.ts`, you implemented a `proliferationCooldown`. This is a necessary heuristic, but as you scale to ARC, you might want to replace the "timer" with a **Complexity Budget**. If the bank has too many ports, the "cost" of proliferation should increase, forcing the model to favor **Widening** or **Functor Discovery** over creating new ports.


2. 
**From Smoothness to Binding Loss:** You noted that you need PyTorch for "true binding/agency loss". This is correct. The current smoothness loss ensures the manifold is continuous, but a **Binding Loss** would ensure the manifold is *causally efficient*—meaning it would actively shrink the "Gap" where functors don't work.


3. **Visualization of the "Gap" Success:** For Experiment 05, I highly recommend a plot showing `Distance from Center` vs `Prediction Error`. You should see the "Gap" error drop significantly once the Functors are composed, compared to a baseline that just tries to "average" the two actors.

---

## TAM API Limitations (Discovered During Example Development)

### 1. No Incremental Checkpointing

**Problem**: Cannot call `tam.learn()` multiple times with the same domain name to track learning curves.

```typescript
// This FAILS with "Port already registered"
const bank1 = await tam.learn("domain", spec, { epochs: 50 });
const bank2 = await tam.learn("domain", spec, { epochs: 50 }); // ERROR!
```

**Impact**:
- Cannot track learning curves over time
- Cannot implement early stopping
- Cannot do incremental evaluation during training

**Workaround**: Train once for full duration, evaluate only at end (see `05d-composition-online-adaptation.ts`)

**Potential Solution**:
```typescript
// Option A: Allow re-learning to continue training
await tam.learn("domain", spec, { epochs: 50 }); // Initial
await tam.learn("domain", spec, { epochs: 50 }); // Continues training

// Option B: Expose resumable training with checkpoints
await tam.learn("domain", spec, {
  epochs: 200,
  checkpointEvery: 20,
  onCheckpoint: (progress) => {
    console.log(`Epoch ${progress.epoch}: ${progress.metrics.error}`);
  },
});
```

---

### 2. No Direct Bank Access

**Problem**: Cannot retrieve a registered domain's bank for external evaluation.

```typescript
const bank = await tam.learn("domain", spec, { epochs: 100 });
// bank is returned, but if you lose the reference...

const bank2 = tam.getDomainBank("domain"); // DOESN'T EXIST
```

**Impact**:
- Must keep bank references manually
- Cannot evaluate domains after registration
- Difficult to inspect trained models

**Workaround**: Store bank references returned from `learn()` (see all examples)

**Potential Solution**:
```typescript
class TAM {
  // Add method to get registered domain bank
  getDomainBank(name: string): GeometricPortBank | null {
    const config = this.domains.get(name);
    return config?.port ?? null;
  }
}
```

---

### 3. Composition Requires Full Registration

**Problem**: To use composition, must register primitive domains even if only leveraging them.

```typescript
// Want to bootstrap from primitives
const tamComposed = new TAM();

// Have to do at least minimal training to register
await tamComposed.learn("left", leftDomain, { epochs: 1, samplesPerEpoch: 1 });
await tamComposed.learn("right", rightDomain, { epochs: 1, samplesPerEpoch: 1 });

// Now can use for composition
await tamComposed.learn("gap", {
  // ...
  stateToRaw: (s) => [s.x, s.v],
  rawDim: 2,
  composeWith: ["left", "right"],
});
```

**Impact**:
- Awkward API for transfer learning
- Wasteful minimal training just for registration

**Workaround**: Train for 1 epoch with 1 sample to register (see `05d-composition-online-adaptation.ts`)

**Potential Solution**:
```typescript
class TAM {
  // Register pre-trained bank without training
  registerBank(name: string, bank: GeometricPortBank, config: DomainConfig): void;

  // Or: import from another TAM
  importDomain(name: string, sourceTAM: TAM, sourceName: string): void;
}
```

---

### 4. Registry is Append-Only

**Root Cause**: `PortRegistry.register()` throws on duplicate names, preventing updates or continued training.

```typescript
// From src/composition/registry.ts
if (this.ports.has(name)) {
  throw new Error(`Port "${name}" is already registered`);
}
```

**Impact**: All of the above limitations stem from this design choice.

**Design Question**: Is registration meant to be permanent, or should TAM support dynamic bank updates?

---

## Example-Specific Workarounds Implemented

### Example 05d: Online Adaptation

**Challenge**: Wanted to track learning curves comparing composed vs from-scratch training.

**Workaround**:
```typescript
// Simplified to single training run
async function trainOnce(tam: TAM, domain: DomainSpec, epochs: number) {
  const bank = await tam.learn(domain, { epochs });
  const error = await evaluateError(bank, testStates);
  return { bank, error, portCount: bank.getPortIds().length };
}
```

**What We Lost**: Cannot see learning progress over time, only final result.

### Example 06: Learned Port Selection

**Challenge**: Needed to track convergence of different selection strategies.

**Workaround**: Manually tracked learning curves during training by evaluating every N episodes within the training loop. Required keeping separate TAM instances for each strategy.

---

## Feature Requests for Future API

### 1. Checkpoint Callbacks

```typescript
await tam.learn("domain", spec, {
  epochs: 200,
  checkpointEvery: 20,
  onCheckpoint: async (progress: TrainingProgress) => {
    console.log(`Epoch ${progress.epoch}: Error ${progress.error}`);
    await saveCheckpoint(progress);
    if (shouldEarlyStop(progress)) return "stop";
  },
});
```

### 2. Bank Introspection

```typescript
const bank = tam.getDomainBank("domain");
const stats = bank.getStatistics();
// { portCount, usageCounts, successRates, coneVolumes, ... }
```

### 3. Serialization

```typescript
// Save trained model
const serialized = tam.serialize();
await Bun.write("model.json", JSON.stringify(serialized));

// Load later
const tam2 = TAM.deserialize(serialized);
```

### 4. Bank Import/Export

```typescript
// Export primitives from one TAM
const leftBank = tamPrimitives.exportBank("left-stiff");
const rightBank = tamPrimitives.exportBank("right-soft");

// Import into composition TAM
tamComposed.importBank("left", leftBank);
tamComposed.importBank("right", rightBank);
```

---

## Refinement Policy Issues (Discovered 2025-12-30)

### Issue: Asymmetric Training in CommitmentNet ✅ FIXED

**Problem**: Narrowing and widening use fundamentally different training objectives:

```typescript
// OLD trainNarrow(): Unbounded maximization
minimize(-mean(distance))  // "Make it as big as possible"

// OLD trainWiden(): Supervised regression with explicit target
target = currentDistance - (violation × 0.5)
minimize(MSE(distance, target))
```

**Impact**:
- Unbounded maximization is unstable, especially in high dimensions
- No principled stopping criterion for narrowing
- Asymmetry makes system learn to widen but not narrow effectively
- Leads to distance collapse (experiment 09d shows distance: 0.59 → 0.0005)

**Root Cause**: Fixed thresholds (`narrowThreshold = 0.5`) don't adapt to domain characteristics or dimensionality.

**Fix Applied (2025-12-30)**: Symmetric supervised learning for both directions:
```typescript
// trainNarrow(): Supervised increase
target = currentDistance × (1 + narrowScale);  // e.g., +50%
minimize(MSE(distance, target));

// trainWiden(): Supervised decrease
target = currentDistance - (violation × widenScale);
minimize(MSE(distance, target));
```
Both use explicit targets with configurable scale factors (default 0.5).

---

### Proposed Fix: Calibration-Based Learning ✅ SUPERSEDED BY EMA CONTROL

**Approach**: CommitmentNet should learn to achieve target binding rate, not arbitrary distances.

**Note**: This approach inspired the implemented EMA-based control policy (see below), which achieves the same goal more directly at the policy level rather than in the training objective.

```typescript
// After observing binding outcome:
const currentBindingRate = history.getRecentBindingRate(portId);
const targetRate = this.cfg.equilibriumRate; // e.g., 0.95

// Compute target distance that would achieve desired binding rate
if (currentBindingRate > targetRate) {
  // Binding too much → increase distance (narrow)
  targetDistance = currentDistance × (1 + adjustmentRate);
} else {
  // Binding too little → decrease distance (widen)
  targetDistance = currentDistance × (1 - adjustmentRate);
}

// Symmetric supervised learning!
loss = MSE(predictedDistance, targetDistance);
```

**Benefits**:
- ✅ Symmetric: Both directions use supervised learning
- ✅ Self-calibrating: Learns correct distance for desired binding rate
- ✅ No fixed thresholds: Target is always equilibriumRate (e.g., 95%)
- ✅ Adapts to dimensionality naturally

**Analogy**: Temperature scaling / Platt scaling for geometric commitments. Similar to how probabilistic classifiers calibrate their confidence scores.

---

### Issue: Dimensional Contamination in High-Dimensional Spaces

**Problem**: In mixed predictable/unpredictable systems (e.g., 2D causal spring + 8D noise), binding evaluation averages error across ALL dimensions:

```typescript
// In evaluateBinding()
for (let i = 0; i < dim; i++) {
  sumSq += (diff[i]! / radius[i]!) ** 2;
}
distance = sqrt(sumSq / dim);  // Average across all dims!
margin = 1 - distance;
```

**Impact in 10D with 2 causal + 8 noise dimensions**:
- Causal dims: perfect predictions (error ≈ 0)
- Noise dims: unpredictable (error ≈ radius)
- Total: `distance = sqrt(8/10) ≈ 0.89`
- `margin = 0.11 < narrowThreshold (0.5)` → **NO narrowing occurs**
- Only widening happens → distance collapses → agency → 0%

**Result**: System achieves 100% coverage on causal dims but reports 0% agency due to noise contamination.

---

### Fiber-Based Dimensional Attention (Implementation via Per-Dim Alignment)

**Core Insight**: Port embeddings already encode dimensional structure but we collapse it to scalar via cosine similarity.

**Current (Isotropic)**:
```typescript
// Collapse embedding to scalar
alignment = max(0, cosineSimilarity(port.embedding, state));

// Same radius for all dimensions
radius[i] = aperture × alignment / (1 + distance);  // All radii equal
```

**Proposed (Anisotropic via Per-Dimension Alignment)**:
```typescript
// Per-dimension alignment (no collapse)
alignment[i] = sigmoid(port.embedding[i] × state[i]);

// Different radius per dimension
radius[i] = aperture × alignment[i] / (1 + distance);
```

**Why This Works**:
- Port embedding learns which dimensions it cares about
- High embedding value → high alignment → narrow radius on that dim
- Low embedding value → low alignment → wide radius (ignores that dim)
- **Anisotropic cones** naturally emerge from fibration structure
- Distance (scalar) measures overall commitment, alignment (vector) measures dimensional relevance

**Benefits**:
- ✅ No additional parameters (embedding already exists)
- ✅ Scales to any dimension
- ✅ Philosophically clean: embedding = attention, distance = commitment
- ✅ Naturally implements "fiber-based attention" through existing geometry
- ✅ Binding evaluation already supports anisotropic cones via Mahalanobis distance

**Theoretical Grounding**:
- Heteroscedastic modeling (different variance per dimension)
- Inverse-variance weighting (standard in sensor fusion, Kalman filtering)
- Mahalanobis distance (proper metric for elliptical distributions)

**This IS the fiber-based dimensional attention mechanism** - implemented by properly using the fibration's dimensional structure rather than collapsing it prematurely.

---

### Implemented Fix: EMA-Based Control-Theoretic Refinement (2025-12-30)

**Approach**: Replace threshold-based policy with binding-rate feedback control using exponential moving average (EMA).

**Implementation**:
```typescript
// In BindingHistory.record()
const success = outcome.success ? 1.0 : 0.0;
ph.bindingRateEMA = decay * success + (1 - decay) * ph.bindingRateEMA;

// In BindingRateRefinementPolicy.decide()
const bindingRate = history.getBindingRate(portId);
const error = bindingRate - this.equilibriumRate;

if (error > this.tolerance) {
  return "narrow";  // Binding too much
} else if (error < -this.tolerance) {
  return "widen";   // Binding too little
} else {
  return "noop";    // At equilibrium
}
```

**Benefits**:
- ✅ Removes arbitrary thresholds (no more margin > 0.5)
- ✅ Uses empirical binding rate as feedback signal
- ✅ EMA adapts naturally to domain dynamics
- ✅ Configurable equilibriumRate (e.g., 95%) and tolerance (e.g., ±5%)
- ✅ Decay rate controls effective window size (0.1 ≈ 10 samples)

**Status**: ✅ Implemented in `src/geometric/refinement.ts` as `BindingRateRefinementPolicy`

---

### Future Enhancement: Attention-Weighted Binding Rate

**Idea**: Weight binding rate by agency to focus refinement on high-commitment decisions.

**Motivation**: Currently EMA treats all outcomes equally, but:
- High agency (narrow cone) bindings are precise commitments
- Low agency (wide cone) bindings are exploratory/uncertain
- We should refine based on confident predictions, not noisy ones

**Proposed Implementation**:
```typescript
// In BindingHistory.record()
const success = outcome.success ? 1.0 : 0.0;
const weight = outcome.agency ?? 1.0;  // Use agency as attention weight

// Weighted EMA update
ph.bindingRateEMA =
  (decay * weight * success + (1 - decay) * ph.bindingRateEMA) /
  (decay * weight + (1 - decay));
```

**Benefits**:
- Refinement focuses on high-confidence regions where narrow/widen matters
- Low-agency regions naturally get less influence (exploratory phase)
- Philosophically clean: agency = attention, consistency with fiber geometry

**Status**: 📝 Noted for future implementation

---

## Notes on Discovered Behavior

### Composition with `composeWith`

When using the `composeWith` parameter, TAM creates a `LearnableEncoder` that trains alongside the main domain. This encoder learns to map from raw state to embedding while functors learn to map between domain embeddings.

**Key insight from experiments**: This online encoder learning is actually critical for composition to work - it's not just transfer learning, it's **co-adaptation** of encoders and functors during training in the new domain.

### Functor Discovery Always Succeeds

In all experiments (05a-05d), functor discovery achieved 100% binding rate between left/right domains, even though dynamics were discontinuous. This suggests:
- Functors learn scale/affine transformations well
- Perfect binding doesn't guarantee useful composition
- The real test is performance in unseen regions (gap)

### Online Learning is Essential

- **Frozen approach** (05b): Composition showed mixed results, sometimes worse than individuals
- **Online approach** (05d): Composition consistently improved (8.8% better, 43% fewer ports)

This validates TAM's design philosophy: composition is for **accelerating adaptation**, not frozen reuse.


Just as we learn functors between ports (scale invariance), we should learn functors between dimensions. If a port specializes to dimensions [0,1] and another to [2,3], the relationship between them is a functor - a mapping in the categorical sense.

  Implication:
  - Ports naturally proliferate along dimensional boundaries
  - Dimensional relationships (e.g., "velocity predicts position") are learned functors
  - This is essentially automatic dimensionality reduction through port specialization
