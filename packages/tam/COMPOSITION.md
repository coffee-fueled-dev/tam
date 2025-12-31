# TAM Composition API

The composition API enables building modular world models that can be composed, reused, and adapted across tasks.

## Core Concepts

### Domains
A **domain** is a world model specialist trained on specific dynamics:
- Has its own state space and dynamics
- Trained via `TAM.learn()`
- Represented internally as a `GeometricPortBank`

### Functors
A **functor** is a learned mapping between domain embeddings:
- Discovered automatically via gradient descent
- Maps states from source domain → target domain coordinate frames
- Quality measured by binding rate (how well predictions work)

### Composition Paths
A **path** chains functors to enable multi-hop inference:
- Direct: `A → B` (single functor)
- Multi-hop: `A → B → C` (chained functors)
- Quality: Product of binding rates + commutativity score

### Composed Ports
A **ComposedPort** uses functors to:
- Make predictions in target domain from source domain states
- Continue learning via `.observe()` (online adaptation)
- Bootstrap knowledge from primitives into novel regions

## Basic Usage

### 1. Train Domain Specialists

```typescript
import { TAM } from "./src/composition";

const tam = new TAM({
  maxEpochs: 200,
  successThreshold: 0.8,
  learningRate: 0.01,
});

// Train left well specialist
await tam.learn("left", {
  randomState: () => randomState(-2, -0.5),
  simulate: (s) => leftWellDynamics(s),
  embedder: (s) => [s.x, s.v],
  embeddingDim: 2,
});

// Train right well specialist
await tam.learn("right", {
  randomState: () => randomState(0.5, 2),
  simulate: (s) => rightWellDynamics(s),
  embedder: (s) => [s.x, s.v],
  embeddingDim: 2,
});
```

### 2. Discover Functors & Compose

```typescript
// Find composition path (triggers functor discovery)
const path = await tam.findPath("left", "right");

if (path) {
  console.log(`Found path: ${path.describe()}`);
  console.log(`Binding rate: ${path.totalBindingRate}`);

  // Create composed port for inference + learning
  const composed = tam.compose(path);
}
```

### 3. Use for Inference

```typescript
// Predict in target domain from source domain state
const prediction = composed.predict(leftWellState);
console.log(`Delta: ${prediction.delta}`);
console.log(`Confidence: ${prediction.composedBindingRate}`);
```

### 4. Adapt Online (NEW)

```typescript
// Continue learning in novel regions
const gapState = { x: 0.0, v: 0.5 };  // In the gap between wells
const nextState = gapDynamics(gapState);

composed.observe({
  before: gapState,
  after: nextState,
});

// Composed model now adapts to gap dynamics while retaining primitive knowledge
```

## Advanced Features

### Commutativity Testing

Test whether transformations compose cleanly:

```typescript
const path1 = await tam.findPath("rotate", "translate");
const path2 = await tam.findPath("translate", "rotate");

const commScore = await graph.checkCommutativity(path1, path2);
console.log(`Commutativity: ${(commScore * 100).toFixed(1)}%`);
// High score = transformations are orthogonal/compositional
```

### Multi-Hop Composition

```typescript
// Try up to 3-hop paths
const path = await tam.findPath("A", "C", maxHops: 3);
// Might find: A → B → C
```

### Retry Failed Discovery

```typescript
// First attempt failed, try with more budget
await tam.retryDiscovery("source", "target", {
  maxEpochs: 500,
  patience: 100,
});
```

## Use Cases

### 1. Discontinuous Composition (Bootstrap + Adapt)
**Problem**: Novel region has different dynamics than primitives

**Solution**:
- Train primitives on known regions
- Compose functors to bootstrap into novel region
- Use `.observe()` to adapt online

**Example**: See `examples/05-composition.ts`

### 2. Zero-Shot Transfer
**Problem**: Need predictions in domain B, only trained on domain A

**Solution**:
- Discover functor from A → B
- Use composed port for inference (no online learning needed)

### 3. Logical Locality Testing
**Problem**: Do transformations compose cleanly?

**Solution**:
- Test commutativity of different paths
- High scores indicate true compositional structure

### 4. Modular Architecture Discovery
**Problem**: Find orthogonal dimensions/factors

**Solution**:
- Train specialists on hypothesized factors
- Test functor success rates
- High commutativity = good decomposition

## API Reference

### TAM

**`constructor(config?: Partial<FunctorDiscoveryConfig>)`**
Create TAM instance with functor discovery settings.

**`learn(name, spec, training?): Promise<GeometricPortBank>`**
Train a domain specialist and register it.

**`findPath(source, target, maxHops?): Promise<CompositionPath | null>`**
Find composition path (triggers functor discovery if needed).

**`compose(path): ComposedPort`**
Create a ComposedPort from a path.

**`getComposedPort(source, target, maxHops?): Promise<ComposedPort | null>`**
Convenience: find path and compose in one step.

### ComposedPort

**`predict(sourceState): ComposedPrediction`**
Predict in target domain from source state.

**`observe(transition): void`** *(NEW)*
Train on a transition, adapting the target port online.

**`describe(): string`**
Human-readable path description.

**Properties:**
- `totalBindingRate`: Combined quality score
- `hopCount`: Number of functors in chain
- `isDirect`: Whether this is a single-hop composition

## Limitations & Future Work

**Current Limitations:**
1. `.observe()` only trains target port (doesn't update functors)
2. No multi-task batching across domains
3. Functor discovery is expensive (gradient descent on neural nets)
4. No automatic proliferation across composed ports

**Possible Extensions:**
1. Backprop through functors during `.observe()`
2. Shared embedding spaces across domains
3. Meta-learning for faster functor discovery
4. Hierarchical composition (compose composed ports)

## Configuration

### FunctorDiscoveryConfig

```typescript
{
  maxEpochs: 200,           // Budget for discovery
  successThreshold: 0.8,    // Binding rate to declare "found"
  patience: 50,             // Early stopping
  minEpochs: 20,            // Minimum before stopping
  hiddenSizes: [32, 32],    // Functor network architecture
  learningRate: 0.01,       // Optimizer learning rate
  samplesPerEpoch: 50,      // Training samples per epoch
}
```

### TrainingConfig

```typescript
{
  epochs: 100,              // Domain training epochs
  samplesPerEpoch: 50,      // Samples per epoch
  flushFrequency: 20,       // How often to train networks
}
```
