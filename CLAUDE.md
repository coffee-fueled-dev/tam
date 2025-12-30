# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a monorepo for research on ARC-AGI (Abstract Reasoning Corpus) using a novel **Trajectory-Affordance Model (TAM)** framework. The core idea: agents learn world models by predicting outcome distributions p(Δ | situation, action) and refining predictions through commitment-based learning.

## Build System

This project uses **Bun** (not Node.js, npm, or Vite):
- `bun install` - Install dependencies
- `bun run <script>` - Run package scripts
- `bun --hot <file>` - Run with hot reloading
- `bun test` - Run tests (uses bun:test, not Jest/Vitest)
- `bunx <package>` - Run packages (not npx)

Bun automatically loads `.env` files (don't use dotenv).

## Workspace Structure

```
packages/
  tam/          - Core TAM framework (world model learning)
  tkn/          - Token processing utilities
apps/
  sokoban/      - Example app: Sokoban puzzle game
  physics/      - Example app: Physics simulation
tools/
  scripts/      - Build and development scripts
```

## Running Applications

Apps use Bun.serve() with HTML imports:
```bash
cd apps/sokoban
bun run dev           # Development with hot reload
bun run build         # Production build
bun run start         # Production server
```

Apps import TAM via workspace protocol: `"tam": "workspace:*"`

## TAM Package Architecture

**Location:** `packages/tam/`

### Core Concepts

The TAM framework models agent learning through geometric port theory:

1. **Ports** - Model p(Δ | situation, action) as cones in trajectory space
   - Each port predicts outcome distributions for an action
   - Multiple ports per action for multi-modal outcomes

2. **Geometric Architecture** (commitment-based learning)
   - **CausalNet**: Predicts trajectory centers π(p, x) - the fibration map
   - **CommitmentNet**: Predicts tolerances/radii - how specific the commitment is
   - **Port Embeddings**: Learned positions in action manifold
   - **Cones**: Tolerance regions defining acceptable outcomes

3. **Agency** - How specific/committed a prediction is
   - Narrow cone = high agency (confident, specific)
   - Wide cone = low agency (uncertain, exploratory)
   - Computed from cone volume relative to reference

4. **Refinement Cycle**
   - Bind: commit to a cone, check if actual trajectory falls within
   - Narrow: tighten cone on success (increase agency)
   - Widen: loosen cone on failure (decrease agency for safety)
   - Proliferate: spawn specialist ports for inconsistent fibers

5. **Composition & Functors**
   - Learn mappings between domain embeddings
   - Compose world models: learn physics → gravity → bounce
   - Automatic functor discovery with TensorFlow.js networks

### Key Source Files

**Core Types** (`src/types.ts`):
- `Situation`, `Transition`, `Encoders` - Domain abstractions
- `Port`, `PortBank` - Base interfaces
- `Cone`, `GeometricPortId` - Geometric port types
- `BindingOutcome`, `RefinementAction` - Learning mechanics

**Geometric Implementation** (`src/geometric/`):
- `causal.ts` - CausalNet (trajectory prediction)
- `commitment.ts` - CommitmentNet (tolerance prediction)
- `fibration.ts` - Cone assembly and evaluation
- `port.ts` - GeometricPort implementation
- `bank.ts` - GeometricPortBank (manages multiple ports)
- `history.ts` - Binding history and agency tracking
- `refinement.ts` - Refinement policies

**Composition** (`src/composition/`):
- `index.ts` - TAM main API
- `functor.ts` - Functor discovery with TensorFlow
- `encoder.ts` - Learnable encoders
- `graph.ts` - CompositionGraph for multi-step composition
- `registry.ts` - Port registry
- `cache.ts` - Functor caching
- `composed.ts` - ComposedPort implementation

**Utilities**:
- `src/vec.ts` - Vector operations (no external deps)
- `src/agency.ts` - Agency computation from cones

### Examples

**Location:** `packages/tam/examples/`

- `domains/` - Example domains (bounce, gravity, physics, primitives)
- `experiments/` - Research experiments

To run examples:
```bash
cd packages/tam
bun run examples/experiments/<experiment>.ts
```

## TypeScript Configuration

- Module system: ESNext with `"module": "Preserve"`
- Module resolution: bundler mode
- Strict mode enabled with `noUncheckedIndexedAccess`
- JSX: `react-jsx` for React 19
- No emit (Bun handles bundling)

## Working with TAM

### Creating a Port

```typescript
import { GeometricPortBank, type Encoders } from "tam";

// Define encoders for your domain
const encoders: Encoders<MyState> = {
  embedSituation: (sit) => /* embed to vector */,
  delta: (before, after) => /* compute change vector */
};

// Create port bank
const bank = new GeometricPortBank("domain-name", encoders, config);

// Observe transitions
bank.observe({ before, after, action: "move" });

// Get predictions
const predictions = bank.predict("move", situation, topK);
```

### Using Composition API

```typescript
import { TAM } from "tam";

const tam = new TAM();

// Learn domains
await tam.learn("gravity", {
  actions: ["drop"],
  embedder: (state) => /* vector */,
  transitions: gravityData,
  // ... or use composition-learned encoders
});

// Compose domains with functors
await tam.composePorts("gravity", "bounce");

// Use composed port
const composed = tam.composed("gravity", "bounce");
composed.predict("drop", situation);
```

## Key Architectural Patterns

### Fibration Structure
Ports model dynamics as fibrations: π: P × X → T
- P = Port space (embeddings in action manifold)
- X = Situation space (agent's context)
- T = Trajectory space (possible outcomes)

### Alignment & Viewing Distance
Cones scale by geometric factors:
- α(p, x) = alignment (cosine similarity) - "viewing angle"
- d(p, x) = viewing distance from CommitmentNet - "how far to commitment"
- radius = aperture × α / (1 + d)

### Agency-Based Proliferation
Instead of detecting bimodal failures:
1. Track rolling average agency per port
2. If agency < threshold for N samples → fiber is inconsistent
3. Spawn specialist port (not triggered by individual failures)
4. Cooldown period prevents rapid proliferation

### Intra-Domain Encoder Learning (NEW)
**New feature**: Learn encoder within domain via end-to-end backprop through binding objective.

Instead of hand-crafting embedders, learn `E: S_raw → X` such that:
- Binding succeeds: trajectories fall within predicted cones
- Agency is high: cones are narrow (informative predictions)
- Representation is smooth: temporal coherence

**Architecture**:
```
raw_state → IntraDomainEncoder → embedding → CausalNet/CommitmentNet → cone
                                                                          ↓
                                                       binding_loss + agency_loss
                                                                          ↓
                                                               backprop to encoder
```

**Differentiable losses**:
- `L_binding`: Soft distance from trajectory to cone (smooth version of discrete predicate)
- `L_agency`: -log(agency) = log(vol(cone)) (encourages narrow cones)
- `L_smoothness`: ||E(s_t) - E(s_{t-1})||² (temporal coherence)

**Benefits**:
- **No hand-crafted features**: System discovers relevant features automatically
- **Adaptive compression**: Learns minimal representation that preserves predictability
- **Noise robustness**: Learns to ignore irrelevant features
- **Functor inversion**: Same principle as composition-based learning, applied internally

**Configuration**:
```typescript
{
  enableEncoderLearning: true,     // Enable joint encoder training
  encoderRawDim: 10,               // Raw feature dimension
  encoderHiddenSizes: [32, 16],    // Encoder network architecture
  encoderLearningRate: 0.001,      // Lower than port networks
  encoderAgencyWeight: 0.1,        // Encourage narrow cones
  encoderSmoothnessWeight: 0.05,   // Temporal coherence
}
```

**Usage with EncoderBridge**:
```typescript
const learnableEncoder = new IntraDomainEncoder({
  rawDim: 10,
  embeddingDim: 4,
});

const bridge = createEncoderBridge({
  extractRaw: (state) => [state.x, state.y, ...allFeatures],
  learnableEncoder,
});

const bank = new GeometricPortBank(bridge.encoders, config);
// Encoder learns during training via joint optimization
```

**Experiment**: See `examples/experiments/intra-encoder.ts`

**Status**: Architecture implemented, full integration pending

### Port Functor Discovery (Intra-Domain Composition)
**New feature**: Ports within a domain share a learned manifold, enabling systematic transforms:

**Hybrid proliferation approach**:
1. When proliferation triggers, collect recent failure cases
2. If failures are structured (bimodal clusters), try discovering functor F: P → P
3. New port = F(parent_embedding) inherits structure systematically
4. Falls back to random perturbation if functor discovery fails

**Benefits**:
- **Structured specialization**: Ports learn systematic relationships (e.g., left-half ↔ right-half)
- **Knowledge reuse**: New ports bootstrap from parents via learned transforms
- **Interpretability**: Port relationships are explicit (not random exploration)
- **Efficiency**: Fewer ports needed when transforms are discovered

**Use cases**:
- Spatial symmetries (mirror, rotation, translation)
- Scale invariance (3×3 grids → 5×5 grids)
- Continuous parameter sweeps (slow velocity → fast velocity)

**Configuration**:
```typescript
{
  enablePortFunctors: true,        // Enable functor discovery (default: false)
  portFunctorTolerance: 0.3,       // RMSE threshold for accepting functor
  portFunctorMaxEpochs: 50,        // Training budget (small, in learned space)
}
```

**Experiment**: See `examples/experiments/port-functors.ts` for validation

### Fiber Consistency Regularization
Training penalizes when similar embeddings have different dynamics:
- Groups embeddings within `fiberThreshold` distance
- Adds consistency loss to causal/commitment training
- Weight controlled by `fiberConsistencyWeight`

## Dependencies

**Core:**
- Bun runtime
- TypeScript 5+
- React 19 (for apps)
- TensorFlow.js (for TAM neural networks)

**Packages:**
- `tam` - Uses @tensorflow/tfjs
- `tkn` - Uses zod for validation
- Apps depend on `tam` via workspace

## Development Notes

- Source files are used directly (no build step for packages)
- Apps bundle with `bun build` for production
- Hot reload available with `bun --hot`
- Frontend uses HTML imports with `<script type="module">`
- WebSocket support available via Bun.serve() websocket option

## Geometric Port Configuration

Key settings in `defaultGeometricPortConfig`:
- `embeddingDim`: Port embedding dimensionality (default: 8)
- `defaultAperture`: Base cone aperture (default: 1.0)
- `equilibriumRate`: Success rate to stop narrowing (default: 0.95)

**Proliferation settings**:
- `enableProliferation`: Create specialist ports (default: false)
- `proliferationAgencyThreshold`: Agency threshold for proliferation (default: 0.3)
- `proliferationMinSamples`: Minimum samples before proliferation (default: 20)
- `proliferationCooldown`: Samples between proliferations (default: 50)

**Port functor discovery** (new):
- `enablePortFunctors`: Learn systematic transforms (default: false)
- `portFunctorTolerance`: RMSE threshold for accepting functor (default: 0.3)
- `portFunctorMaxEpochs`: Training epochs for discovery (default: 50)

**Fiber consistency**:
- `fiberConsistencyWeight`: Fiber regularization strength (default: 0.1)
- `fiberThreshold`: Distance for same fiber (default: 0.1)

## Common Patterns

### Hand-Crafted Encoders
When state structure is known, write explicit embedders:
```typescript
embedSituation: (sit) => [sit.state.x, sit.state.y, sit.state.vx, sit.state.vy]
```

### Composition-Learned Encoders
For complex domains, learn embeddings through composition:
```typescript
await tam.learn("complex", {
  stateToRaw: (s) => rawFeatures,
  rawDim: 64,
  composeWith: ["primitives", "physics"], // Learn from composed embeddings
  // ...
});
```

### Multi-Modal Predictions
Enable proliferation for domains with distinct outcome modes:
```typescript
config: {
  enableProliferation: true,
  proliferationAgencyThreshold: 0.3,
  proliferationMinSamples: 20,
}
```

## Testing Approach

Use `bun:test` for testing:
```typescript
import { test, expect } from "bun:test";

test("port learns dynamics", () => {
  // test implementation
});
```

Run with: `bun test`
