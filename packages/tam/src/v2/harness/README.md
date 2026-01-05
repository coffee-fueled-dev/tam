# TAM v2 Test Harness

Utilities for testing, validation, and analytics logging for TAM v2.

## Components

### StatsLogger

Manages structured logging for training runs. Creates a directory structure with:
- `metadata.json` - Run configuration and timestamp
- `checkpoints.jsonl` - Global metrics time series (avgAgency, avgError, testBindingRate, portCount)
- `ports/port-N.jsonl` - Per-port metrics time series

**Usage:**

```typescript
import { StatsLogger } from "tam/v2";

const logger = await StatsLogger.create({
  experiment: "My Experiment",
  config: { /* your config */ },
});

// Log checkpoint metrics
logger.logCheckpoint({
  sample: 100,
  portCount: 3,
  avgAgency: 0.45,
  avgError: 0.12,
  testBindingRate: 0.95,
});

// Log per-port metrics
logger.logPort({
  portIdx: 0,
  sample: 100,
  samples: 50,
  bindings: 45,
  bindingRate: 0.9,
  avgAgency: 0.42,
  agencyStdDev: 0.027,
  avgError: 0.072,
  errorStdDev: 0.035,
});

// Close when done
logger.close();
```

### evaluate()

Evaluates actor performance on test samples.

**Usage:**

```typescript
import { evaluate } from "tam/v2";

const results = evaluate(
  actor,
  () => {
    const before = generateState();
    const after = stepForward(before);
    return { before, after };
  },
  embedState,
  50 // number of test samples
);

console.log(results.avgAgency);
console.log(results.avgError);
console.log(results.testBindingRate);
```

### generateVisualization()

Generates an interactive HTML visualization of training metrics (error, agency, binding rate) over time.

**Usage:**

```typescript
import { generateVisualization } from "tam/v2";

await generateVisualization("runs/my-run-1234567890");
// Creates: runs/my-run-1234567890/visualization.html
```

**CLI Usage:**

```bash
bun src/v2/harness/visualize.ts runs/my-run-1234567890
```

Opens in your browser to view three stacked charts with aligned X-axes:
- **Error** (top chart, red line) - Shows prediction error over time
- **Agency** (middle chart, blue line) - Shows agency growth as percentage over time
- **Binding Rate** (bottom chart, green line) - Shows binding success rate as percentage over time
- Each chart has its own properly scaled Y-axis
- Interactive tooltips showing metrics at each sample point

### Latent Space Visualization

View how port embeddings and commitment distances evolve over time.

**Usage:**

```typescript
import { LatentLogger } from "tam/v2";

const latentLogger = LatentLogger.create(
  runDir,
  generateRandomState,  // Function to generate representative states
  embedState,
  { numStates: 10 }  // Number of test states to track
);

// At each checkpoint
latentLogger.logCheckpoint(sample, actor);

// Close when done
latentLogger.close();
```

**Visualization:** Open `visualize-latent.html` and load `latent.jsonl` to see:
- **Port Embeddings**: 2D scatter plot showing port positions in state space
  - Circle sizes represent average commitment distance (cone volume)
  - Colors distinguish different ports
- **Commitment Heatmap**: Shows commitment distances for each port at each test state
  - Brighter colors = larger commitment distances
  - Bold borders indicate which port was selected
- **Timeline Slider**: Scrub through training to see how latent space evolves

This helps understand:
- How ports specialize to different regions of state space
- Which states activate which ports
- How commitment distances (cone volumes) change over training

## Complete Example

See `examples/v2-test.ts` for a complete example that uses all harness utilities together.
