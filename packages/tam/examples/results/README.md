# Experiment Results

This directory stores JSON results from TAM experiments for later analysis and comparison.

## Structure

Each JSON file contains:

```typescript
{
  name: string;              // Experiment name
  config: {                  // Experiment configuration
    domain: {
      name: string;
      embeddingDim: number;
      rawDim?: number;
    };
    episodes: number;
    checkpointEvery?: number;
    testSize?: number;
    horizons?: number[];
    encoder?: { type: "static" | "learnable"; config?: any };
    bankConfig?: any;
    options?: Record<string, any>;
  };
  checkpoints: Array<{       // Training checkpoints
    episode: number;
    train: Partial<Metrics>;
    test: Partial<AggregateMetrics>;
    timestamp: number;
  }>;
  summary: {                 // Summary statistics
    grokked: boolean;
    grokkingEpisode?: number;
    wellCalibrated: boolean;
    finalMetrics: Partial<AggregateMetrics>;
  };
  timestamp: string;         // ISO timestamp of when results were saved
}
```

## Usage

### Saving Results

```typescript
import { runExperiment, saveResultsToJson } from "./harness";

const result = await runExperiment("My Experiment", config);
await saveResultsToJson(result, "examples/results/my-experiment.json");
```

### Loading Results

```typescript
const resultsFile = Bun.file("examples/results/01-basic-training.json");
const results = await resultsFile.json();

console.log(`Experiment: ${results.name}`);
console.log(`Final Error: ${results.summary.finalMetrics.predictionError}`);
console.log(`Grokked: ${results.summary.grokked}`);
```

### Comparing Results

```typescript
// Load multiple result files
const result1 = await Bun.file("examples/results/01-basic-training.json").json();
const result2 = await Bun.file("examples/results/02-multi-horizon.json").json();

// Compare final metrics
console.log(`${result1.name}: ${result1.summary.finalMetrics.predictionError}`);
console.log(`${result2.name}: ${result2.summary.finalMetrics.predictionError}`);
```

## Metrics

### Prediction Error
- Mean L2 distance between predicted and actual trajectories
- Lower is better
- Well-trained models: < 0.05

### Agency
- 1 - normalized cone volume
- Measures prediction specificity
- Range: [0, 1], higher = more confident/specific

### Coverage Rate
- Percentage of actual trajectories that fall within predicted cones
- Measures calibration
- Well-calibrated: > 0.85

### Agency-Error Correlation
- Correlation between agency and prediction accuracy
- Good calibration: negative correlation (high agency = low error)

### Port Count
- Number of specialist ports created
- Indicates degree of specialization

## Notes

- JSON files are gitignored by default
- Bank objects are excluded (not serializable)
- Timestamp is added automatically for tracking
