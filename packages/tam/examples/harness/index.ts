/**
 * Experiment Harness
 *
 * Standardized infrastructure for TAM experiments:
 * - Domain definitions
 * - Training & evaluation loops
 * - Metric tracking & analysis
 * - Experiment comparison
 *
 * Usage:
 * ```typescript
 * import { runExperiment, dampedSpring1D } from "./harness";
 *
 * const result = await runExperiment("My Experiment", {
 *   domain: dampedSpring1D,
 *   episodes: 500,
 * });
 * ```
 */

// Core types
export type {
  Domain,
  Metrics,
  AggregateMetrics,
  Checkpoint,
  ExperimentConfig,
  ExperimentResult,
  TrainingCallback,
  EvaluationCallback,
} from "./types";

// Runner
export { runExperiment, compareExperiments } from "./runner";

// Standard domains
export {
  dampedSpring1D,
  dampedSpringVariant,
  driftModes,
  noisyPendulum,
  createMultiHorizonDomain,
  type DriftMode,
  type DriftState,
  type PendulumState,
} from "./domains";
