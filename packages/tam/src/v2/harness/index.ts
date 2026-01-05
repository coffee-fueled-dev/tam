/**
 * TAM v2 Test Harness
 *
 * Utilities for testing, validation, and analytics logging
 */

export { StatsLogger } from "./stats-logger";
export type {
  CheckpointMetrics,
  PortMetrics,
  RunMetadata,
  StatsLoggerConfig,
} from "./stats-logger";

export { evaluate } from "./evaluate";
export type { EvaluationResults } from "./evaluate";

export { generateVisualization } from "./visualize";
export { generateLatentVisualization } from "./visualize-latent";

export { LatentLogger } from "./latent-logger";
export type {
  PortSnapshot,
  LatentCheckpoint,
  LatentLoggerConfig,
} from "./latent-logger";
