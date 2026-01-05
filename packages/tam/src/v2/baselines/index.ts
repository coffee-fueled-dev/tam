/**
 * Baseline Methods for Comparison
 *
 * Standard uncertainty and OOD detection methods from ML literature.
 */

export { StandardMLP } from "./standard-mlp";
export type { StandardMLPConfig } from "./standard-mlp";

export {
  learnTemperature,
  applyTemperatureScaling,
  type CalibrationSample,
} from "./temperature-scaling";

export {
  computeODINScore,
  MahalanobisDetector,
} from "./ood-detectors";
