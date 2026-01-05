/**
 * Temperature Scaling
 *
 * Post-hoc calibration method for neural networks.
 * Learns a single temperature parameter T that scales predictions:
 *   calibrated_confidence = softmax(logits / T)
 *
 * For regression, we adapt this to scale prediction variance:
 *   calibrated_variance = variance * T²
 *
 * The temperature is learned on a held-out validation set to minimize
 * negative log-likelihood or calibration error.
 */

import type { Vec } from "../types";

export interface CalibrationSample {
  prediction: Vec;
  actual: Vec;
}

/**
 * Learn temperature parameter from calibration samples.
 *
 * For regression, we estimate a scalar variance from errors,
 * then find temperature that minimizes NLL on calibration set.
 */
export function learnTemperature(samples: CalibrationSample[]): number {
  if (samples.length === 0) return 1.0;

  // Estimate variance from squared errors
  let totalError = 0;
  let dims = 0;
  for (const sample of samples) {
    dims = sample.actual.length;
    const error = sample.actual.reduce(
      (sum, actual, i) => sum + Math.pow(actual - sample.prediction[i]!, 2),
      0
    );
    totalError += error;
  }
  const avgError = totalError / samples.length;
  const baseVariance = avgError / dims;

  // Grid search for temperature that minimizes NLL
  let bestTemperature = 1.0;
  let bestNLL = Infinity;

  for (let T = 0.1; T <= 10.0; T += 0.1) {
    const scaledVariance = baseVariance * T * T;

    // Compute NLL with this temperature
    let nll = 0;
    for (const sample of samples) {
      for (let i = 0; i < sample.actual.length; i++) {
        const error = sample.actual[i]! - sample.prediction[i]!;
        const errorSq = error * error;
        // NLL = 0.5 * (log(2π) + log(variance) + error²/variance)
        nll += 0.5 * (Math.log(2 * Math.PI * scaledVariance) + errorSq / scaledVariance);
      }
    }
    nll /= samples.length * dims;

    if (nll < bestNLL) {
      bestNLL = nll;
      bestTemperature = T;
    }
  }

  return bestTemperature;
}

/**
 * Apply temperature scaling to estimate calibrated variance.
 */
export function applyTemperatureScaling(
  error: number,
  temperature: number
): number {
  // Base variance estimate from error
  const baseVariance = error * error;
  // Scale by temperature²
  return baseVariance * temperature * temperature;
}
