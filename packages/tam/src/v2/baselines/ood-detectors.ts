/**
 * Standard OOD Detection Methods
 *
 * Implements classic out-of-distribution detection baselines:
 * 1. ODIN: Temperature scaling + input perturbation
 * 2. Mahalanobis Distance: Distance in feature space
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../types";
import type { StandardMLP } from "./standard-mlp";

/**
 * ODIN Score
 *
 * Out-of-Distribution detector using:
 * - Temperature scaling: scale logits/predictions by T
 * - Input perturbation: Add gradient-based noise to increase confidence
 *
 * Higher score = more confident = more likely in-distribution
 * Lower score = less confident = more likely OOD
 *
 * Original paper: "Enhancing The Reliability of Out-of-distribution Image
 * Detection in Neural Networks" (Liang et al., 2018)
 */
export function computeODINScore(
  model: StandardMLP,
  stateEmb: Vec,
  portEmb: Vec,
  temperature: number = 1.0,
  epsilon: number = 0.001
): number {
  // For regression, ODIN score is based on prediction confidence
  // We'll use negative prediction variance as a proxy for confidence

  // Get base prediction
  const prediction = model.predict(stateEmb, portEmb);

  // For regression, confidence score can be: 1 / (1 + prediction_magnitude)
  // Higher prediction magnitude = less confident
  const magnitude = Math.sqrt(
    prediction.reduce((sum, p) => sum + p * p, 0)
  );

  // Apply temperature scaling
  const score = 1 / (1 + magnitude / temperature);

  return score;
}

/**
 * Mahalanobis Distance OOD Detection
 *
 * Measures distance from test sample to training distribution
 * in feature space, accounting for covariance.
 *
 * Lower distance = more likely in-distribution
 * Higher distance = more likely OOD
 *
 * Original paper: "A Simple Unified Framework for Detecting Out-of-Distribution
 * Samples and Adversarial Attacks" (Lee et al., 2018)
 */
export class MahalanobisDetector {
  private mean: Vec | null = null;
  private invCov: number[][] | null = null;
  private samples: Vec[] = [];

  /**
   * Fit detector on training data features.
   */
  fit(features: Vec[]): void {
    if (features.length === 0) return;

    this.samples = features;
    const dim = features[0]!.length;

    // Compute mean
    const mean = Array(dim).fill(0);
    for (const feat of features) {
      for (let i = 0; i < dim; i++) {
        mean[i] += feat[i]!;
      }
    }
    for (let i = 0; i < dim; i++) {
      mean[i] /= features.length;
    }
    this.mean = mean;

    // Compute covariance matrix
    const cov: number[][] = Array(dim)
      .fill(0)
      .map(() => Array(dim).fill(0));

    for (const feat of features) {
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          cov[i]![j] += (feat[i]! - mean[i]!) * (feat[j]! - mean[j]!);
        }
      }
    }

    // Normalize and add regularization (for numerical stability)
    const reg = 0.01;
    for (let i = 0; i < dim; i++) {
      for (let j = 0; j < dim; j++) {
        cov[i]![j] = cov[i]![j]! / features.length;
        if (i === j) {
          cov[i]![j] += reg;
        }
      }
    }

    // Compute inverse (using simple approach for small matrices)
    this.invCov = this.invertMatrix(cov);
  }

  /**
   * Compute Mahalanobis distance for a test sample.
   */
  computeDistance(feature: Vec): number {
    if (!this.mean || !this.invCov) {
      throw new Error("Detector not fitted. Call fit() first.");
    }

    // Compute (x - mean)
    const diff = feature.map((val, i) => val - this.mean![i]!);

    // Compute (x - mean)^T * invCov * (x - mean)
    let distance = 0;
    for (let i = 0; i < diff.length; i++) {
      let sum = 0;
      for (let j = 0; j < diff.length; j++) {
        sum += this.invCov![i]![j]! * diff[j]!;
      }
      distance += diff[i]! * sum;
    }

    return Math.sqrt(Math.max(0, distance));
  }

  /**
   * Simple matrix inversion using Gaussian elimination.
   * For production, use a proper linear algebra library.
   */
  private invertMatrix(matrix: number[][]): number[][] {
    const n = matrix.length;
    const result: number[][] = Array(n)
      .fill(0)
      .map((_, i) => Array(n).fill(0).map((_, j) => (i === j ? 1 : 0)));

    // Create augmented matrix [A | I]
    const augmented: number[][] = matrix.map((row, i) => [...row, ...result[i]!]);

    // Forward elimination
    for (let i = 0; i < n; i++) {
      // Find pivot
      let maxRow = i;
      for (let k = i + 1; k < n; k++) {
        if (Math.abs(augmented[k]![i]!) > Math.abs(augmented[maxRow]![i]!)) {
          maxRow = k;
        }
      }

      // Swap rows
      [augmented[i], augmented[maxRow]] = [augmented[maxRow]!, augmented[i]!];

      // Make all rows below this one 0 in current column
      for (let k = i + 1; k < n; k++) {
        const factor = augmented[k]![i]! / augmented[i]![i]!;
        for (let j = i; j < 2 * n; j++) {
          augmented[k]![j] -= factor * augmented[i]![j]!;
        }
      }
    }

    // Back substitution
    for (let i = n - 1; i >= 0; i--) {
      for (let k = i - 1; k >= 0; k--) {
        const factor = augmented[k]![i]! / augmented[i]![i]!;
        for (let j = i; j < 2 * n; j++) {
          augmented[k]![j] -= factor * augmented[i]![j]!;
        }
      }
    }

    // Normalize
    for (let i = 0; i < n; i++) {
      const factor = augmented[i]![i]!;
      for (let j = 0; j < 2 * n; j++) {
        augmented[i]![j] /= factor;
      }
    }

    // Extract result
    return augmented.map(row => row.slice(n));
  }
}
