/**
 * Agency computation for TAM.
 *
 * Agency measures how specific/committed a prediction is:
 *   agency(Φ) = 1 - |Φ| / |T|
 *
 * Where |Φ| is the "volume" of the prediction region and |T| is the volume
 * of all possible outcomes. High agency = narrow cone = specific commitment.
 *
 * This module provides implementation-agnostic agency computation that works
 * with any representation that provides variance or entropy information.
 */

import type { Vec } from "./vec";
import { clamp, sum } from "./vec";

// ============================================================================
// Configuration
// ============================================================================

/**
 * Configuration for agency computation.
 */
export interface AgencyConfig {
  /** Minimum variance per dimension (prevents infinite agency) */
  minVar: number;
  /** Maximum variance per dimension (prevents zero agency) */
  maxVar: number;
  /** Reference variance for computing relative agency */
  referenceVar: number;
}

export const defaultAgencyConfig: AgencyConfig = {
  minVar: 1e-4,
  maxVar: 100,
  referenceVar: 1.0,
};

// ============================================================================
// Agency from Variance
// ============================================================================

/**
 * Compute agency from variance vector (works for Gaussians and neural outputs).
 *
 * Uses geometric mean of variance ratios:
 *   agency = 1 - (Π(σ²_i / σ²_ref))^(1/d)
 *
 * @param variance - Variance per dimension
 * @param cfg - Agency configuration
 * @returns Agency in [0, 1]
 */
export function agencyFromVariance(
  variance: Vec,
  cfg: AgencyConfig = defaultAgencyConfig
): number {
  const d = variance.length;
  if (d === 0) return 0;

  let logRatio = 0;
  for (const v of variance) {
    const clampedV = clamp(v, cfg.minVar, cfg.maxVar);
    logRatio += Math.log(clampedV / cfg.referenceVar);
  }

  const geoMeanRatio = Math.exp(logRatio / d);
  return clamp(1 - geoMeanRatio, 0, 1);
}

// ============================================================================
// Agency from Entropy (for discrete/categorical outputs)
// ============================================================================

/**
 * Compute agency from probability distribution (for discrete outcomes).
 *
 * Uses normalized entropy:
 *   agency = 1 - H(p) / H_max
 *
 * Where H(p) = -Σ p_i log(p_i) and H_max = log(n).
 *
 * @param probs - Probability distribution (should sum to 1)
 * @returns Agency in [0, 1]
 */
export function agencyFromEntropy(probs: Vec): number {
  if (probs.length <= 1) return 1; // Single outcome = full certainty

  // Compute entropy
  let H = 0;
  for (const p of probs) {
    if (p > 1e-12) {
      H -= p * Math.log(p);
    }
  }

  // Maximum entropy (uniform distribution)
  const maxH = Math.log(probs.length);
  if (maxH < 1e-12) return 1;

  return clamp(1 - H / maxH, 0, 1);
}

// ============================================================================
// Agency Utilities
// ============================================================================

/**
 * Compute weighted average agency across multiple predictions.
 */
export function weightedAgency(agencies: number[], weights: number[]): number {
  if (agencies.length === 0) return 0;

  const totalWeight = sum(weights) || 1;
  let weightedSum = 0;

  for (let i = 0; i < agencies.length; i++) {
    weightedSum += (agencies[i] ?? 0) * (weights[i] ?? 0);
  }

  return weightedSum / totalWeight;
}

/**
 * Check if a point is within n standard deviations of the mean.
 * Works with variance vectors for independent dimensions.
 */
export function isWithinBounds(
  point: Vec,
  mean: Vec,
  variance: Vec,
  nSigma: number = 3
): boolean {
  if (point.length !== mean.length || point.length !== variance.length) {
    return false;
  }

  let mahalanobisSq = 0;
  for (let i = 0; i < point.length; i++) {
    const v = Math.max(variance[i]!, 1e-12);
    const d = point[i]! - mean[i]!;
    mahalanobisSq += (d * d) / v;
  }

  // Chi-squared threshold for d dimensions
  const threshold = nSigma * nSigma * point.length;
  return mahalanobisSq <= threshold;
}

// ============================================================================
// Agency from Cone (for geometric ports)
// ============================================================================

import type { Cone } from "./types";

/**
 * Configuration for cone-based agency computation.
 */
export interface ConeAgencyConfig {
  /** Reference radius for computing relative agency */
  referenceRadius: number;
  /** Minimum radius (prevents infinite agency) */
  minRadius: number;
  /** Maximum radius (prevents zero agency) */
  maxRadius: number;
}

export const defaultConeAgencyConfig: ConeAgencyConfig = {
  referenceRadius: 1.0,
  minRadius: 0.01,
  maxRadius: 10,
};

/**
 * Compute agency from a cone (geometric port architecture).
 *
 * Uses geometric mean of radius ratios:
 *   agency = 1 - (Π(r_i / r_ref))^(1/d)
 *
 * @param cone - The committed tolerance region
 * @param cfg - Agency configuration
 * @returns Agency in [0, 1]
 */
export function agencyFromCone(
  cone: Cone,
  cfg: ConeAgencyConfig = defaultConeAgencyConfig
): number {
  const d = cone.radius.length;
  if (d === 0) return 0;

  let logRatio = 0;
  for (const r of cone.radius) {
    const clampedR = clamp(r, cfg.minRadius, cfg.maxRadius);
    logRatio += Math.log(clampedR / cfg.referenceRadius);
  }

  const geoMeanRatio = Math.exp(logRatio / d);
  return clamp(1 - geoMeanRatio, 0, 1);
}

/**
 * Compute agency from cone log-volume.
 * More efficient when you already have the log-volume.
 *
 * @param logVolume - Sum of log(radius) across dimensions
 * @param dimensions - Number of dimensions
 * @param referenceLogVolume - Reference log-volume (typically d * log(referenceRadius))
 * @returns Agency in [0, 1]
 */
export function agencyFromLogVolume(
  logVolume: number,
  dimensions: number,
  referenceLogVolume: number
): number {
  if (dimensions === 0) return 0;

  const logRatio = (logVolume - referenceLogVolume) / dimensions;
  const geoMeanRatio = Math.exp(logRatio);
  return clamp(1 - geoMeanRatio, 0, 1);
}
