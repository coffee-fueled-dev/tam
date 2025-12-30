/**
 * Fibration: Aligns the causal and commitment manifolds.
 *
 * In the geometric TAM architecture, the fibration is the correspondence
 * between trajectory predictions (CausalNet) and tolerance regions (CommitmentNet).
 *
 * For our current implementation, the fibration is IMPLICIT:
 * - Both networks are conditioned on the same (state_emb, port_emb) input
 * - CausalNet outputs the cone center (predicted trajectory)
 * - CommitmentNet outputs the cone radius (tolerance)
 * - The cone is assembled from both outputs
 *
 * This module provides utilities for:
 * - Assembling cones from the two networks
 * - Evaluating the binding predicate (trajectory ∈ cone?)
 * - Computing binding outcome with margin/violation
 * - Ellipsoidal cones with full covariance for high-dimensional spaces
 */

import type { Vec } from "../vec";
import type { Cone, BindingOutcome } from "../types";

/**
 * Assemble a cone from CausalNet and CommitmentNet outputs.
 */
export function assembleCone(center: Vec, radius: Vec): Cone {
  return { center, radius };
}

/**
 * Assemble an ellipsoidal cone with covariance.
 * The covariance is stored as a flattened lower-triangular Cholesky factor.
 */
export function assembleConeWithCovariance(
  center: Vec,
  choleskyLower: Vec
): Cone {
  const dim = center.length;
  // Extract diagonal of Cholesky as "effective radius" for backward compatibility
  const radius = extractDiagonalFromCholesky(choleskyLower, dim);
  return { center, radius, covariance: choleskyLower };
}

/**
 * Extract diagonal elements from flattened lower-triangular Cholesky factor.
 */
function extractDiagonalFromCholesky(cholesky: Vec, dim: number): Vec {
  const diag: Vec = [];
  let idx = 0;
  for (let i = 0; i < dim; i++) {
    // In lower-triangular, diagonal of row i is at position idx + i
    diag.push(Math.abs(cholesky[idx + i] ?? 1));
    idx += i + 1;
  }
  return diag;
}

/**
 * Evaluate the binding predicate: is trajectory inside the cone?
 *
 * Supports two modes:
 * 1. Isotropic: Uses per-dimension radius (diagonal covariance)
 * 2. Ellipsoidal: Uses full covariance matrix (if cone.covariance is present)
 *
 * Distance is normalized such that:
 *   distance ≤ 1 → inside cone
 *   distance > 1 → outside cone
 */
export function evaluateBinding(trajectory: Vec, cone: Cone): BindingOutcome {
  if (trajectory.length !== cone.center.length) {
    // Dimension mismatch - treat as failure
    return { success: false, violation: Infinity };
  }

  const dim = trajectory.length;
  const diff: Vec = [];
  for (let i = 0; i < dim; i++) {
    diff.push(trajectory[i]! - cone.center[i]!);
  }

  let distance: number;

  if (cone.precision) {
    // Use precision matrix directly (most efficient)
    distance = Math.sqrt(mahalanobisDistanceWithPrecision(diff, cone.precision, dim) / dim);
  } else if (cone.covariance) {
    // Use Cholesky factor to compute Mahalanobis distance
    distance = Math.sqrt(mahalanobisDistanceWithCholesky(diff, cone.covariance, dim) / dim);
  } else {
    // Fallback to diagonal (isotropic) version
    let sumSq = 0;
    for (let i = 0; i < dim; i++) {
      const r = Math.max(cone.radius[i]!, 1e-8);
      sumSq += (diff[i]! / r) ** 2;
    }
    distance = Math.sqrt(sumSq / dim);
  }

  if (distance <= 1) {
    // Inside cone: margin is how far inside (1 = at center, 0 = at boundary)
    return { success: true, margin: 1 - distance };
  } else {
    // Outside cone: violation is how far outside
    return { success: false, violation: distance - 1 };
  }
}

/**
 * Compute Mahalanobis distance using Cholesky factor.
 * d² = x' Σ⁻¹ x = (L⁻¹ x)' (L⁻¹ x) where Σ = L L'
 */
function mahalanobisDistanceWithCholesky(
  diff: Vec,
  choleskyLower: Vec,
  dim: number
): number {
  // Solve L y = diff for y using forward substitution
  const y = solveCholeskyLower(diff, choleskyLower, dim);

  // d² = y' y
  let sumSq = 0;
  for (let i = 0; i < dim; i++) {
    sumSq += y[i]! ** 2;
  }
  return sumSq;
}

/**
 * Compute Mahalanobis distance using precision matrix directly.
 * d² = x' P x where P = Σ⁻¹
 */
function mahalanobisDistanceWithPrecision(
  diff: Vec,
  precision: Vec,
  dim: number
): number {
  // Precision matrix is stored as flattened row-major
  let sumSq = 0;
  for (let i = 0; i < dim; i++) {
    let inner = 0;
    for (let j = 0; j < dim; j++) {
      inner += precision[i * dim + j]! * diff[j]!;
    }
    sumSq += diff[i]! * inner;
  }
  return sumSq;
}

/**
 * Forward substitution to solve L y = b for y.
 * L is lower-triangular, stored as flattened lower-triangular.
 */
function solveCholeskyLower(b: Vec, L: Vec, dim: number): Vec {
  const y: Vec = new Array(dim).fill(0);
  let idx = 0;

  for (let i = 0; i < dim; i++) {
    let sum = b[i]!;
    for (let j = 0; j < i; j++) {
      sum -= L[idx + j]! * y[j]!;
    }
    const diag = Math.max(L[idx + i]!, 1e-8);
    y[i] = sum / diag;
    idx += i + 1;
  }

  return y;
}

/**
 * Check if trajectory is in the "narrow-safe" zone of the cone.
 * Returns true if trajectory is well inside (not near boundary).
 */
export function isInNarrowZone(
  trajectory: Vec,
  cone: Cone,
  threshold: number
): boolean {
  const outcome = evaluateBinding(trajectory, cone);
  return outcome.success && outcome.margin > threshold;
}

/**
 * Compute the volume of a cone (product of radii).
 * Used for agency computation.
 *
 * For ellipsoidal cones: volume ∝ det(Σ)^(1/2) = det(L)
 */
export function coneVolume(cone: Cone): number {
  if (cone.covariance) {
    // Volume is proportional to det(L) = product of diagonal elements
    const dim = cone.center.length;
    return choleskyDeterminant(cone.covariance, dim);
  }
  return cone.radius.reduce((acc, r) => acc * r, 1);
}

/**
 * Compute the log-volume of a cone (sum of log-radii).
 * More numerically stable for high dimensions.
 *
 * For ellipsoidal cones: log-volume = sum of log(diagonal of L)
 */
export function coneLogVolume(cone: Cone): number {
  if (cone.covariance) {
    const dim = cone.center.length;
    return choleskyLogDeterminant(cone.covariance, dim);
  }
  return cone.radius.reduce((acc, r) => acc + Math.log(Math.max(r, 1e-8)), 0);
}

/**
 * Compute determinant from Cholesky factor (product of diagonal elements).
 */
function choleskyDeterminant(choleskyLower: Vec, dim: number): number {
  let det = 1;
  let idx = 0;
  for (let i = 0; i < dim; i++) {
    det *= Math.abs(choleskyLower[idx + i] ?? 1);
    idx += i + 1;
  }
  return det;
}

/**
 * Compute log-determinant from Cholesky factor (sum of log diagonal elements).
 */
function choleskyLogDeterminant(choleskyLower: Vec, dim: number): number {
  let logDet = 0;
  let idx = 0;
  for (let i = 0; i < dim; i++) {
    logDet += Math.log(Math.max(Math.abs(choleskyLower[idx + i] ?? 1), 1e-8));
    idx += i + 1;
  }
  return logDet;
}

