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
 * Evaluate the binding predicate: is trajectory inside the cone?
 *
 * Uses per-dimension comparison with ellipsoid distance:
 *   distance = sqrt(Σ ((τ_i - c_i) / r_i)²)
 *
 * If distance ≤ 1, trajectory is inside the cone.
 */
export function evaluateBinding(trajectory: Vec, cone: Cone): BindingOutcome {
  if (trajectory.length !== cone.center.length) {
    // Dimension mismatch - treat as failure
    return { success: false, violation: Infinity };
  }

  // Compute normalized distance (Mahalanobis-like with diagonal covariance)
  let sumSq = 0;
  for (let i = 0; i < trajectory.length; i++) {
    const diff = trajectory[i]! - cone.center[i]!;
    const r = Math.max(cone.radius[i]!, 1e-8);
    sumSq += (diff / r) ** 2;
  }

  const distance = Math.sqrt(sumSq / trajectory.length); // Normalized by dimensions

  if (distance <= 1) {
    // Inside cone: margin is how far inside (1 = at center, 0 = at boundary)
    return { success: true, margin: 1 - distance };
  } else {
    // Outside cone: violation is how far outside
    return { success: false, violation: distance - 1 };
  }
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
 */
export function coneVolume(cone: Cone): number {
  return cone.radius.reduce((acc, r) => acc * r, 1);
}

/**
 * Compute the log-volume of a cone (sum of log-radii).
 * More numerically stable for high dimensions.
 */
export function coneLogVolume(cone: Cone): number {
  return cone.radius.reduce((acc, r) => acc + Math.log(Math.max(r, 1e-8)), 0);
}

