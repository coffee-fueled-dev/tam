/**
 * RefinementPolicy: Decides how to update ports after binding.
 *
 * Fiber-based policy (geometry-grounded):
 * - SUCCESS + near center → NARROW (increase agency)
 * - SUCCESS + near boundary → NOOP (at equilibrium)
 * - FAILURE + low average agency → PROLIFERATE (fiber is inconsistent)
 * - FAILURE + otherwise → WIDEN (cone too tight)
 *
 * Proliferation signal: A port with consistently low agency has an
 * inconsistent fiber - states in its domain behave too differently
 * for a single cone to cover them well. Splitting the port into
 * specialists allows each to have a tighter cone.
 *
 * Future: The decision itself could be a learned policy:
 * minimize prediction error + λ₁(cone volume) + λ₂(port count)
 */

import type {
  BindingOutcome,
  BindingHistory,
  RefinementAction,
  RefinementPolicy,
  GeometricPortConfig,
} from "../types";
import { defaultGeometricPortConfig } from "../types";

export class FixedRefinementPolicy implements RefinementPolicy {
  private readonly narrowThreshold: number;
  private readonly enableProliferation: boolean;
  private readonly proliferationAgencyThreshold: number;
  private readonly proliferationMinSamples: number;
  private readonly proliferationCooldown: number;

  constructor(config?: Partial<GeometricPortConfig>) {
    const cfg = { ...defaultGeometricPortConfig, ...config };
    this.narrowThreshold = cfg.commitment.narrowThreshold;
    this.enableProliferation = cfg.enableProliferation;
    this.proliferationAgencyThreshold = cfg.proliferationAgencyThreshold;
    this.proliferationMinSamples = cfg.proliferationMinSamples;
    this.proliferationCooldown = cfg.proliferationCooldown;
  }

  decide(
    outcome: BindingOutcome,
    portId: string,
    _situationKey: string,
    history: BindingHistory
  ): RefinementAction {
    if (outcome.success) {
      // Binding succeeded - trajectory was inside cone
      const margin = outcome.margin;

      // Narrow if trajectory was well inside cone (room to tighten)
      // The margin indicates how far inside: 1 = at center, 0 = at boundary
      if (margin > this.narrowThreshold) {
        // Trajectory was well inside cone - safe to narrow (increase agency)
        return "narrow";
      } else {
        // Near boundary - don't over-tighten, stay at current level
        return "noop";
      }
    } else {
      // Binding failed - trajectory was outside cone

      // Only consider proliferation if enabled
      if (this.enableProliferation) {
        // Fiber-based proliferation: if port has consistently low agency,
        // its fiber is inconsistent and should be split
        const samples = history.getSampleCount(portId);
        const avgAgency = history.getAverageAgency(portId);
        const inCooldown = history.isInCooldown(
          portId,
          this.proliferationCooldown
        );

        if (
          samples >= this.proliferationMinSamples &&
          avgAgency < this.proliferationAgencyThreshold &&
          !inCooldown
        ) {
          // Fiber is inconsistent (low agency = wide cones = varied dynamics)
          // Proliferate to create specialists for distinct behaviors
          return "proliferate";
        }
      }

      // Default: widen the cone to accommodate the unexpected trajectory
      return "widen";
    }
  }
}

/**
 * BindingRateRefinementPolicy: Control-theoretic refinement using binding rate feedback.
 *
 * Instead of fixed thresholds (e.g., margin > 0.5), uses empirical binding rate
 * to maintain equilibrium through feedback control:
 *
 * - If binding rate > equilibrium + tolerance → NARROW (binding too much)
 * - If binding rate < equilibrium - tolerance → WIDEN (binding too little)
 * - Otherwise → NOOP (at equilibrium)
 *
 * This removes arbitrary thresholds and adapts naturally to the environment.
 * Binding rate is tracked via exponential moving average (EMA) in BindingHistory.
 */
export class BindingRateRefinementPolicy implements RefinementPolicy {
  private readonly equilibriumRate: number;
  private readonly tolerance: number;
  private readonly enableProliferation: boolean;
  private readonly proliferationAgencyThreshold: number;
  private readonly proliferationMinSamples: number;
  private readonly proliferationCooldown: number;

  constructor(config?: Partial<GeometricPortConfig>) {
    const cfg = { ...defaultGeometricPortConfig, ...config };
    this.equilibriumRate = cfg.equilibriumRate;
    this.tolerance = cfg.bindingRateTolerance;
    this.enableProliferation = cfg.enableProliferation;
    this.proliferationAgencyThreshold = cfg.proliferationAgencyThreshold;
    this.proliferationMinSamples = cfg.proliferationMinSamples;
    this.proliferationCooldown = cfg.proliferationCooldown;
  }

  decide(
    outcome: BindingOutcome,
    portId: string,
    _situationKey: string,
    history: BindingHistory
  ): RefinementAction {
    // Check proliferation first (same fiber-based logic as FixedPolicy)
    if (!outcome.success && this.enableProliferation) {
      const samples = history.getSampleCount(portId);
      const avgAgency = history.getAverageAgency(portId);
      const inCooldown = history.isInCooldown(
        portId,
        this.proliferationCooldown
      );

      if (
        samples >= this.proliferationMinSamples &&
        avgAgency < this.proliferationAgencyThreshold &&
        !inCooldown
      ) {
        // Fiber is inconsistent - proliferate to create specialists
        return "proliferate";
      }
    }

    // Control-theoretic refinement based on binding rate
    const bindingRate = history.getBindingRate(portId);
    const error = bindingRate - this.equilibriumRate;

    if (error > this.tolerance) {
      // Binding too much → increase distance (narrow cone, raise agency)
      return "narrow";
    } else if (error < -this.tolerance) {
      // Binding too little → decrease distance (widen cone, lower agency)
      return "widen";
    } else {
      // At equilibrium - no adjustment needed
      return "noop";
    }
  }
}

/**
 * AgencyGradientRefinementPolicy: Gradient-based refinement with no fixed thresholds.
 *
 * Core insight: Agency naturally reflects environmental uncertainty. Instead of
 * targeting a fixed binding rate, track the trajectory in (agency, bindingRate)
 * phase space and use gradients to guide adaptation.
 *
 * Decision logic:
 * - Success + margin + positive gradient → NARROW (room to commit more)
 * - Success + margin + negative gradient → NOOP (hit uncertainty ceiling)
 * - Failure → WIDEN (reduce commitment)
 *
 * The system automatically discovers environment-specific homeostasis by finding
 * where narrowing stops improving the (agency, binding) relationship.
 *
 * IMPORTANT: Uses only port-specific measurements. No global agency averaging.
 */
export class AgencyGradientRefinementPolicy implements RefinementPolicy {
  private readonly enableProliferation: boolean;
  private readonly proliferationAgencyThreshold: number;
  private readonly proliferationMinSamples: number;
  private readonly proliferationCooldown: number;

  // Per-port tracking of (agency, bindingRate, distance) trajectory
  private portTrajectories = new Map<
    string,
    Array<{ agency: number; bindingRate: number; distance: number; timestamp: number }>
  >();
  private readonly trajectoryWindowSize: number;
  private readonly dampingFactor: number; // Prevents overshoot

  constructor(config?: Partial<GeometricPortConfig>) {
    const cfg = { ...defaultGeometricPortConfig, ...config };
    this.enableProliferation = cfg.enableProliferation;
    this.proliferationAgencyThreshold = cfg.proliferationAgencyThreshold;
    this.proliferationMinSamples = cfg.proliferationMinSamples;
    this.proliferationCooldown = cfg.proliferationCooldown;
    this.trajectoryWindowSize = 20; // Track recent trajectory for gradient
    this.dampingFactor = 0.3; // If distance changing rapidly, be conservative
  }

  decide(
    outcome: BindingOutcome,
    portId: string,
    _situationKey: string,
    history: BindingHistory
  ): RefinementAction {
    // Check proliferation first (same fiber-based logic)
    if (!outcome.success && this.enableProliferation) {
      const samples = history.getSampleCount(portId);
      const avgAgency = history.getAverageAgency(portId);
      const inCooldown = history.isInCooldown(
        portId,
        this.proliferationCooldown
      );

      if (
        samples >= this.proliferationMinSamples &&
        avgAgency < this.proliferationAgencyThreshold &&
        !inCooldown
      ) {
        return "proliferate";
      }
    }

    // Update trajectory for this port
    const agency = history.getAverageAgency(portId); // Port-specific agency
    const bindingRate = history.getBindingRate(portId); // Port-specific binding rate
    // Recover distance from agency: agency = d / (1 + d), so d = agency / (1 - agency)
    const distance = agency > 0.999 ? 999 : agency / Math.max(1 - agency, 1e-8);
    this.updateTrajectory(portId, agency, bindingRate, distance);

    // Decision based on binding rate gradient WITH damping to prevent overshoot
    if (outcome.success) {
      // Binding succeeded - check if binding rate is improving
      const gradient = this.computeGradient(portId);

      if (!gradient.bindingDecreasing) {
        // Binding rate stable or increasing
        // But check velocity - if distance changing rapidly, be conservative
        if (gradient.distanceVelocity > this.dampingFactor) {
          // Distance increasing rapidly - ease off narrowing to prevent overshoot
          return "noop";
        }
        // Safe to narrow
        return "narrow";
      } else {
        // Binding rate decreasing - we've hit a limit, stop narrowing
        return "noop";
      }
    } else {
      // Binding failed - we were surprised, reduce commitment
      const gradient = this.computeGradient(portId);

      // Only apply damping if binding rate is reasonable
      // If binding rate is very low, we need to widen aggressively
      if (bindingRate > 0.3 && gradient.distanceVelocity < -this.dampingFactor) {
        // Distance decreasing rapidly and we're not desperate - ease off widening
        return "noop";
      }
      return "widen";
    }
  }

  /**
   * Update the (agency, bindingRate, distance) trajectory for a port.
   */
  private updateTrajectory(
    portId: string,
    agency: number,
    bindingRate: number,
    distance: number
  ): void {
    let trajectory = this.portTrajectories.get(portId);
    if (!trajectory) {
      trajectory = [];
      this.portTrajectories.set(portId, trajectory);
    }

    trajectory.push({ agency, bindingRate, distance, timestamp: Date.now() });

    // Trim to window size
    if (trajectory.length > this.trajectoryWindowSize) {
      trajectory.shift();
    }
  }

  /**
   * Compute gradient in (agency, bindingRate, distance) phase space.
   * Returns whether each metric is increasing over recent history, plus distance velocity.
   */
  private computeGradient(portId: string): {
    agencyIncreasing: boolean;
    bindingDecreasing: boolean;
    distanceVelocity: number;
  } {
    const trajectory = this.portTrajectories.get(portId);
    if (!trajectory || trajectory.length < 3) {
      // Not enough history - assume neutral
      return { agencyIncreasing: false, bindingDecreasing: false, distanceVelocity: 0 };
    }

    // Compare recent half vs older half
    const midpoint = Math.floor(trajectory.length / 2);
    const older = trajectory.slice(0, midpoint);
    const recent = trajectory.slice(midpoint);

    const olderAgency =
      older.reduce((sum, p) => sum + p.agency, 0) / older.length;
    const recentAgency =
      recent.reduce((sum, p) => sum + p.agency, 0) / recent.length;

    const olderBinding =
      older.reduce((sum, p) => sum + p.bindingRate, 0) / older.length;
    const recentBinding =
      recent.reduce((sum, p) => sum + p.bindingRate, 0) / recent.length;

    const olderDistance =
      older.reduce((sum, p) => sum + p.distance, 0) / older.length;
    const recentDistance =
      recent.reduce((sum, p) => sum + p.distance, 0) / recent.length;

    // Threshold for considering a change significant (5%)
    const significanceThreshold = 0.05;

    const agencyIncreasing =
      recentAgency - olderAgency > significanceThreshold;
    const bindingDecreasing =
      olderBinding - recentBinding > significanceThreshold;

    // Distance velocity (rate of change) - used for damping
    const distanceVelocity = recentDistance - olderDistance;

    return { agencyIncreasing, bindingDecreasing, distanceVelocity };
  }
}
