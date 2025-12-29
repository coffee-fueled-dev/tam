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
