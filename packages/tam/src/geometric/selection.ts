/**
 * Port selection strategies for GeometricPortBank.
 *
 * Port selection is itself a learning problem - different strategies
 * may be appropriate for different tasks and contexts.
 *
 * The strategy pattern allows:
 * - Simple hardcoded rules (e.g., max agency)
 * - Learned policies (e.g., TAM-based selection)
 * - Hybrid approaches (e.g., binding-weighted agency)
 */

import type { Vec } from "../vec";
import type { GeometricPort } from "./port";
import type { DefaultBindingHistory } from "./history";

/**
 * Context provided to selection strategy.
 */
export interface SelectionContext<S, C = unknown> {
  /** The action being selected for */
  action: string;

  /** Current state embedding */
  stateEmb: Vec;

  /** Candidate ports for this action */
  candidates: GeometricPort<S, C>[];

  /** Binding history (for tracking success rates) */
  history: DefaultBindingHistory;

  /** Optional: Intended outcome (for goal-directed selection) */
  intention?: Vec;
}

/**
 * Result of port selection.
 */
export interface SelectionResult<S, C = unknown> {
  /** The selected port (null if none applicable) */
  port: GeometricPort<S, C> | null;

  /** Optional: Confidence in this selection (for diagnostics) */
  confidence?: number;

  /** Optional: Reason for selection (for interpretability) */
  reason?: string;
}

/**
 * Strategy for selecting a port from candidates.
 *
 * Implementations can be:
 * - Simple heuristics (max agency, random, etc.)
 * - Learned policies (neural networks, TAM actors, etc.)
 * - Hybrid approaches (combining multiple signals)
 */
export interface PortSelectionStrategy<S, C = unknown> {
  /**
   * Select a port from candidates given the context.
   *
   * @param context - Selection context (state, candidates, history, etc.)
   * @returns Selection result (port and optional metadata)
   */
  select(context: SelectionContext<S, C>): SelectionResult<S, C>;

  /**
   * Optional: Learn from the outcome of a selection.
   *
   * Called after observing the binding outcome of a selected port.
   * Allows the strategy to improve over time.
   *
   * @param context - The original selection context
   * @param selectedPort - The port that was selected
   * @param bindingSuccess - Whether the binding was successful
   * @param predictionError - The prediction error magnitude
   */
  learn?(
    context: SelectionContext<S, C>,
    selectedPort: GeometricPort<S, C>,
    bindingSuccess: boolean,
    predictionError: number
  ): void;

  /**
   * Optional: Clean up resources (e.g., dispose TensorFlow models).
   */
  dispose?(): void;
}

/**
 * Max-agency selection strategy (current default behavior).
 *
 * Selects the port with highest agency (narrowest cone) among
 * applicable ports (non-empty cones).
 *
 * This is a simple heuristic that doesn't consider:
 * - Binding history (reliability)
 * - Goal alignment (utility)
 * - Long-term consequences
 *
 * Best for: Pure prediction tasks without specific goals.
 */
export class MaxAgencySelectionStrategy<S, C = unknown>
  implements PortSelectionStrategy<S, C>
{
  private minAlignmentThreshold: number;

  constructor(config?: { minAlignmentThreshold?: number }) {
    this.minAlignmentThreshold = config?.minAlignmentThreshold ?? 0.1;
  }

  select(context: SelectionContext<S, C>): SelectionResult<S, C> {
    const { stateEmb, candidates } = context;

    if (candidates.length === 0) {
      return { port: null, reason: "No candidates available" };
    }

    // Compute agency for each candidate
    const withAgency = candidates.map((port) => ({
      port,
      agency: port.computeAgencyFor(stateEmb),
      applicable: port.isApplicable(stateEmb),
    }));

    // Filter to applicable ports (non-empty cones)
    const eligible = withAgency.filter(({ applicable }) => applicable);

    if (eligible.length === 0) {
      return { port: null, reason: "No applicable ports (all have empty cones)" };
    }

    // Select max agency (narrowest cone = most specific commitment)
    const best = eligible.reduce((best, curr) =>
      curr.agency > best.agency ? curr : best
    );

    return {
      port: best.port,
      confidence: best.agency,
      reason: `Max agency: ${best.agency.toFixed(3)}`,
    };
  }
}

/**
 * Binding-weighted selection strategy.
 *
 * Like max-agency, but weights by historical binding success rate.
 * This penalizes overconfident ports that fail often.
 *
 * score = agency × binding_rate
 *
 * Best for: Tasks where binding reliability is known and important.
 */
export class BindingWeightedSelectionStrategy<S, C = unknown>
  implements PortSelectionStrategy<S, C>
{
  private minAlignmentThreshold: number;

  constructor(config?: { minAlignmentThreshold?: number }) {
    this.minAlignmentThreshold = config?.minAlignmentThreshold ?? 0.1;
  }

  select(context: SelectionContext<S, C>): SelectionResult<S, C> {
    const { stateEmb, candidates, history } = context;

    if (candidates.length === 0) {
      return { port: null, reason: "No candidates available" };
    }

    // Compute agency and binding rate for each candidate
    const withScores = candidates.map((port) => {
      const agency = port.computeAgencyFor(stateEmb);
      const applicable = port.isApplicable(stateEmb);

      // Get binding success rate from history
      // Use coverage rate (binding success) across all situations for this port
      const sampleCount = history.getSampleCount(port.id);
      const calibrationDiag = history.getCalibrationDiagnostics(port.id);
      const bindingRate = sampleCount > 0 ? calibrationDiag.coverageRate : 0.5;

      return {
        port,
        agency,
        bindingRate,
        score: agency * bindingRate,
        applicable,
      };
    });

    // Filter to applicable ports
    const eligible = withScores.filter(({ applicable }) => applicable);

    if (eligible.length === 0) {
      return { port: null, reason: "No applicable ports (all have empty cones)" };
    }

    // Select max score (agency × binding_rate)
    const best = eligible.reduce((best, curr) =>
      curr.score > best.score ? curr : best
    );

    return {
      port: best.port,
      confidence: best.score,
      reason: `Binding-weighted: agency=${(best.agency ?? 0).toFixed(3)}, binding=${(best.bindingRate ?? 0).toFixed(3)}`,
    };
  }
}

/**
 * Random selection strategy (for baseline comparison).
 *
 * Selects a random applicable port.
 * Useful for establishing baseline performance.
 */
export class RandomSelectionStrategy<S, C = unknown>
  implements PortSelectionStrategy<S, C>
{
  select(context: SelectionContext<S, C>): SelectionResult<S, C> {
    const { stateEmb, candidates } = context;

    if (candidates.length === 0) {
      return { port: null, reason: "No candidates available" };
    }

    // Filter to applicable ports
    const eligible = candidates.filter((port) => port.isApplicable(stateEmb));

    if (eligible.length === 0) {
      return { port: null, reason: "No applicable ports" };
    }

    // Select random
    const selected = eligible[Math.floor(Math.random() * eligible.length)]!;

    return {
      port: selected,
      confidence: 1.0 / eligible.length,
      reason: "Random selection",
    };
  }
}
