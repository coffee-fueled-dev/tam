/**
 * TAM agent for learning physics dynamics
 *
 * Uses a SINGLE port that learns p(Δ | situation, action) where action
 * is encoded as part of the context. This allows the port to naturally
 * discover structure through proliferation rather than having ad-hoc
 * per-action ports.
 *
 * The "wave function" visualization shows these predictions.
 */

import {
  AgencyPort,
  type Encoders,
  type Situation,
  type Transition,
  type Vec,
} from "tam";
import { type Vec2, type PhysicsState, type Force, lenVec } from "./types";

// ============================================================================
// Situation types for TAM
// ============================================================================

interface PhysicsSitState {
  pos: Vec2;
  vel: Vec2;
}

interface PhysicsSitContext {
  // Action encoded as 2D force direction (for manual mode)
  actionX: number;
  actionY: number;
}

type PhysicsSituation = Situation<PhysicsSitState, PhysicsSitContext>;

// ============================================================================
// Encoders
// ============================================================================

/**
 * Minimal encoders: position + velocity + action.
 * The agent should discover structure (walls, gravity, etc.) from data alone.
 * No hand-crafted domain features like "nearWall".
 */
const physicsEncoders: Encoders<PhysicsSitState, PhysicsSitContext> = {
  embedSituation(sit: PhysicsSituation): Vec {
    const { pos, vel } = sit.state;
    const ctx = sit.context;

    // Normalize position to [0, 1] range (assuming 200x150 canvas for scenarios)
    // For playground (600x450), this will be 0-3 range which is still reasonable
    return [
      // Position (key for learning wall locations)
      pos.x / 200,
      pos.y / 150,
      // Velocity (key for predicting trajectories)
      vel.x / 10,
      vel.y / 10,
      // Action direction (for manual pushes)
      ctx.actionX,
      ctx.actionY,
    ];
  },

  delta(before: PhysicsSituation, after: PhysicsSituation): Vec {
    // Delta is change in position and velocity
    return [
      after.state.pos.x - before.state.pos.x,
      after.state.pos.y - before.state.pos.y,
      after.state.vel.x - before.state.vel.x,
      after.state.vel.y - before.state.vel.y,
    ];
  },
};

// ============================================================================
// Physics Agent with Single Universal Port
// ============================================================================

export class PhysicsAgent {
  /**
   * Single port that learns ALL dynamics.
   * Proliferation will naturally create modes for different action+context combinations.
   */
  private port: AgencyPort<PhysicsSitState, PhysicsSitContext>;
  private observations = 0;

  // Tracking prediction accuracy
  private lastPrediction: { pos: Vec2; vel: Vec2 } | null = null;
  private predictionErrors: number[] = []; // Rolling window of errors
  private surpriseCount = 0; // Times reality was outside the cone
  private correctCount = 0; // Times prediction was close to reality

  constructor() {
    // Single universal port for all actions
    this.port = new AgencyPort({
      action: "*", // Universal action
      name: "UniversalDynamics",
      encoders: physicsEncoders,
      config: {
        maxComponents: 16,
        reservoirSize: 40,
        // Reference variance must match the actual delta scale!
        // Physics deltas are ~1-10 for position, ~1-5 for velocity
        // So variance of ~5-10 is "average", not 100
        referenceVar: 10,
        minVar: 0.5, // Don't let variance collapse too far
        maxVar: 100,
        // Slow down narrowing so agent doesn't become overconfident
        narrowRate: 0.1,
        widenRate: 0.3, // Widen more aggressively when surprised
        // Split detection
        splitMinSamples: 10,
        splitCooldown: 30,
        splitMinAgencyGain: 0.03,
        minSplitSeparation: 1.0, // Physics deltas need more separation
        insideSigma: 2.5, // Stricter "inside" check
      },
    });
  }

  /**
   * Convert physics state + action to TAM situation
   * Minimal context: position + velocity + action
   * The agent discovers structure (walls, gravity) from data
   */
  stateToSituation(state: PhysicsState, action: string): PhysicsSituation {
    const actionVec = actionToVec(action);

    return {
      state: {
        pos: state.ballPos,
        vel: state.ballVel,
      },
      context: {
        actionX: actionVec.x,
        actionY: actionVec.y,
      },
    };
  }

  /**
   * Make a prediction BEFORE observing the transition.
   * Call this before the physics step, then call observe() after.
   */
  predictBefore(state: PhysicsState, action: string): void {
    const preds = this.predict(state, action);
    if (preds.length > 0) {
      // Use the highest-weighted prediction
      const best = preds.reduce((a, b) => (b.weight > a.weight ? b : a));
      this.lastPrediction = {
        pos: {
          x: state.ballPos.x + best.deltaPos.x,
          y: state.ballPos.y + best.deltaPos.y,
        },
        vel: {
          x: state.ballVel.x + best.deltaVel.x,
          y: state.ballVel.y + best.deltaVel.y,
        },
      };
    } else {
      this.lastPrediction = null;
    }
  }

  /**
   * Observe a physics transition and evaluate prediction accuracy.
   */
  observe(
    beforeState: PhysicsState,
    afterState: PhysicsState,
    action: string
  ): void {
    this.observations++;

    // Evaluate prediction accuracy if we made a prediction
    if (this.lastPrediction) {
      const actualPos = afterState.ballPos;
      const predictedPos = this.lastPrediction.pos;

      const posError = Math.sqrt(
        (actualPos.x - predictedPos.x) ** 2 +
          (actualPos.y - predictedPos.y) ** 2
      );

      // Track errors (rolling window of 50)
      this.predictionErrors.push(posError);
      if (this.predictionErrors.length > 50) {
        this.predictionErrors.shift();
      }

      // Count as "correct" if within 5 pixels, "surprise" if > 20 pixels
      if (posError < 5) {
        this.correctCount++;
      } else if (posError > 20) {
        this.surpriseCount++;
      }
    }

    // Note: we use "*" as the action in transition since the real action
    // is encoded in the context
    const transition: Transition<PhysicsSitState, PhysicsSitContext> = {
      action: "*", // Universal
      before: this.stateToSituation(beforeState, action),
      after: this.stateToSituation(afterState, action),
    };

    this.port.observe(transition);
  }

  /**
   * Get the last prediction for visualization.
   */
  getLastPrediction(): { pos: Vec2; vel: Vec2 } | null {
    return this.lastPrediction;
  }

  /**
   * Get prediction accuracy statistics.
   */
  getAccuracyStats(): {
    avgError: number;
    accuracy: number; // % of predictions within tolerance
    surpriseRate: number; // % of large misses
  } {
    const total = this.correctCount + this.surpriseCount;
    const avgError =
      this.predictionErrors.length > 0
        ? this.predictionErrors.reduce((a, b) => a + b, 0) /
          this.predictionErrors.length
        : 0;

    return {
      avgError,
      accuracy: total > 0 ? this.correctCount / total : 0,
      surpriseRate: total > 0 ? this.surpriseCount / total : 0,
    };
  }

  /**
   * Predict future state given current state and action
   * Returns predictions with uncertainty (for wave function visualization)
   */
  predict(
    state: PhysicsState,
    action: string
  ): Array<{
    deltaPos: Vec2;
    deltaVel: Vec2;
    weight: number;
    agency: number;
    variance: Vec2;
  }> {
    const sit = this.stateToSituation(state, action);
    const predictions = this.port.predictDeltas(sit, 5);

    return predictions.map((p) => {
      const delta = p.delta;
      return {
        deltaPos: { x: delta[0] ?? 0, y: delta[1] ?? 0 },
        deltaVel: { x: delta[2] ?? 0, y: delta[3] ?? 0 },
        weight: Math.exp(p.score),
        agency: p.agency,
        variance: {
          x: (1 - p.agency) * 50 + 5,
          y: (1 - p.agency) * 50 + 5,
        },
      };
    });
  }

  /**
   * Get overall agency (how well the agent understands dynamics)
   */
  getTotalAgency(): number {
    return this.port.getTotalAgency();
  }

  /**
   * Get number of observations
   */
  getObservations(): number {
    return this.observations;
  }

  /**
   * Get stats for display
   */
  getStats() {
    const snapshot = this.port.snapshot() as {
      totalAgency: number;
      components: Array<{
        index: number;
        agency: number;
        seen: number;
        mean: number[];
        varDiag: number[];
      }>;
    };

    const accuracy = this.getAccuracyStats();

    return {
      observations: this.observations,
      totalAgency: snapshot.totalAgency,
      components: snapshot.components.length,
      // Accuracy metrics (ground truth comparison)
      avgError: accuracy.avgError,
      accuracy: accuracy.accuracy,
      surpriseRate: accuracy.surpriseRate,
      modes: snapshot.components.map((c) => ({
        index: c.index,
        agency: c.agency,
        seen: c.seen,
        avgDelta: c.mean.slice(0, 2),
        avgVar: c.varDiag.reduce((s, v) => s + v, 0) / c.varDiag.length,
      })),
    };
  }

  /**
   * Get detailed mode information for visualization
   */
  getModes(): Array<{
    index: number;
    agency: number;
    seen: number;
    avgDeltaPos: Vec2;
    avgDeltaVel: Vec2;
  }> {
    const snapshot = this.port.snapshot() as {
      components: Array<{
        index: number;
        agency: number;
        seen: number;
        mean: number[];
      }>;
    };

    return snapshot.components.map((c) => ({
      index: c.index,
      agency: c.agency,
      seen: c.seen,
      avgDeltaPos: { x: c.mean[0] ?? 0, y: c.mean[1] ?? 0 },
      avgDeltaVel: { x: c.mean[2] ?? 0, y: c.mean[3] ?? 0 },
    }));
  }
}

// ============================================================================
// Action encoding utilities
// ============================================================================

/**
 * Convert action string to normalized direction vector.
 * This becomes part of the context for the universal port.
 */
function actionToVec(action: string): Vec2 {
  switch (action) {
    case "up":
      return { x: 0, y: -1 };
    case "down":
      return { x: 0, y: 1 };
    case "left":
      return { x: -1, y: 0 };
    case "right":
      return { x: 1, y: 0 };
    default:
      return { x: 0, y: 0 };
  }
}

/**
 * Force direction to string action
 */
export function forceToAction(force: Force): string {
  if (Math.abs(force.x) > Math.abs(force.y)) {
    return force.x > 0 ? "right" : "left";
  } else if (Math.abs(force.y) > 0) {
    return force.y > 0 ? "down" : "up";
  }
  return "none";
}

/**
 * Action string to force
 */
export function actionToForce(action: string, magnitude: number = 5): Force {
  switch (action) {
    case "up":
      return { x: 0, y: -magnitude };
    case "down":
      return { x: 0, y: magnitude };
    case "left":
      return { x: -magnitude, y: 0 };
    case "right":
      return { x: magnitude, y: 0 };
    default:
      return { x: 0, y: 0 };
  }
}
