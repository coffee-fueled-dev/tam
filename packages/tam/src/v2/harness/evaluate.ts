/**
 * Evaluation utilities for TAM v2 experiments
 */

import type { Actor } from "../actor";
import type { Vec } from "../types";

/**
 * Test evaluation results
 */
export interface EvaluationResults {
  avgAgency: number;
  avgError: number;
  testBindingRate: number;
}

/**
 * Evaluate actor performance on test samples
 *
 * @param actor - Actor to evaluate
 * @param generateSample - Function that generates a test state and its ground truth next state
 * @param embedState - Function to embed states into vectors
 * @param testSamples - Number of test samples (default: 50)
 * @returns Evaluation metrics
 */
export function evaluate<S>(
  actor: Actor<S>,
  generateSample: () => { before: S; after: S },
  embedState: (s: S) => Vec,
  testSamples: number = 50
): EvaluationResults {
  let totalAgency = 0;
  let totalError = 0;
  let totalBindings = 0;

  for (let i = 0; i < testSamples; i++) {
    const { before, after } = generateSample();
    const beforeEmb = embedState(before);
    const afterEmb = embedState(after);
    const truthDelta = afterEmb.map((val, idx) => val - beforeEmb[idx]!);

    const pred = actor.predict(before);

    // Compute prediction error (MSE)
    const error = Math.sqrt(
      truthDelta.reduce(
        (sum, truth, idx) => sum + (pred.delta[idx]! - truth) ** 2,
        0
      )
    );

    // Check if prediction would bind
    const normalizedDist = Math.sqrt(
      pred.cone.radius.reduce((sum, r, idx) => {
        const residual = truthDelta[idx]! - pred.cone.center[idx]!;
        const normalized = r > 0 ? residual / r : residual;
        return sum + normalized * normalized;
      }, 0)
    );
    const wouldBind = normalizedDist < 1.0;

    totalAgency += pred.agency;
    totalError += error;
    if (wouldBind) totalBindings++;
  }

  return {
    avgAgency: totalAgency / testSamples,
    avgError: totalError / testSamples,
    testBindingRate: totalBindings / testSamples,
  };
}
