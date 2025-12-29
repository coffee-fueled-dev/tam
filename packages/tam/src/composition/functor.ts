/**
 * FunctorNetwork: Learnable mapping between domain embeddings.
 *
 * Implements a neural network that transforms state embeddings
 * from a source domain to a target domain.
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";
import type {
  FunctorNetwork,
  FunctorDiscoveryConfig,
  FunctorResult,
  FunctorStatus,
  FunctorReason,
} from "./types";
import { defaultFunctorDiscoveryConfig } from "./types";

/**
 * TensorFlow.js implementation of FunctorNetwork.
 */
export class TFFunctorNetwork implements FunctorNetwork {
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;
  public readonly inputDim: number;
  public readonly outputDim: number;

  constructor(
    inputDim: number,
    outputDim: number,
    hiddenSizes: number[] = [32, 32],
    learningRate: number = 0.01
  ) {
    this.inputDim = inputDim;
    this.outputDim = outputDim;

    const input = tf.input({ shape: [inputDim] });
    let x: tf.SymbolicTensor = input;

    for (const units of hiddenSizes) {
      x = tf.layers
        .dense({ units, activation: "relu", kernelInitializer: "heNormal" })
        .apply(x) as tf.SymbolicTensor;
    }

    const output = tf.layers
      .dense({ units: outputDim, kernelInitializer: "glorotNormal" })
      .apply(x) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: output });
    this.optimizer = tf.train.adam(learningRate);
  }

  apply(stateEmb: Vec): Vec {
    return tf.tidy(() => {
      const input = tf.tensor2d([stateEmb]);
      const output = this.model.predict(input) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  /**
   * Train the functor to map source deltas to target deltas.
   * Returns loss and whether the transformed delta is close to the target.
   */
  trainStep(
    sourceDelta: Vec,
    targetDelta: Vec,
    tolerance: number = 0.5
  ): { loss: number; success: boolean } {
    let loss = 0;

    tf.tidy(() => {
      const sourceT = tf.tensor2d([sourceDelta]);
      const targetT = tf.tensor2d([targetDelta]);

      this.optimizer.minimize(() => {
        const predicted = this.model.predict(sourceT) as tf.Tensor;
        const mse = tf.losses.meanSquaredError(targetT, predicted);
        loss = (mse as tf.Scalar).dataSync()[0]!;
        return mse as tf.Scalar;
      });
    });

    // Check if prediction is within tolerance
    const predicted = this.apply(sourceDelta);
    const error = Math.sqrt(
      predicted.reduce((s, p, i) => s + (p - targetDelta[i]!) ** 2, 0) /
        predicted.length
    );
    const success = error < tolerance;

    return { loss, success };
  }

  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}

/**
 * Discover a functor between two domains using paired transitions.
 *
 * The functor learns to map deltas from the source domain to deltas
 * in the target domain. This works when both domains share underlying
 * structure that can be learned.
 */
export async function discoverFunctor(
  sourceName: string,
  targetName: string,
  sourceRandomState: () => unknown,
  sourceSimulate: (s: unknown) => unknown,
  sourceEmbedder: (s: unknown) => Vec,
  targetRandomState: () => unknown,
  targetSimulate: (s: unknown) => unknown,
  targetEmbedder: (s: unknown) => Vec,
  config: Partial<FunctorDiscoveryConfig> = {}
): Promise<FunctorResult> {
  const cfg = { ...defaultFunctorDiscoveryConfig, ...config };

  // Determine dimensions from samples
  const sourceSample = sourceRandomState();
  const sourceEmb = sourceEmbedder(sourceSample);
  const inputDim = sourceEmb.length;

  const targetSample = targetRandomState();
  const targetEmb = targetEmbedder(targetSample);
  const outputDim = targetEmb.length;

  // Create functor that maps deltas, not states
  const functor = new TFFunctorNetwork(
    inputDim,
    outputDim,
    cfg.hiddenSizes,
    cfg.learningRate
  );

  let bestSuccessRate = 0;
  let epochsSinceImprovement = 0;
  let finalSuccessRate = 0;
  let finalEpoch = 0;
  let reason: FunctorReason = "timeout";

  // Tolerance for considering a mapping successful
  const tolerance = 0.3;

  for (let epoch = 0; epoch <= cfg.maxEpochs; epoch++) {
    let successes = 0;

    for (let i = 0; i < cfg.samplesPerEpoch; i++) {
      // Sample paired transitions from both domains
      const sourceState = sourceRandomState();
      const sourceAfter = sourceSimulate(sourceState);
      const targetState = targetRandomState();
      const targetAfter = targetSimulate(targetState);

      // Compute deltas in embedding space
      const sourceBeforeEmb = sourceEmbedder(sourceState);
      const sourceAfterEmb = sourceEmbedder(sourceAfter);
      const sourceDelta = sourceAfterEmb.map((a, i) => a - sourceBeforeEmb[i]!);

      const targetBeforeEmb = targetEmbedder(targetState);
      const targetAfterEmb = targetEmbedder(targetAfter);
      const targetDelta = targetAfterEmb.map((a, i) => a - targetBeforeEmb[i]!);

      // Train functor to map source delta → target delta
      const { success } = functor.trainStep(sourceDelta, targetDelta, tolerance);
      if (success) successes++;
    }

    const successRate = successes / cfg.samplesPerEpoch;
    finalSuccessRate = successRate;
    finalEpoch = epoch;

    // Check for improvement
    if (successRate > bestSuccessRate) {
      bestSuccessRate = successRate;
      epochsSinceImprovement = 0;
    } else {
      epochsSinceImprovement++;
    }

    // Check termination conditions
    if (successRate >= cfg.successThreshold) {
      reason = "converged";
      break;
    }

    if (epoch >= cfg.minEpochs && epochsSinceImprovement >= cfg.patience) {
      reason = "stalled";
      break;
    }
  }

  // Determine status
  let status: FunctorStatus;
  if (finalSuccessRate >= cfg.successThreshold) {
    status = "found";
  } else if (finalSuccessRate >= cfg.successThreshold * 0.5) {
    status = "inconclusive";
  } else {
    status = "not_found";
  }

  // If not found, dispose the functor
  if (status === "not_found") {
    functor.dispose();
    return {
      status,
      bindingRate: finalSuccessRate,
      epochs: finalEpoch,
      reason,
      source: sourceName,
      target: targetName,
    };
  }

  return {
    status,
    functor,
    bindingRate: finalSuccessRate,
    epochs: finalEpoch,
    reason,
    source: sourceName,
    target: targetName,
  };
}
