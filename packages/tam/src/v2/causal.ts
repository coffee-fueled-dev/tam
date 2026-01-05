/**
 * CausalNet: Predicts trajectories in state space WITH uncertainty
 *
 * Input: (stateEmb, portEmb)
 * Output: {mean: trajectory delta, variance: uncertainty per dimension}
 *
 * Dual-head architecture:
 * - Mean head: predicts trajectory
 * - Log-variance head: predicts log(σ²) for heteroscedastic uncertainty
 *
 * Soft conditional NLL (one-sided bounded) creates homeostatic feedback:
 * - High binding (success): Standard NLL - learn accurate predictions
 * - Low binding (failure): Min variance penalty - push variance up if below floor
 *
 * Key: ONE-SIDED penalty. Variance can shrink freely when binding is high,
 * but is prevented from going below minVariance when binding is low.
 *
 * This completes the self-organizing loop:
 *   variance ↓ → narrow cones → binding ↓ → push variance above floor → cones widen
 *
 * The system stabilizes at domain-dependent equilibrium. The minVariance parameter
 * (default 5.0) can be tuned per domain for optimal performance.
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "./types";
import type { CausalNetConfig } from "./types";
import type Queue from "queue";

interface TrainingExample {
  stateEmb: Vec;
  portEmb: Vec;
  trajectory: Vec;
  weight?: number; // Binding strength for homeostatic gating
}

export interface PredictionWithUncertainty {
  mean: Vec;
  variance: Vec;
}

export class CausalNet {
  private model: tf.LayersModel | null = null;
  private trainingBuffer: TrainingExample[] = [];
  private config: Required<CausalNetConfig>;
  private queue: Queue;

  constructor(queue: Queue, config?: CausalNetConfig) {
    this.queue = queue;
    this.config = {
      hiddenSizes: config?.hiddenSizes ?? [32, 32],
      learningRate: config?.learningRate ?? 0.001,
      batchSize: config?.batchSize ?? 10,
      minVariance: config?.minVariance ?? 5.0,  // Not used anymore
    };
  }

  /**
   * Predict trajectory and uncertainty for a (state, port) pair.
   */
  predict(stateEmb: Vec, portEmb: Vec): PredictionWithUncertainty {
    if (!this.model) {
      // No model yet - return zero prediction with high uncertainty
      const dim = stateEmb.length;
      return {
        mean: Array(dim).fill(0),
        variance: Array(dim).fill(1.0), // High initial uncertainty
      };
    }

    const input = [...stateEmb, ...portEmb];
    const inputTensor = tf.tensor2d([input]);
    const outputs = this.model.predict(inputTensor) as tf.Tensor[];

    // Dual outputs: [mean, logVariance]
    const meanResult = outputs[0]!.arraySync() as number[][];
    const logVarResult = outputs[1]!.arraySync() as number[][];

    inputTensor.dispose();
    outputs[0]!.dispose();
    outputs[1]!.dispose();

    const mean = meanResult[0]!;
    const logVar = logVarResult[0]!;

    // Clip log-variance for numerical stability: log(σ²) ∈ [-10, 10] → σ² ∈ [4.5e-5, 22026]
    // Defensive measure to prevent NaN if model outputs extreme values
    const clippedLogVar = logVar.map(lv => Math.max(-10, Math.min(10, lv)));
    const variance = clippedLogVar.map(lv => Math.exp(lv)); // Convert log(σ²) → σ²

    return { mean, variance };
  }

  /**
   * Queue a training example with optional binding-based weighting.
   *
   * @param weight - Binding strength [0, 1]. Used for homeostatic gating:
   *   - 1.0: Strong binding, learn fully from this example
   *   - 0.5: Weak binding, learn partially
   *   - 0.0: No binding, minimal learning but still provides signal
   *
   * This creates negative feedback: narrow cones → low binding → less training
   * → worse predictions → higher variance → wider cones. Equilibrium emerges
   * naturally based on domain noise.
   *
   * No threshold - the continuous weighting creates smooth self-balancing.
   */
  async observe(stateEmb: Vec, portEmb: Vec, trajectory: Vec, weight: number = 1.0): Promise<void> {
    // Always add to buffer - let continuous weighting create the feedback
    this.trainingBuffer.push({
      stateEmb: [...stateEmb],
      portEmb: [...portEmb],
      trajectory: [...trajectory],
      weight
    });

    if (this.trainingBuffer.length >= this.config.batchSize) {
      await this.train();
    }
  }

  /**
   * Train on buffered examples with custom NLL loss.
   */
  async train(): Promise<void> {
    if (this.trainingBuffer.length === 0) return;

    const stateDim = this.trainingBuffer[0]!.stateEmb.length;
    const portDim = this.trainingBuffer[0]!.portEmb.length;
    const trajectoryDim = this.trainingBuffer[0]!.trajectory.length;

    // Build model if needed
    if (!this.model) {
      this.model = this.buildModel(stateDim + portDim, trajectoryDim);
    }

    // Prepare training data with weights
    const inputs: number[][] = [];
    const targets: number[][] = [];
    const weights: number[] = [];

    for (const example of this.trainingBuffer) {
      inputs.push([...example.stateEmb, ...example.portEmb]);
      targets.push(example.trajectory);
      weights.push(example.weight ?? 1.0); // Default to full weight if not specified
    }

    const inputTensor = tf.tensor2d(inputs);
    const targetTensor = tf.tensor2d(targets);
    const weightTensor = tf.tensor1d(weights);

    // Enqueue training to prevent concurrent operations
    await new Promise<void>((resolve, reject) => {
      this.queue.push(async () => {
        try {
          // Custom training step with NLL loss and binding-based weighting
          await this.trainStep(inputTensor, targetTensor, weightTensor);

          // Dispose tensors after training completes
          inputTensor.dispose();
          targetTensor.dispose();
          weightTensor.dispose();

          resolve();
        } catch (error) {
          reject(error);
        }
      });
    });

    // Clear buffer
    this.trainingBuffer = [];
  }

  /**
   * Custom training step with soft conditional NLL (one-sided bounded).
   *
   * Creates homeostatic feedback through asymmetric learning:
   * - High binding (success): Standard NLL - learn accurate mean & variance
   * - Low binding (failure): Min variance penalty - push variance UP if too low
   *
   * Key: ONE-SIDED penalty. Only pushes variance up when it's below minimum.
   * Never prevents variance from shrinking when predictions are good.
   *
   * This prevents overconfidence while allowing calibration:
   *   narrow cones → low binding → push variance above min → cones widen
   *   accurate predictions → high binding → NLL can shrink variance freely
   *
   * Minimum variance is configurable per domain (default 5.0).
   */
  private async trainStep(
    inputs: tf.Tensor2D,
    targets: tf.Tensor2D,
    weights: tf.Tensor1D
  ): Promise<void> {
    const optimizer = tf.train.adam(this.config.learningRate);

    // Minimum log-variance for binding failures (configurable)
    const minLogVar = Math.log(this.config.minVariance);

    // Single gradient descent step
    optimizer.minimize(() => {
      return tf.tidy(() => {
        // Forward pass
        const outputs = this.model!.predict(inputs) as tf.Tensor[];
        const meanPred = outputs[0]! as tf.Tensor2D;
        const logVarPred = outputs[1]! as tf.Tensor2D;

        // Compute error terms
        const error = tf.sub(targets, meanPred);
        const errorSquared = tf.square(error);
        const variance = tf.exp(logVarPred);

        // Prevent numerical issues with very small variance
        const clippedVariance = tf.maximum(variance, 1e-6);

        // Standard NLL per example (for binding successes)
        const nllPerExample = tf.mean(
          tf.add(
            tf.div(errorSquared, clippedVariance),
            logVarPred
          ),
          1 // Reduce over output dimensions
        );

        // One-sided minimum variance penalty (for binding failures)
        // Only penalize if variance is BELOW minimum (too confident)
        // relu(minLogVar - logVar) = max(0, minLogVar - logVar)
        const minLogVarTensor = tf.scalar(minLogVar);
        const varianceDeficit = tf.sub(minLogVarTensor, logVarPred);
        const minVariancePenaltyPerExample = tf.mean(
          tf.square(tf.relu(varianceDeficit)),
          1 // Reduce over output dimensions
        );
        minLogVarTensor.dispose();

        // Soft conditional: blend based on binding strength
        // High binding → mostly NLL (can shrink variance freely)
        // Low binding → mostly min penalty (push variance up if too low)
        const bindingStrength = weights;
        const successLoss = tf.mul(nllPerExample, bindingStrength);
        const failureLoss = tf.mul(
          minVariancePenaltyPerExample,
          tf.sub(1, bindingStrength)
        );

        // Combined loss
        const combinedLoss = tf.add(successLoss, failureLoss);
        const loss = tf.mean(combinedLoss);

        // Scale by 0.5 for proper NLL calibration
        return tf.mul(loss, 0.5);
      });
    });

    optimizer.dispose();
  }

  /**
   * Force flush remaining buffered data.
   */
  async flush(): Promise<void> {
    if (this.trainingBuffer.length > 0) {
      await this.train();
    }
  }

  /**
   * Build the dual-head neural network.
   *
   * Architecture:
   *   input → shared hidden layers → mean head (linear)
   *                                 → log-variance head (linear)
   */
  private buildModel(inputDim: number, outputDim: number): tf.LayersModel {
    const inputs = tf.input({ shape: [inputDim] });
    let x = inputs;

    // Shared hidden layers
    for (const size of this.config.hiddenSizes) {
      x = tf.layers.dense({ units: size, activation: "relu" }).apply(x) as tf.SymbolicTensor;
    }

    // Dual output heads
    const meanOutput = tf.layers.dense({
      units: outputDim,
      activation: "linear",
      name: "mean",
    }).apply(x) as tf.SymbolicTensor;

    const logVarOutput = tf.layers.dense({
      units: outputDim,
      activation: "linear",
      name: "logVariance",
    }).apply(x) as tf.SymbolicTensor;

    const model = tf.model({
      inputs,
      outputs: [meanOutput, logVarOutput],
    });

    // Note: We don't use model.compile() because we use a custom training loop
    // with NLL loss in trainStep()

    return model;
  }

  /**
   * Clean up resources.
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.trainingBuffer = [];
  }
}
