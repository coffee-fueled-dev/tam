/**
 * CommitmentNet: Shared neural network for viewing distance prediction.
 *
 * Maps (state_embedding, port_embedding) → distance (commitment level).
 *
 * In the geometric port model:
 * - Distance d controls how specific the commitment is (like stepping toward/away from window)
 * - d = 0: standing right at the window, seeing everything (low agency, wide cone)
 * - d → ∞: far from window, seeing very little (high agency, narrow cone)
 *
 * The effective cone radius = aperture × alignment / (1 + d)
 *
 * Training uses SYMMETRIC SUPERVISED LEARNING:
 * - On binding SUCCESS: target = current × (1 + narrowScale) → increase distance
 * - On binding FAILURE: target = current - (violation × widenScale) → decrease distance
 * - Both directions use MSE loss to explicit targets (stable, principled)
 *
 * This replaces the previous asymmetric approach (unbounded maximize vs supervised)
 * which was unstable in high dimensions and led to distance collapse.
 *
 * Initialization uses d ≈ 0 (close to window, wide cone) for epistemic humility.
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";
import { clamp } from "../vec";
import type { CommitmentNetConfig, RefinementAction } from "../types";
import { defaultCommitmentNetConfig } from "../types";

export class CommitmentNet {
  private model: tf.LayersModel | null = null;
  private narrowOptimizer: tf.Optimizer;
  private widenOptimizer: tf.Optimizer;
  private readonly cfg: CommitmentNetConfig;

  // Distance-based buffers (d = viewing distance / commitment level)
  private narrowBuffer: Array<{ input: Vec; currentDistance: number }> = [];
  private widenBuffer: Array<{
    input: Vec;
    currentDistance: number;
    violation: number;
  }> = [];
  private inputDim: number | null = null;

  constructor(config?: Partial<CommitmentNetConfig>) {
    this.cfg = { ...defaultCommitmentNetConfig, ...(config ?? {}) };
    // Separate optimizers for asymmetric training
    this.narrowOptimizer = tf.train.adam(this.cfg.learningRate);
    this.widenOptimizer = tf.train.adam(this.cfg.learningRate);
  }

  /**
   * Build the neural network model.
   * Input: concatenation of state embedding and port embedding.
   * Output: single scalar distance d (always non-negative via softplus).
   */
  private buildModel(inputDim: number): tf.LayersModel {
    const input = tf.input({ shape: [inputDim] });

    // Hidden layers to learn complex situation-dependent distance
    // Use He initialization for ReLU activations
    let x: tf.SymbolicTensor = input;
    for (const units of this.cfg.hiddenSizes) {
      x = tf.layers
        .dense({
          units,
          activation: "relu",
          kernelInitializer: "heNormal",
          biasInitializer: "zeros",
        })
        .apply(x) as tf.SymbolicTensor;
    }

    // Output single distance value
    // EPISTEMIC HUMILITY: Start with maximally wide cones (distance ≈ 0)
    // "I have no meaningful expectation until reality teaches me something"
    // softplus(-2) ≈ 0.13 → very low distance → very wide cone → low agency
    // Initialize kernel small so network can learn to increase distance from experience
    const rawDistance = tf.layers
      .dense({
        units: 1,
        kernelInitializer: tf.initializers.randomNormal({ mean: 0, stddev: 0.01 }),
        biasInitializer: tf.initializers.constant({ value: -2.0 }), // softplus(-2) ≈ 0.13
        name: "rawDistance",
      })
      .apply(x) as tf.SymbolicTensor;

    return tf.model({ inputs: input, outputs: rawDistance });
  }

  private ensureModel(inputDim: number): void {
    if (this.model) return;
    this.inputDim = inputDim;
    this.model = this.buildModel(inputDim);
  }

  /**
   * Convert raw network output to non-negative distance.
   */
  private rawToDistance(raw: tf.Tensor): tf.Tensor {
    // Softplus ensures d ≥ 0 with non-zero gradients everywhere
    // Unlike ReLU which has zero gradient for negative inputs
    // Gradient is sigmoid(x), which is non-zero everywhere but can saturate at extremes
    return raw.softplus();
  }

  /**
   * Predict viewing distance for a given state and port embedding.
   * Returns scalar d where cone_radius = aperture × alignment / (1 + d)
   */
  predictDistance(stateEmb: Vec, portEmb: Vec): number {
    const input = [...stateEmb, ...portEmb];
    const inputDim = input.length;

    // If no model yet, return d = 0 (at the window, maximum cone width)
    if (!this.model || this.inputDim !== inputDim) {
      return 0;
    }

    return tf.tidy(() => {
      const inputT = tf.tensor2d([input]);
      const rawOutput = this.model!.predict(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      return clamp(distance.dataSync()[0] ?? 0, 0, 100);
    });
  }

  /**
   * Legacy predict method for backward compatibility.
   * Converts distance to per-dimension radius using a reference aperture.
   * @deprecated Use predictDistance() for the geometric model
   */
  predict(stateEmb: Vec, portEmb: Vec): Vec {
    const d = this.predictDistance(stateEmb, portEmb);
    // For backward compatibility, convert d to radius-like values
    // Assuming alignment = 1 and aperture = initialRadius
    const effectiveRadius = this.cfg.initialRadius / (1 + d);
    const outputDim = stateEmb.length;
    return new Array(outputDim).fill(
      Math.max(effectiveRadius, this.cfg.minRadius)
    );
  }

  /**
   * Queue a refinement action for this input.
   * Actual training happens in batches.
   */
  queueRefinement(
    stateEmb: Vec,
    portEmb: Vec,
    action: RefinementAction,
    violation?: number
  ): void {
    if (action === "noop" || action === "proliferate") {
      return; // No training for these actions
    }

    const input = [...stateEmb, ...portEmb];
    const currentDistance = this.predictDistance(stateEmb, portEmb);

    this.ensureModel(input.length);

    if (action === "narrow") {
      this.narrowBuffer.push({ input: [...input], currentDistance });
      if (this.narrowBuffer.length >= this.cfg.batchSize) {
        this.trainNarrow();
      }
    } else if (action === "widen") {
      // Only widen if distance is meaningfully above minimum
      // (starting from near-zero, we have nowhere to widen to)
      if (currentDistance > this.cfg.minRadius) {
        this.widenBuffer.push({
          input: [...input],
          currentDistance,
          violation: violation ?? 1.0,
        });
        if (this.widenBuffer.length >= this.cfg.batchSize) {
          this.trainWiden();
        }
      }
    }
  }

  /**
   * Train to NARROW (increase distance): step back from window, narrow view, increase agency.
   * Only called on binding success.
   *
   * Uses additive update to escape zero trap: target = current + narrowStep
   */
  private trainNarrow(): void {
    if (!this.model || this.narrowBuffer.length === 0) return;

    const inputs = this.narrowBuffer.map((b) => b.input);
    // Target: increase distance by additive step (escapes zero)
    const targetDistances = this.narrowBuffer.map((b) => {
      // Additive increase: target = current + narrowStep
      const target = b.currentDistance + this.cfg.narrowStep;
      return [target];
    });

    // Debug: Log training targets and check predictions after training
    const avgCurrent = this.narrowBuffer.reduce((sum, b) => sum + b.currentDistance, 0) / this.narrowBuffer.length;
    const avgTarget = targetDistances.reduce((sum, t) => sum + t[0]!, 0) / targetDistances.length;

    // Check prediction BEFORE training
    const testInput = inputs[0]!;
    const predBefore = tf.tidy(() => {
      const inputT = tf.tensor2d([testInput]);
      const rawOutput = this.model!.predict(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      return distance.dataSync()[0];
    });

    if (Math.random() < 0.01) { // Log 1% of the time
      console.log(`[NARROW] Batch: ${this.narrowBuffer.length}, Pred before: ${predBefore?.toFixed(4)}, Avg target: ${avgTarget.toFixed(4)}`);
    }

    const inputT = tf.tensor2d(inputs);
    const targetT = tf.tensor2d(targetDistances);

    // Debug: Compute loss and gradients manually to see what's happening
    const lossBefore = tf.tidy(() => {
      const rawOutput = this.model!.apply(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      const loss = tf.losses.meanSquaredError(targetT, distance) as tf.Scalar;
      return loss.dataSync()[0];
    });

    // Supervise toward larger distance
    // CRITICAL: Use apply() not predict() for gradient tracking!
    const lossValue = this.narrowOptimizer.minimize(() => {
      const rawOutput = this.model!.apply(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      // Loss: MSE to target distance
      const loss = tf.losses.meanSquaredError(targetT, distance) as tf.Scalar;
      return loss;
    }, true); // returnCost = true

    if (Math.random() < 0.02) { // Log 2% of the time for better visibility
      console.log(`\n[NARROW DEBUG]`);
      console.log(`  Current distance: ${avgCurrent.toFixed(4)}, Target: ${avgTarget.toFixed(4)}`);
      console.log(`  Loss: ${lossBefore?.toFixed(6)}`);

      // Check prediction
      const pred = tf.tidy(() => {
        const raw = this.model!.apply(inputT) as tf.Tensor;
        const dist = this.rawToDistance(raw);
        return dist.dataSync()[0];
      });
      console.log(`  Network predicts: ${pred?.toFixed(4)} (should move toward ${avgTarget.toFixed(4)})`);

      // Check bias before/after
      const biasValBefore = this.model!.getWeights()[1]?.dataSync()[0];
      console.log(`  Bias before training: ${biasValBefore?.toFixed(4)}`);
    }

    if (lossValue) lossValue.dispose();

    // Check prediction AFTER training
    const predAfter = tf.tidy(() => {
      const inputT2 = tf.tensor2d([testInput]);
      const rawOutput = this.model!.predict(inputT2) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      return distance.dataSync()[0];
    });

    if (Math.random() < 0.01 && predBefore !== undefined) { // Log 1% of the time
      console.log(`    → After: ${predAfter?.toFixed(4)}, Delta: ${((predAfter ?? 0) - predBefore).toFixed(4)}`);
    }

    inputT.dispose();
    targetT.dispose();
    this.narrowBuffer = [];
  }

  /**
   * Train to WIDEN (decrease distance): step toward window, widen view, decrease agency.
   * Only called on binding failure.
   *
   * Uses additive update: target = current - (violation × widenStep)
   * Reduction scales with violation severity but uses additive step.
   */
  private trainWiden(): void {
    if (!this.model || this.widenBuffer.length === 0) return;

    const inputs = this.widenBuffer.map((b) => b.input);
    // Target: reduce distance by additive step scaled by violation
    const targetDistances = this.widenBuffer.map((b) => {
      // Additive reduction: target = current - (violation × widenStep)
      // Allow widening back to near-zero (epistemic humility)
      // Only prevent actual zero for numerical stability
      const reduction = b.violation * this.cfg.widenStep;
      return [Math.max(this.cfg.minRadius, b.currentDistance - reduction)];
    });

    // Debug: Log training targets
    const avgCurrent = this.widenBuffer.reduce((sum, b) => sum + b.currentDistance, 0) / this.widenBuffer.length;
    const avgTarget = targetDistances.reduce((sum, t) => sum + t[0]!, 0) / targetDistances.length;
    if (Math.random() < 0.01) { // Log 1% of the time
      console.log(`[WIDEN] Batch size: ${this.widenBuffer.length}, Avg current: ${avgCurrent.toFixed(4)}, Avg target: ${avgTarget.toFixed(4)}`);
    }

    const inputT = tf.tensor2d(inputs);
    const targetT = tf.tensor2d(targetDistances);

    // Supervise toward smaller distance
    // CRITICAL: Use apply() not predict() for gradient tracking!
    this.widenOptimizer.minimize(() => {
      const rawOutput = this.model!.apply(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      // Loss: MSE to target distance
      return tf.losses.meanSquaredError(targetT, distance) as tf.Scalar;
    });

    inputT.dispose();
    targetT.dispose();
    this.widenBuffer = [];
  }

  /**
   * Force training on any remaining buffered observations.
   */
  flush(): void {
    if (this.narrowBuffer.length > 0) {
      this.trainNarrow();
    }
    if (this.widenBuffer.length > 0) {
      this.trainWiden();
    }
  }

  /**
   * Get diagnostic snapshot.
   */
  snapshot(): unknown {
    return {
      inputDim: this.inputDim,
      narrowBufferSize: this.narrowBuffer.length,
      widenBufferSize: this.widenBuffer.length,
      hasModel: this.model !== null,
      config: this.cfg,
    };
  }

  /**
   * Clean up TensorFlow resources.
   */
  dispose(): void {
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.narrowOptimizer.dispose();
    this.widenOptimizer.dispose();
  }
}
