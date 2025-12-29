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
 * Training is ASYMMETRIC:
 * - On binding SUCCESS: increase distance (step back, narrow view, increase agency)
 * - On binding FAILURE: decrease distance (step forward, widen view, decrease agency)
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

    let x: tf.SymbolicTensor = input;
    for (const units of this.cfg.hiddenSizes) {
      x = tf.layers
        .dense({
          units,
          activation: "relu",
          kernelInitializer: "heNormal",
        })
        .apply(x) as tf.SymbolicTensor;
    }

    // Output single distance value
    // Initialize bias to produce d ≈ 0 (close to window, wide cone, low agency)
    const rawDistance = tf.layers
      .dense({
        units: 1,
        kernelInitializer: "zeros",
        biasInitializer: tf.initializers.constant({ value: 0 }),
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
    // softplus ensures d ≥ 0
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

  /**
   * Train to NARROW (increase distance): step back from window, narrow view, increase agency.
   * Only called on binding success.
   */
  private trainNarrow(): void {
    if (!this.model || this.narrowBuffer.length === 0) return;

    const inputs = this.narrowBuffer.map((b) => b.input);
    const inputT = tf.tensor2d(inputs);

    // Maximize distance (step back from window)
    this.narrowOptimizer.minimize(() => {
      const rawOutput = this.model!.predict(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      // Minimize negative distance (maximize distance)
      return tf.neg(tf.mean(distance)) as tf.Scalar;
    });

    inputT.dispose();
    this.narrowBuffer = [];
  }

  /**
   * Train to WIDEN (decrease distance): step toward window, widen view, decrease agency.
   * Only called on binding failure.
   */
  private trainWiden(): void {
    if (!this.model || this.widenBuffer.length === 0) return;

    const inputs = this.widenBuffer.map((b) => b.input);
    // Target: reduce distance proportional to violation severity
    const targetDistances = this.widenBuffer.map((b) => {
      // Reduce distance by violation factor (but don't go below 0)
      const reduction = Math.min(b.violation * 0.5, b.currentDistance);
      return [Math.max(0, b.currentDistance - reduction)];
    });

    const inputT = tf.tensor2d(inputs);
    const targetT = tf.tensor2d(targetDistances);

    // Supervise toward smaller distance
    this.widenOptimizer.minimize(() => {
      const rawOutput = this.model!.predict(inputT) as tf.Tensor;
      const distance = this.rawToDistance(rawOutput);
      // Loss: distance from target (smaller) distance
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
