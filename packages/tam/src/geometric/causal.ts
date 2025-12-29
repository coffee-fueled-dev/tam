/**
 * CausalNet: Shared neural network for trajectory prediction.
 *
 * Maps (state_embedding, port_embedding) → predicted trajectory.
 * This is the causal manifold - predicting what will happen given a port and situation.
 *
 * Training uses MSE loss toward observed trajectories.
 * The network is shared across all ports - each port is just an embedding vector.
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";
import type { CausalNetConfig, GeometricPortConfig } from "../types";
import { defaultCausalNetConfig, defaultGeometricPortConfig } from "../types";

export class CausalNet {
  private model: tf.LayersModel | null = null;
  private optimizer: tf.Optimizer;
  private readonly cfg: CausalNetConfig;

  // Fiber consistency config (optional)
  private fiberConsistencyWeight: number;
  private fiberThreshold: number;

  private buffer: Array<{ stateEmb: Vec; portEmb: Vec; target: Vec }> = [];
  private inputDim: number | null = null;
  private outputDim: number | null = null;

  constructor(
    config?: Partial<CausalNetConfig>,
    fiberConfig?: Pick<GeometricPortConfig, "fiberConsistencyWeight" | "fiberThreshold">
  ) {
    this.cfg = { ...defaultCausalNetConfig, ...(config ?? {}) };
    this.optimizer = tf.train.adam(this.cfg.learningRate);
    this.fiberConsistencyWeight =
      fiberConfig?.fiberConsistencyWeight ??
      defaultGeometricPortConfig.fiberConsistencyWeight;
    this.fiberThreshold =
      fiberConfig?.fiberThreshold ?? defaultGeometricPortConfig.fiberThreshold;
  }

  /**
   * Build the neural network model.
   * Input: concatenation of state embedding and port embedding.
   * Output: predicted trajectory (delta).
   */
  private buildModel(inputDim: number, outputDim: number): tf.LayersModel {
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

    const output = tf.layers
      .dense({
        units: outputDim,
        kernelInitializer: "glorotNormal",
        name: "trajectory",
      })
      .apply(x) as tf.SymbolicTensor;

    return tf.model({ inputs: input, outputs: output });
  }

  private ensureModel(inputDim: number, outputDim: number): void {
    if (this.model) return;
    this.inputDim = inputDim;
    this.outputDim = outputDim;
    this.model = this.buildModel(inputDim, outputDim);
  }

  /**
   * Predict trajectory for a given state and port embedding.
   */
  predict(stateEmb: Vec, portEmb: Vec): Vec {
    const input = [...stateEmb, ...portEmb];
    const inputDim = input.length;

    // If no model yet, return zeros (will be created on first observe)
    if (!this.model || this.inputDim !== inputDim) {
      return new Array(this.outputDim ?? stateEmb.length).fill(0);
    }

    return tf.tidy(() => {
      const inputT = tf.tensor2d([input]);
      const output = this.model!.predict(inputT) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  /**
   * Add an observation to the training buffer.
   * Trains when buffer is full.
   */
  observe(stateEmb: Vec, portEmb: Vec, actualTrajectory: Vec): void {
    const input = [...stateEmb, ...portEmb];

    this.ensureModel(input.length, actualTrajectory.length);

    this.buffer.push({
      stateEmb: [...stateEmb],
      portEmb: [...portEmb],
      target: [...actualTrajectory],
    });

    if (this.buffer.length >= this.cfg.batchSize) {
      this.trainStep();
    }
  }

  /**
   * Train on buffered observations using MSE loss with fiber consistency regularization.
   *
   * Fiber consistency: If two state embeddings are close (within fiberThreshold),
   * their dynamics (targets) should also be close. This regularization pushes
   * the model to learn consistent behavior within fibers.
   */
  private trainStep(): void {
    if (!this.model || this.buffer.length === 0) return;

    const inputs = this.buffer.map((b) => [...b.stateEmb, ...b.portEmb]);
    const targets = this.buffer.map((b) => b.target);
    const stateEmbs = this.buffer.map((b) => b.stateEmb);

    const inputT = tf.tensor2d(inputs);
    const targetT = tf.tensor2d(targets);

    // Compute fiber consistency pairs once (outside TF graph for efficiency)
    const fiberPairs = this.findFiberPairs(stateEmbs);

    this.optimizer.minimize(() => {
      const pred = this.model!.predict(inputT) as tf.Tensor;
      let loss = tf.losses.meanSquaredError(targetT, pred) as tf.Scalar;

      // Add fiber consistency regularization if weight > 0 and we have pairs
      if (this.fiberConsistencyWeight > 0 && fiberPairs.length > 0) {
        const fiberLoss = this.computeFiberLoss(targets, fiberPairs);
        loss = loss.add(tf.scalar(this.fiberConsistencyWeight * fiberLoss));
      }

      return loss;
    });

    inputT.dispose();
    targetT.dispose();

    this.buffer = [];
  }

  /**
   * Find pairs of indices where state embeddings are within the fiber threshold.
   */
  private findFiberPairs(stateEmbs: Vec[]): Array<[number, number]> {
    const pairs: Array<[number, number]> = [];
    const n = stateEmbs.length;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dist = this.embeddingDistance(stateEmbs[i]!, stateEmbs[j]!);
        if (dist < this.fiberThreshold) {
          pairs.push([i, j]);
        }
      }
    }

    return pairs;
  }

  /**
   * Compute fiber consistency loss: MSE between targets that should be similar.
   */
  private computeFiberLoss(
    targets: Vec[],
    pairs: Array<[number, number]>
  ): number {
    if (pairs.length === 0) return 0;

    let totalLoss = 0;
    for (const [i, j] of pairs) {
      const t1 = targets[i]!;
      const t2 = targets[j]!;
      let pairLoss = 0;
      for (let k = 0; k < t1.length; k++) {
        pairLoss += (t1[k]! - t2[k]!) ** 2;
      }
      totalLoss += pairLoss / t1.length;
    }

    return totalLoss / pairs.length;
  }

  /**
   * Euclidean distance between two embeddings.
   */
  private embeddingDistance(a: Vec, b: Vec): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i]! - b[i]!) ** 2;
    }
    return Math.sqrt(sum);
  }

  /**
   * Force training on any remaining buffered observations.
   */
  flush(): void {
    if (this.buffer.length > 0) {
      this.trainStep();
    }
  }

  /**
   * Get diagnostic snapshot.
   */
  snapshot(): unknown {
    return {
      inputDim: this.inputDim,
      outputDim: this.outputDim,
      bufferSize: this.buffer.length,
      hasModel: this.model !== null,
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
    this.optimizer.dispose();
  }
}

