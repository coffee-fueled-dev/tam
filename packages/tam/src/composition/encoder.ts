/**
 * LearnableEncoder: Composition-aware encoder learning.
 *
 * Learns to encode raw state features into an embedding space
 * that maximizes binding success with target domains.
 *
 * The encoder is learned relative to a basis of known actors -
 * it represents the new domain in terms of existing dynamics.
 *
 * Future Optimizations (when scaling to 15+ targets):
 * 1. Weighted targets: Learn importance weights per target domain
 * 2. Automatic pruning: Drop targets with consistently 0% binding
 * 3. Curriculum learning: Start with easiest target, add more as encoder stabilizes
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";
import type { GeometricPortBank } from "../geometric";
import type { FunctorNetwork } from "./types";

export interface EncoderConfig {
  /** Hidden layer sizes for the encoder MLP */
  hiddenSizes: number[];
  /** Learning rate for optimizer */
  learningRate: number;
  /** L2 regularization weight */
  l2Weight: number;
}

export const defaultEncoderConfig: EncoderConfig = {
  hiddenSizes: [32, 16],
  learningRate: 0.01,
  l2Weight: 0.001,
};

/**
 * Target domain info for training.
 */
export interface EncoderTarget<S> {
  /** Registered name of the target domain */
  name: string;
  /** The trained port bank */
  port: GeometricPortBank<S, unknown>;
  /** Functor from new domain embedding to this target */
  functor: FunctorNetwork;
  /** Embedder for the target domain */
  embedder: (state: S) => Vec;
}

/**
 * LearnableEncoder: Neural network that learns state → embedding mapping
 * by optimizing for composition success with target domains.
 */
export class LearnableEncoder {
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;
  private readonly config: EncoderConfig;

  public readonly rawDim: number;
  public readonly embeddingDim: number;

  constructor(
    rawDim: number,
    embeddingDim: number,
    config: Partial<EncoderConfig> = {}
  ) {
    this.rawDim = rawDim;
    this.embeddingDim = embeddingDim;
    this.config = { ...defaultEncoderConfig, ...config };

    // Build MLP: raw features → embedding
    const input = tf.input({ shape: [rawDim] });
    let x: tf.SymbolicTensor = input;

    for (const units of this.config.hiddenSizes) {
      x = tf.layers
        .dense({
          units,
          activation: "relu",
          kernelInitializer: "heNormal",
          kernelRegularizer: tf.regularizers.l2({
            l2: this.config.l2Weight,
          }),
        })
        .apply(x) as tf.SymbolicTensor;
    }

    // Output layer: no activation, direct embedding
    const output = tf.layers
      .dense({
        units: embeddingDim,
        kernelInitializer: "glorotNormal",
      })
      .apply(x) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: output });
    this.optimizer = tf.train.adam(this.config.learningRate);
  }

  /**
   * Encode raw state features into embedding.
   */
  encode(raw: Vec): Vec {
    return tf.tidy(() => {
      const input = tf.tensor2d([raw]);
      const output = this.model.predict(input) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  /**
   * Encode a batch of raw features.
   */
  encodeBatch(raws: Vec[]): Vec[] {
    return tf.tidy(() => {
      const input = tf.tensor2d(raws);
      const output = this.model.predict(input) as tf.Tensor;
      const data = output.arraySync() as number[][];
      return data;
    });
  }

  /**
   * Train step: optimize encoder to maximize binding success across targets.
   *
   * Uses a proxy loss: train the encoder to produce embeddings where
   * the delta matches the structure of target deltas (after functor).
   * We train by computing MSE between embedding delta and target delta
   * directly, since backprop through the functor breaks the TF graph.
   *
   * @param rawBefore - Raw features of state before transition
   * @param rawAfter - Raw features of state after transition
   * @param targetDeltas - Target deltas to match
   * @returns Object with loss value
   */
  trainStep(
    rawBefore: Vec,
    rawAfter: Vec,
    targetDeltas: Vec[]
  ): { loss: number } {
    let totalLoss = 0;

    tf.tidy(() => {
      const rawBeforeT = tf.tensor2d([rawBefore]);
      const rawAfterT = tf.tensor2d([rawAfter]);

      // Stack all target deltas into a single tensor for comparison
      // We average across targets
      const avgTargetDelta = targetDeltas.reduce(
        (acc, d) => acc.map((v, i) => v + d[i]! / targetDeltas.length),
        new Array(this.embeddingDim).fill(0) as number[]
      );
      const targetT = tf.tensor2d([avgTargetDelta]);

      this.optimizer.minimize(() => {
        // Encode both states
        const embBefore = this.model.predict(rawBeforeT) as tf.Tensor;
        const embAfter = this.model.predict(rawAfterT) as tf.Tensor;

        // Compute delta in embedding space
        const embDelta = embAfter.sub(embBefore);

        // MSE loss: embedding delta should match average target delta
        const mse = tf.losses.meanSquaredError(targetT, embDelta);
        totalLoss = (mse as tf.Scalar).dataSync()[0]!;

        return mse as tf.Scalar;
      });
    });

    return { loss: totalLoss };
  }

  /**
   * Evaluate binding success for current encoder state.
   */
  evaluateBinding(
    rawBefore: Vec,
    rawAfter: Vec,
    targets: Array<{
      functor: FunctorNetwork;
      targetDelta: Vec;
      tolerance: number;
    }>
  ): boolean[] {
    const embBefore = this.encode(rawBefore);
    const embAfter = this.encode(rawAfter);
    const embDelta = embAfter.map((a, i) => a - embBefore[i]!);

    return targets.map((target) => {
      const transformedDelta = target.functor.apply(embDelta);
      const error = Math.sqrt(
        transformedDelta.reduce(
          (s, p, i) => s + (p - target.targetDelta[i]!) ** 2,
          0
        ) / transformedDelta.length
      );
      return error < target.tolerance;
    });
  }

  /**
   * Get the model's trainable weights (for inspection).
   */
  getWeights(): tf.Tensor[] {
    return this.model.getWeights();
  }

  /**
   * Dispose of resources.
   */
  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}
