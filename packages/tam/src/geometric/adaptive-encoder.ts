/**
 * AdaptiveEncoder: Co-learn encoder during port training via binding objective.
 *
 * Key idea: Apply functor inversion internally within a domain.
 * Instead of hand-crafting embedder, learn E such that:
 *   1. Binding succeeds: actual trajectory ∈ cone(π(p, E(x)))
 *   2. Agency is high: cone volume is small
 *
 * This is analogous to composition-based encoder learning, but the
 * "target" is the binding success objective rather than cross-domain functors.
 *
 * Training objective:
 *   L = L_causal + L_commitment + L_encoder
 *
 * Where:
 *   L_causal: MSE(π(p, E(x)), actual_trajectory)
 *   L_commitment: Asymmetric loss (narrow on success, widen on failure)
 *   L_encoder: Regularization to prevent collapse
 *
 * The encoder learns: "What representation makes binding easiest?"
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";

export interface AdaptiveEncoderConfig {
  /** Dimension of raw features */
  rawDim: number;
  /** Dimension of learned embedding */
  embeddingDim: number;
  /** Hidden layer sizes */
  hiddenSizes: number[];
  /** Learning rate */
  learningRate: number;
  /** Contrastive margin for preventing collapse */
  contrastiveMargin: number;
  /** Weight for contrastive loss */
  contrastiveWeight: number;
}

export const defaultAdaptiveEncoderConfig: AdaptiveEncoderConfig = {
  rawDim: 0, // Must be specified
  embeddingDim: 8,
  hiddenSizes: [32, 16],
  learningRate: 0.001, // Lower than port networks
  contrastiveMargin: 0.5,
  contrastiveWeight: 0.1,
};

/**
 * AdaptiveEncoder: Learns situation embedding during port training.
 *
 * Unlike composition encoder (which learns relative to other domains),
 * this learns relative to the binding success objective.
 */
export class AdaptiveEncoder {
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;
  private readonly cfg: AdaptiveEncoderConfig;

  // Track recent embeddings for contrastive loss
  private recentEmbeddings: Vec[] = [];
  private readonly maxRecentEmbeddings = 32;

  constructor(config: Partial<AdaptiveEncoderConfig> & { rawDim: number }) {
    if (!config.rawDim) {
      throw new Error("rawDim must be specified");
    }

    this.cfg = { ...defaultAdaptiveEncoderConfig, ...config };

    // Build encoder network: raw → embedding
    const input = tf.input({ shape: [this.cfg.rawDim] });
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

    // Output: learned embedding
    const output = tf.layers
      .dense({
        units: this.cfg.embeddingDim,
        kernelInitializer: "glorotNormal",
        name: "embedding",
      })
      .apply(x) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: output });
    this.optimizer = tf.train.adam(this.cfg.learningRate);
  }

  /**
   * Encode raw features into embedding.
   */
  encode(raw: Vec): Vec {
    return tf.tidy(() => {
      const input = tf.tensor2d([raw]);
      const output = this.model.predict(input) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  /**
   * Training step: optimize encoder based on binding feedback.
   *
   * The encoder learns to produce embeddings that:
   * 1. Help CausalNet predict accurately
   * 2. Enable narrow cones (high agency)
   * 3. Don't collapse (contrastive regularization)
   *
   * This is called AFTER CausalNet/CommitmentNet training, using the
   * gradient of binding success with respect to embedding.
   *
   * @param raw - Raw features
   * @param bindingSuccess - Whether binding succeeded
   * @param agency - Current agency for this situation
   * @param targetAgency - Target agency (encourage narrow cones)
   */
  trainStep(
    raw: Vec,
    bindingSuccess: boolean,
    agency: number,
    targetAgency: number = 0.8
  ): { loss: number } {
    let totalLoss = 0;

    tf.tidy(() => {
      const rawT = tf.tensor2d([raw]);

      this.optimizer.minimize(() => {
        const embedding = this.model.predict(rawT) as tf.Tensor;

        // Main loss: encourage embeddings that enable high agency
        // When binding succeeds, push toward target agency
        // When binding fails, this signals encoder needs adjustment
        let agencyLoss: tf.Tensor;
        if (bindingSuccess) {
          // Success: encourage high agency (narrow cones)
          const agencyDiff = targetAgency - agency;
          agencyLoss = tf.scalar(agencyDiff * agencyDiff);
        } else {
          // Failure: large penalty (encoder produced bad embedding)
          agencyLoss = tf.scalar(1.0);
        }

        // Contrastive loss: prevent collapse
        // Encourage this embedding to be distinct from recent embeddings
        let contrastiveLoss = tf.scalar(0);
        if (this.recentEmbeddings.length > 0) {
          // Sample a recent embedding
          const idx = Math.floor(Math.random() * this.recentEmbeddings.length);
          const otherEmb = this.recentEmbeddings[idx]!;
          const otherT = tf.tensor2d([otherEmb]);

          // Distance between embeddings
          const diff = embedding.sub(otherT);
          const distance = diff.norm();

          // Loss: penalize if embeddings are too close
          // margin - distance → 0 if distance > margin, positive otherwise
          const margin = this.cfg.contrastiveMargin;
          contrastiveLoss = tf.maximum(
            tf.scalar(0),
            tf.scalar(margin).sub(distance)
          );
        }

        // Combined loss
        const loss = agencyLoss.add(
          contrastiveLoss.mul(this.cfg.contrastiveWeight)
        );
        totalLoss = (loss as tf.Scalar).dataSync()[0]!;

        return loss as tf.Scalar;
      });
    });

    // Track this embedding for contrastive loss
    const embedding = this.encode(raw);
    this.recentEmbeddings.push(embedding);
    if (this.recentEmbeddings.length > this.maxRecentEmbeddings) {
      this.recentEmbeddings.shift();
    }

    return { loss: totalLoss };
  }

  /**
   * Alternative training: Backprop through full pipeline.
   *
   * This requires integration with CausalNet/CommitmentNet to compute
   * end-to-end gradient from binding loss back to encoder.
   *
   * Pseudocode:
   * ```
   * embedding = encoder(raw)
   * trajectory = causalNet(embedding, port)
   * tolerance = commitmentNet(embedding, port)
   * binding_loss = evaluate_binding(actual, trajectory, tolerance)
   * encoder_grad = backprop(binding_loss, encoder.params)
   * ```
   *
   * This is more principled but requires tighter integration.
   */
  // TODO: Implement end-to-end backprop version

  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}

/**
 * Example usage pattern:
 *
 * ```typescript
 * // Create encoder for raw features
 * const encoder = new AdaptiveEncoder({
 *   rawDim: 10,  // Raw feature dimension
 *   embeddingDim: 8,
 * });
 *
 * // Create encoders using adaptive encoder
 * const encoders: Encoders<State, Context> = {
 *   embedSituation: (sit) => encoder.encode(extractRaw(sit.state)),
 *   delta: (before, after) => {
 *     const embBefore = encoder.encode(extractRaw(before.state));
 *     const embAfter = encoder.encode(extractRaw(after.state));
 *     return sub(embAfter, embBefore);
 *   },
 * };
 *
 * // Create port bank
 * const bank = new GeometricPortBank(encoders, config);
 *
 * // Training loop with encoder adaptation
 * for (const transition of transitions) {
 *   // 1. Get prediction before observation
 *   const predictions = bank.predict(transition.action, transition.before);
 *   const agencyBefore = predictions[0]?.agency ?? 0;
 *
 *   // 2. Observe (trains CausalNet/CommitmentNet)
 *   await bank.observe(transition);
 *
 *   // 3. Get feedback after observation
 *   const agencyAfter = bank.predict(transition.action, transition.before)[0]?.agency ?? 0;
 *   const bindingSuccess = agencyAfter > agencyBefore; // Proxy for success
 *
 *   // 4. Train encoder based on binding feedback
 *   const raw = extractRaw(transition.before.state);
 *   encoder.trainStep(raw, bindingSuccess, agencyAfter);
 * }
 * ```
 *
 * Benefits:
 * - No hand-crafted embedder needed
 * - Encoder adapts to what port needs
 * - Discovers representations that maximize binding success
 * - Co-evolution of encoder and port
 *
 * Challenges:
 * - Need regularization to prevent collapse
 * - Training may be less stable initially
 * - Requires careful learning rate tuning (encoder slower than port)
 * - May need curriculum (start with fixed encoder, gradually adapt)
 */
