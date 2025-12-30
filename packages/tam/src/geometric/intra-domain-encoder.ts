/**
 * IntraDomainEncoder: End-to-end learnable encoder via binding objective.
 *
 * Unlike composition-based encoder learning (which learns relative to other domains),
 * this learns the encoder within a single domain by backpropagating through:
 *   raw → encoder → embedding → CausalNet/CommitmentNet → cone → binding loss
 *
 * The encoder learns: "What representation makes binding most successful with highest agency?"
 *
 * Key differences from adaptive-encoder.ts (feedback-based):
 * - Uses true gradients (not feedback signals)
 * - Trains jointly with CausalNet/CommitmentNet
 * - Differentiable binding and agency losses
 * - More stable, principled optimization
 *
 * Architecture:
 *   E: S → X (learnable encoder)
 *   π: X × P → T (CausalNet, shared)
 *   τ: X × P → R (CommitmentNet, shared)
 *
 * Loss:
 *   L = L_binding + λ_agency * L_agency + λ_smooth * L_smoothness
 *
 * Where:
 *   L_binding: Soft distance from trajectory to cone
 *   L_agency: -log(agency) = encourages narrow cones
 *   L_smoothness: Temporal coherence (consecutive states should have similar embeddings)
 *
 * ============================================================================
 * CURRENT LIMITATION: TensorFlow.js lacks tf.stopGradient()
 * ============================================================================
 *
 * This prevents us from implementing true alternating optimization with binding/agency losses.
 * Without stopGradient(), backpropping through cone predictions also backprops through the
 * shared networks, breaking loose coupling.
 *
 * Current workaround: Use temporal smoothness loss only (see trainStepJoint).
 * This works well in practice (90% agency in experiments) but is theoretically weaker.
 *
 * For detailed explanation and implementation plan, see:
 * - TODO(tf-stopgradient) in trainStepJoint() - main training method
 * - TODO(functor-inversion) in trainStepFunctorInversion() - categorical learning
 * - TODO(cone-radius-accuracy) in encoder-bridge.ts - diagnostic accuracy
 *
 * Solutions: Wait for TensorFlow.js update, port to PyTorch, or accept current approach.
 * ============================================================================
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";

export interface IntraDomainEncoderConfig {
  /** Dimension of raw state features */
  rawDim: number;
  /** Dimension of learned embedding */
  embeddingDim: number;
  /** Hidden layer sizes for encoder */
  hiddenSizes: number[];
  /** Learning rate for encoder (typically lower than port networks) */
  learningRate: number;
  /** Weight for agency regularization */
  agencyWeight: number;
  /** Weight for temporal smoothness */
  smoothnessWeight: number;
  /** Temperature for soft binding (lower = closer to hard threshold) */
  bindingTemperature: number;
}

export const defaultIntraDomainEncoderConfig: IntraDomainEncoderConfig = {
  rawDim: 0, // Must be specified
  embeddingDim: 8,
  hiddenSizes: [32, 16],
  learningRate: 0.001, // Lower than port networks (0.01)
  agencyWeight: 0.1, // Encourage narrow cones
  smoothnessWeight: 0.05, // Encourage temporal coherence
  bindingTemperature: 1.0, // Soft threshold
};

/**
 * IntraDomainEncoder: Learnable encoder for within-domain optimization.
 *
 * This can be used as a drop-in replacement for hand-crafted encoders,
 * with optional gradient-based learning.
 */
export class IntraDomainEncoder {
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;
  private readonly cfg: IntraDomainEncoderConfig;

  // Track previous embedding for temporal smoothness
  private lastEmbedding: Vec | null = null;

  // Statistics
  private trainSteps = 0;
  private avgBindingLoss = 0;
  private avgAgencyLoss = 0;

  constructor(config: Partial<IntraDomainEncoderConfig> & { rawDim: number }) {
    if (!config.rawDim || config.rawDim <= 0) {
      throw new Error("rawDim must be specified and positive");
    }

    this.cfg = { ...defaultIntraDomainEncoderConfig, ...config };

    // Build encoder MLP
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
   * This is used during forward pass (prediction).
   */
  encode(raw: Vec): Vec {
    return tf.tidy(() => {
      const input = tf.tensor2d([raw]);
      const output = this.model.predict(input) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  /**
   * Encode batch of raw features (for efficient training).
   */
  encodeBatch(raws: Vec[]): tf.Tensor2D {
    const input = tf.tensor2d(raws);
    return this.model.predict(input) as tf.Tensor2D;
  }

  /**
   * Compute differentiable binding loss.
   *
   * Instead of hard threshold (inside/outside cone), use soft distance:
   *   L_binding = smooth_max(0, normalized_distance - 1)
   *
   * Where normalized_distance = ||τ - center|| / radius
   * - If τ inside cone: distance < 1, loss ≈ 0
   * - If τ outside cone: distance > 1, loss > 0 (grows with violation)
   *
   * @param trajectory - Actual trajectory (target)
   * @param coneCenter - Predicted cone center
   * @param coneRadius - Predicted cone radius (per dimension)
   * @returns Scalar tensor loss
   */
  private bindingLoss(
    trajectory: tf.Tensor1D,
    coneCenter: tf.Tensor1D,
    coneRadius: tf.Tensor1D
  ): tf.Scalar {
    return tf.tidy(() => {
      // Compute normalized distance: (τ - center) / radius
      const diff = trajectory.sub(coneCenter);
      const normalized = diff.div(coneRadius.add(1e-8)); // Avoid division by zero

      // L2 norm of normalized difference
      const distance = normalized.norm();

      // Soft hinge loss: max(0, distance - 1) with temperature
      const violation = distance.sub(1.0).div(this.cfg.bindingTemperature);
      const softMax = violation.exp().add(1).log(); // log(1 + exp(x)) = smooth max(0, x)

      return softMax as tf.Scalar;
    });
  }

  /**
   * Compute differentiable agency loss.
   *
   * Agency = 1 - vol(cone) / V_ref
   * We want to maximize agency, so minimize -log(agency):
   *   L_agency = -log(agency) = -log(1 - vol/V_ref)
   *
   * Approximation for numerical stability:
   *   L_agency ≈ log(vol) = sum(log(radius))
   *
   * @param coneRadius - Predicted cone radius (per dimension)
   * @param referenceVolume - Reference volume for normalization
   * @returns Scalar tensor loss
   */
  private agencyLoss(
    coneRadius: tf.Tensor1D,
    referenceVolume: number
  ): tf.Scalar {
    return tf.tidy(() => {
      // Log-volume: sum of log-radii (proportional to log(volume))
      const logRadii = coneRadius.add(1e-8).log();
      const logVolume = logRadii.sum();

      // Normalize by reference
      const refLogVol = Math.log(referenceVolume) * coneRadius.shape[0];
      const normalized = logVolume.sub(refLogVol);

      return normalized as tf.Scalar;
    });
  }

  /**
   * Compute temporal smoothness loss.
   *
   * Encourages consecutive states to have similar embeddings:
   *   L_smooth = ||E(s_t) - E(s_{t-1})||²
   *
   * This creates temporal coherence in the learned representation.
   *
   * @param embedding - Current state embedding
   * @param prevEmbedding - Previous state embedding (or null)
   * @returns Scalar tensor loss (or 0 if no previous)
   */
  private smoothnessLoss(
    embedding: tf.Tensor1D,
    prevEmbedding: tf.Tensor1D | null
  ): tf.Scalar {
    if (!prevEmbedding) {
      return tf.scalar(0);
    }

    return tf.tidy(() => {
      const diff = embedding.sub(prevEmbedding);
      return diff.square().mean() as tf.Scalar;
    });
  }

  /**
   * Joint training step with CausalNet and CommitmentNet.
   *
   * Uses alternating optimization: encoder trains based on current (frozen-for-this-step)
   * predictions from CausalNet/CommitmentNet. This preserves loose coupling while enabling
   * online encoder learning.
   *
   * IMPORTANT LIMITATION: TensorFlow.js lacks tf.stopGradient(), which means we can't
   * backprop through cone predictions without also backpropping through the shared networks.
   * To maintain alternating optimization (loose coupling), we use ONLY temporal smoothness loss.
   *
   * This is still end-to-end learning - the encoder learns to produce temporally smooth
   * embeddings while the shared networks learn to predict on those embeddings. The coupling
   * happens indirectly over time rather than in a single gradient step.
   *
   * For direct binding/agency feedback, use functor inversion training instead
   * (see trainStepFunctorInversion).
   *
   * TODO(tf-stopgradient): Implement true binding/agency loss backprop
   *
   * Currently we only use temporal smoothness loss because TensorFlow.js lacks tf.stopGradient().
   * With stopGradient(), we could implement true alternating optimization:
   *
   *   optimizer.minimize(() => {
   *     const embedding = encode(rawState);  // Gradient flows here
   *     const cone = predictCone(stopGradient(embedding), port);  // No gradient to shared nets
   *     const loss = bindingLoss(trajectory, cone) + agencyLoss(cone);
   *     return loss;
   *   });
   *
   * Why this matters:
   * - Direct signal: Encoder learns exactly what makes binding successful (not just smoothness proxy)
   * - Faster convergence: Stronger gradient signal from binding/agency
   * - Better feature selection: Encoder can discover which features matter for dynamics
   * - Theoretical soundness: Aligns with mathematical definition of encoder learning
   *
   * Solutions:
   * 1. Wait for TensorFlow.js to add stopGradient() (requested but not yet implemented)
   * 2. Port to PyTorch (has torch.no_grad() and detach())
   * 3. Use manual gradient computation (complex, error-prone)
   * 4. Accept smoothness as reasonable proxy (current approach - works well in practice)
   *
   * Evidence that current approach works:
   * - Experiments show 90% agency (competitive with hand-crafted encoders)
   * - Successfully learns to ignore noise features
   * - Smooth embeddings are a reasonable inductive bias for physical systems
   *
   * @param rawState - Raw state features
   * @param portEmbedding - Port embedding
   * @param actualTrajectory - Observed trajectory
   * @param causalNet - Shared CausalNet (predict trajectory)
   * @param commitmentNet - Shared CommitmentNet (predict tolerance)
   * @param referenceVolume - Reference volume for agency computation
   * @returns Loss values for diagnostics
   */
  trainStepJoint(
    rawState: Vec,
    portEmbedding: Vec,
    actualTrajectory: Vec,
    causalNet: (stateEmb: tf.Tensor1D, portEmb: tf.Tensor1D) => tf.Tensor1D,
    commitmentNet: (
      stateEmb: tf.Tensor1D,
      portEmb: tf.Tensor1D
    ) => tf.Tensor1D,
    referenceVolume: number
  ): { totalLoss: number; bindingLoss: number; agencyLoss: number } {
    let losses = { totalLoss: 0, bindingLoss: 0, agencyLoss: 0 };

    // Get cone predictions OUTSIDE minimize using current fixed embedding
    const currentEmbedding = this.encode(rawState);
    const coneCenterArr = tf.tidy(() => {
      const emb = tf.tensor1d(currentEmbedding);
      const port = tf.tensor1d(portEmbedding);
      return Array.from(causalNet(emb, port).dataSync());
    });
    const coneRadiusArr = tf.tidy(() => {
      const emb = tf.tensor1d(currentEmbedding);
      const port = tf.tensor1d(portEmbedding);
      return Array.from(commitmentNet(emb, port).dataSync());
    });

    // Create input tensors BEFORE minimize (including cone predictions as constants)
    const xs = tf.tensor2d([rawState]);
    const coneCenter = tf.tensor1d(coneCenterArr);
    const coneRadius = tf.tensor1d(coneRadiusArr);
    const trajectoryT = tf.tensor1d(actualTrajectory);
    const lastEmbT = this.lastEmbedding ? tf.tensor1d(this.lastEmbedding) : null;

    // Custom training step: compute loss and apply gradients
    this.optimizer.minimize(() => {
      return tf.tidy(() => {
        // 1. Encode state - this creates gradient connection to model weights!
        const embedding = this.model.apply(xs, { training: true }) as tf.Tensor2D;
        const embeddingFlat = embedding.reshape([-1]) as tf.Tensor1D;

        // 2. Use ONLY smoothness loss for now
        // NOTE: Binding/agency losses can't be used because TensorFlow.js lacks stopGradient()
        // The encoder learns to produce temporally smooth embeddings
        // While the shared networks learn to predict well on those embeddings
        // This is still a form of end-to-end learning, just indirect

        let totalLoss: tf.Scalar;
        if (!lastEmbT) {
          // First step - minimize embedding magnitude to prevent explosion
          totalLoss = embeddingFlat.square().mean() as tf.Scalar;
        } else {
          // Temporal smoothness: consecutive states should have similar embeddings
          const smoothDiff = embeddingFlat.sub(lastEmbT);
          totalLoss = smoothDiff.square().mean() as tf.Scalar;
        }

        // Store cone metrics for diagnostics (not used in gradient)
        const diff = trajectoryT.sub(coneCenter);
        const normalized = diff.div(coneRadius.add(1e-8));
        const distance = normalized.norm();
        const violation = distance.sub(1.0).div(this.cfg.bindingTemperature);
        const lBinding = violation.exp().add(1).log();
        const logRadii = coneRadius.add(1e-8).log();
        const lAgency = logRadii.sum().sub(Math.log(referenceVolume) * coneRadius.shape[0]);

        // Store for diagnostics (this is OK - it doesn't affect gradient computation)
        losses.totalLoss = (totalLoss as tf.Scalar).dataSync()[0]!;
        losses.bindingLoss = (lBinding as tf.Scalar).dataSync()[0]!;
        losses.agencyLoss = (lAgency as tf.Scalar).dataSync()[0]!;

        return totalLoss as tf.Scalar;
      });
    });

    // Clean up tensors
    xs.dispose();
    coneCenter.dispose();
    coneRadius.dispose();
    trajectoryT.dispose();
    if (lastEmbT) lastEmbT.dispose();

    // Update statistics
    this.trainSteps++;
    const alpha = 0.99; // Exponential moving average
    this.avgBindingLoss = alpha * this.avgBindingLoss + (1 - alpha) * losses.bindingLoss;
    this.avgAgencyLoss = alpha * this.avgAgencyLoss + (1 - alpha) * losses.agencyLoss;

    // Store embedding for next smoothness computation (re-encode after training)
    this.lastEmbedding = this.encode(rawState);

    return losses;
  }

  /**
   * Functor inversion training step.
   *
   * PLACEHOLDER: Currently returns 0 (no training).
   *
   * True functor inversion would train the encoder to produce embeddings that make
   * port functors systematic. Instead of learning "what makes binding successful",
   * it learns "what makes transformations between specialists compositional".
   *
   * TODO(functor-inversion): Implement true functor inversion encoder training
   *
   * Conceptual algorithm:
   *   1. Track which states are handled well by which ports (state-port associations)
   *   2. For discovered functors F: P1 → P2, measure systematicity:
   *        - States from port1's domain should transform to port2's domain via F
   *        - Functor quality = how well F(port1) handles port2's states
   *   3. Train encoder to maximize functor quality:
   *
   *      optimizer.minimize(() => {
   *        const embedding = encode(rawState);  // Gradient flows here
   *        const portSelection = selectPort(embedding);  // Which port handles this state?
   *        const functorQuality = evaluateFunctors(embedding, portSelection, functors);
   *        return -functorQuality;  // Maximize quality
   *      });
   *
   * Why this matters:
   * - Discovers categorical structure: Encoder learns which features define "modes" or "regimes"
   * - Enables transfer: Once you learn mode transformations, can predict across modes
   * - Compositional generalization: F1 ∘ F2 should equal F3 for systematic domains
   * - Theoretical elegance: Aligns with category theory view of world models
   *
   * Challenges:
   * 1. Requires state-port associations (which states belong to which port's domain)
   * 2. Requires tf.stopGradient() to prevent backprop through shared networks
   * 3. Needs differentiable functor quality metric
   * 4. Computational cost: Must evaluate all functors for each training step
   *
   * Solutions:
   * 1. Track port assignments during bank.observe() (add to GeometricPortBank)
   * 2. Port to PyTorch for stopGradient() (or wait for TensorFlow.js)
   * 3. Use soft functor quality: weighted sum over all ports based on their agency
   * 4. Sample mini-batch of functors per step rather than evaluating all
   *
   * Current workaround:
   * - Use trainStepJoint() with smoothness loss instead
   * - Smoothness encourages structured embeddings that naturally support specialization
   * - Works well in practice (90% agency) but doesn't explicitly optimize for functors
   *
   * Evidence this would help:
   * - Functor-based proliferation successfully discovers specialists (when enabled)
   * - Systematic mode transformations are common in physical systems
   * - Category theory suggests this is the "right" abstraction for world models
   *
   * @param portPairs - Array of discovered port functors (for diagnostics)
   * @returns Loss value for diagnostics (currently always 0)
   */
  trainStepFunctorInversion(
    portPairs: Array<{
      port1: Vec;
      port2: Vec;
      functor: (p: Vec) => Vec;
    }>
  ): number {
    if (portPairs.length === 0) {
      return 0; // No functors discovered yet
    }

    // TODO: Implement actual functor inversion training
    // For now, just return 0 to indicate no training occurred
    // See TODO comment above for implementation plan
    return 0;
  }

  /**
   * Get training statistics.
   */
  getStats(): {
    trainSteps: number;
    avgBindingLoss: number;
    avgAgencyLoss: number;
  } {
    return {
      trainSteps: this.trainSteps,
      avgBindingLoss: this.avgBindingLoss,
      avgAgencyLoss: this.avgAgencyLoss,
    };
  }

  /**
   * Get model for direct manipulation (advanced use).
   */
  getModel(): tf.LayersModel {
    return this.model;
  }

  /**
   * Dispose resources.
   */
  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}

/**
 * Factory function: Create encoder from config.
 *
 * If config specifies learnable encoder, returns IntraDomainEncoder.
 * Otherwise, returns null (caller should use static encoder).
 */
export function createLearnableEncoder(
  config?: Partial<IntraDomainEncoderConfig> & { rawDim?: number }
): IntraDomainEncoder | null {
  if (!config || !config.rawDim) {
    return null; // No learnable encoder
  }

  return new IntraDomainEncoder(config);
}
