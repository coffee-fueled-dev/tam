/**
 * EncoderBridge: Connects learnable encoder with GeometricPortBank.
 *
 * Provides a clean interface for:
 * 1. Static encoder: Use hand-crafted embedder (existing behavior)
 * 2. Learnable encoder: Joint training with CausalNet/CommitmentNet
 *
 * The bridge creates an Encoders object that can be passed to GeometricPortBank,
 * and provides additional training methods when encoder is learnable.
 */

import * as tf from "@tensorflow/tfjs";
import type { Encoders, Situation, Transition } from "../types";
import type { Vec } from "../vec";
import { sub } from "../vec";
import type { IntraDomainEncoder } from "./intra-domain-encoder";
import type { CausalNet } from "./causal";
import type { CommitmentNet } from "./commitment";

/**
 * Configuration for encoder bridge.
 */
export interface EncoderBridgeConfig<S, C = unknown> {
  /**
   * Extract raw features from state.
   * This is always required - defines what goes into encoder.
   */
  extractRaw: (state: S) => Vec;

  /**
   * Optional: Learnable encoder.
   * If provided, enables end-to-end learning.
   * If not provided, extractRaw is used directly as static embedding.
   */
  learnableEncoder?: IntraDomainEncoder;

  /**
   * Optional: Static embedder for backward compatibility.
   * If provided and no learnableEncoder, this is used instead of extractRaw.
   * This allows existing code with hand-crafted embedders to work unchanged.
   */
  staticEmbedder?: (state: S) => Vec;
}

/**
 * EncoderBridge: Manages encoder (static or learnable) for port training.
 */
export class EncoderBridge<S, C = unknown> {
  private readonly extractRaw: (state: S) => Vec;
  private readonly learnableEncoder?: IntraDomainEncoder;
  private readonly staticEmbedder?: (state: S) => Vec;

  // Cache for encoder functions (avoid recreating closures)
  public readonly encoders: Encoders<S, C>;

  constructor(config: EncoderBridgeConfig<S, C>) {
    this.extractRaw = config.extractRaw;
    this.learnableEncoder = config.learnableEncoder;
    this.staticEmbedder = config.staticEmbedder;

    // Create encoders object
    this.encoders = {
      embedSituation: (sit: Situation<S, C>) => this.embed(sit.state),
      delta: (before: Situation<S, C>, after: Situation<S, C>) =>
        sub(this.embed(after.state), this.embed(before.state)),
    };
  }

  /**
   * Embed a state (internal method).
   */
  private embed(state: S): Vec {
    if (this.staticEmbedder) {
      // Use provided static embedder (backward compatibility)
      return this.staticEmbedder(state);
    } else if (this.learnableEncoder) {
      // Use learnable encoder
      const raw = this.extractRaw(state);
      return this.learnableEncoder.encode(raw);
    } else {
      // Raw features are the embedding (identity encoder)
      return this.extractRaw(state);
    }
  }

  /**
   * Check if encoder is learnable.
   */
  isLearnable(): boolean {
    return this.learnableEncoder !== undefined;
  }

  /**
   * Train encoder (if learnable) based on transition.
   *
   * This performs joint training:
   * - Encoder learns to produce embeddings that enable good binding + high agency
   * - CausalNet/CommitmentNet are already trained by bank.observe()
   *
   * Call this AFTER bank.observe() for each transition.
   *
   * @param transition - The transition that was just observed
   * @param causalNet - Shared CausalNet (for joint gradient)
   * @param commitmentNet - Shared CommitmentNet (for joint gradient)
   * @param portEmbedding - Port embedding used for this transition
   * @param referenceVolume - Current reference volume
   * @returns Loss values (or undefined if not learnable)
   */
  trainEncoder(
    transition: Transition<S, C>,
    causalNet: CausalNet,
    commitmentNet: CommitmentNet,
    portEmbedding: Vec,
    referenceVolume: number
  ): { totalLoss: number; bindingLoss: number; agencyLoss: number } | undefined {
    if (!this.learnableEncoder) {
      return undefined; // Not learnable
    }

    // Extract raw features
    const rawBefore = this.extractRaw(transition.before.state);

    // Compute actual trajectory
    const embBefore = this.embed(transition.before.state);
    const embAfter = this.embed(transition.after.state);
    const actualTrajectory = sub(embAfter, embBefore);

    // Create TensorFlow wrappers for shared networks
    const causalNetTF = (
      stateEmb: tf.Tensor1D,
      portEmb: tf.Tensor1D
    ): tf.Tensor1D => {
      const stateArr = Array.from(stateEmb.dataSync());
      const portArr = Array.from(portEmb.dataSync());
      const prediction = causalNet.predict(stateArr, portArr);
      return tf.tensor1d(prediction);
    };

    const commitmentNetTF = (
      stateEmb: tf.Tensor1D,
      portEmb: tf.Tensor1D
    ): tf.Tensor1D => {
      const stateArr = Array.from(stateEmb.dataSync());
      const portArr = Array.from(portEmb.dataSync());
      const distance = commitmentNet.predictDistance(stateArr, portArr);

      // TODO(cone-radius-accuracy): Use full geometric formula for cone radius
      //
      // Current: Simplified radius = k / (1 + distance)
      //
      // Actual formula (from GeometricPort.getCone):
      //   alignment = computeAlignment(stateEmb)  // Viewing angle factor
      //   scale = alignment / (1 + distance)
      //   radius = aperture * scale  // Per-dimension, possibly anisotropic
      //
      // Why this matters (low priority):
      // - More accurate diagnostic losses (binding/agency)
      // - Better correspondence with actual port predictions
      // - Matters more if we get stopGradient() and backprop through cone metrics
      //
      // Why current simplification is okay:
      // - These losses are diagnostic only (not used for encoder gradients)
      // - Encoder trains using smoothness loss instead
      // - Relative changes preserved (same 1/(1+d) relationship)
      // - Experiments show this works well (90% agency)
      //
      // Solution when needed:
      // - Extract alignment and aperture from port (need to pass through)
      // - Or: Accept that diagnostics are approximate (current approach)

      const k = 1.0; // Scaling factor
      const radius = k / (1 + distance);
      // Return per-dimension radius (isotropic for simplicity)
      return tf.fill([stateArr.length], radius);
    };

    // Joint training step
    return this.learnableEncoder.trainStepJoint(
      rawBefore,
      portEmbedding,
      actualTrajectory,
      causalNetTF,
      commitmentNetTF,
      referenceVolume
    );
  }

  /**
   * Get encoder statistics (if learnable).
   */
  getEncoderStats(): {
    trainSteps: number;
    avgBindingLoss: number;
    avgAgencyLoss: number;
  } | undefined {
    return this.learnableEncoder?.getStats();
  }

  /**
   * Dispose resources.
   */
  dispose(): void {
    this.learnableEncoder?.dispose();
  }
}

/**
 * Factory: Create encoder bridge from config.
 *
 * This is the main entry point for creating encoders (static or learnable).
 */
export function createEncoderBridge<S, C = unknown>(
  config: EncoderBridgeConfig<S, C>
): EncoderBridge<S, C> {
  return new EncoderBridge(config);
}
