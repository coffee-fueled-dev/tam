/**
 * ComposedPort: A port that chains predictions through multiple domains.
 *
 * Applies functors sequentially to transform state embeddings
 * from source domain through intermediates to target domain.
 */

import type { Vec } from "../vec";
import type { CompositionPath, CompositionStep } from "./types";
import type { PortRegistry } from "./registry";
import type { GeometricPortBank } from "../geometric";
import type { Cone } from "../types";

/**
 * Prediction result from a composed port.
 */
export interface ComposedPrediction {
  /** Predicted delta in the target domain */
  delta: Vec;

  /** Affordance cone in the target domain */
  cone: Cone;

  /** Intermediate embeddings at each step */
  intermediates: Array<{ domain: string; embedding: Vec }>;

  /** Combined binding rate across all steps */
  composedBindingRate: number;
}

/**
 * A port that chains through multiple domains via functors.
 */
export class ComposedPort {
  readonly path: CompositionPath;
  private registry: PortRegistry;
  private sourceEmbedder: (state: unknown) => Vec;
  private targetPortBank: GeometricPortBank<unknown, unknown>;

  constructor(
    path: CompositionPath,
    registry: PortRegistry,
    sourceEmbedder: (state: unknown) => Vec,
    targetPortBank: GeometricPortBank<unknown, unknown>
  ) {
    this.path = path;
    this.registry = registry;
    this.sourceEmbedder = sourceEmbedder;
    this.targetPortBank = targetPortBank;
  }

  /**
   * Predict in the target domain from a source domain input.
   *
   * @param sourceState - State in the source domain
   * @returns Prediction in the target domain
   */
  predict(sourceState: unknown): ComposedPrediction {
    // Start with source embedding
    let currentEmbedding = this.sourceEmbedder(sourceState);
    const intermediates: Array<{ domain: string; embedding: Vec }> = [
      { domain: this.path.steps[0]?.from ?? "source", embedding: currentEmbedding },
    ];

    // Apply each functor in sequence
    for (const step of this.path.steps) {
      currentEmbedding = step.functor.apply(currentEmbedding);
      intermediates.push({ domain: step.to, embedding: currentEmbedding });
    }

    // Get prediction from target port using the transformed embedding directly
    // This is the key fix: predict() now operates in embedding space!
    const targetAction = "default"; // Could be parameterized
    const predictions = this.targetPortBank.predict(targetAction, currentEmbedding, 1);

    if (predictions.length === 0) {
      // No applicable port - return zero prediction with empty cone
      return {
        delta: currentEmbedding.map(() => 0),
        cone: { center: currentEmbedding.map(() => 0), radius: currentEmbedding.map(() => 0) },
        intermediates,
        composedBindingRate: this.path.totalBindingRate,
      };
    }

    const prediction = predictions[0]!;

    return {
      delta: prediction.delta,
      cone: prediction.cone!,
      intermediates,
      composedBindingRate: this.path.totalBindingRate,
    };
  }

  /**
   * Get the description of the composition path.
   */
  describe(): string {
    return this.path.describe();
  }

  /**
   * Get the total binding rate across all steps.
   */
  get totalBindingRate(): number {
    return this.path.totalBindingRate;
  }

  /**
   * Get the number of hops in the composition.
   */
  get hopCount(): number {
    return this.path.steps.length;
  }

  /**
   * Check if this is a direct composition (single hop).
   */
  get isDirect(): boolean {
    return this.path.steps.length === 1;
  }
}

/**
 * Factory function to create a ComposedPort from a path.
 */
export function createComposedPort(
  path: CompositionPath,
  registry: PortRegistry,
  sourceEmbedder: (state: unknown) => Vec
): ComposedPort {
  if (path.steps.length === 0) {
    throw new Error("Cannot create ComposedPort from empty path");
  }

  const targetDomain = path.steps[path.steps.length - 1]!.to;
  const targetPortBank = registry.get(targetDomain);

  return new ComposedPort(path, registry, sourceEmbedder, targetPortBank);
}

