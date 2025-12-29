/**
 * TAM Composition API
 *
 * Framework for learning and composing world models
 * with automatic functor discovery and arbitrary composition graphs.
 */

import { GeometricPortBank } from "../geometric";
import type { Vec } from "../vec";
import type { Encoders, Situation } from "../types";
import type {
  FunctorDiscoveryConfig,
  FunctorResult,
  CompositionPath,
  PortMetadata,
  DomainSpec,
  TrainingConfig,
} from "./types";
import { defaultFunctorDiscoveryConfig, defaultTrainingConfig } from "./types";
import { PortRegistry } from "./registry";
import { FunctorCache } from "./cache";
import { CompositionGraph } from "./graph";
import { ComposedPort, createComposedPort } from "./composed";
import { discoverFunctor, TFFunctorNetwork } from "./functor";
import { sub } from "../vec";
import { LearnableEncoder } from "./encoder";

/**
 * Domain configuration for registration (internal use).
 * Use TAM.learn() for the simple API.
 */
export interface DomainConfig<S, C = unknown> {
  /** The trained port bank */
  port: GeometricPortBank<S, C>;

  /** Function to embed state into vector */
  embedder: (state: S) => Vec;

  /** Function to generate random states for training */
  randomState: () => S;

  /** Function to simulate one step */
  simulate: (state: S) => S;

  /** Embedding dimension */
  embeddingDim: number;
}

/**
 * Create default encoders from an embedder function.
 */
function createEncoders<S, C>(embedder: (s: S) => Vec): Encoders<S, C> {
  return {
    embedSituation: (sit: Situation<S, C>) => embedder(sit.state),
    delta: (before: Situation<S, C>, after: Situation<S, C>) =>
      sub(embedder(after.state), embedder(before.state)),
  };
}

/**
 * TAM: Main entry point for compositional world models.
 */
export class TAM {
  private registry: PortRegistry;
  private cache: FunctorCache;
  private graph: CompositionGraph;
  private domains = new Map<string, DomainConfig<unknown, unknown>>();
  private config: FunctorDiscoveryConfig;

  constructor(config: Partial<FunctorDiscoveryConfig> = {}) {
    this.config = { ...defaultFunctorDiscoveryConfig, ...config };
    this.registry = new PortRegistry();

    // Create cache with discovery function bound to this instance
    this.cache = new FunctorCache(
      this.performDiscovery.bind(this),
      this.config
    );

    this.graph = new CompositionGraph(this.registry, this.cache);
  }

  /**
   * Register a domain with its port and configuration.
   * For most cases, use learn() instead.
   */
  register<S, C = unknown>(name: string, config: DomainConfig<S, C>): void {
    this.registry.register(name, config.port, config.embeddingDim);
    this.domains.set(name, config as DomainConfig<unknown, unknown>);
  }

  /**
   * Learn a domain from a specification.
   * Creates and trains a port, then registers it.
   *
   * Two modes:
   * 1. Hand-crafted encoder: provide `embedder`
   * 2. Composition-learned encoder: provide `stateToRaw`, `rawDim`, and `composeWith`
   *
   * @param name - Unique name for the domain
   * @param spec - Domain specification (dynamics, embedder, etc.)
   * @param training - Training configuration (epochs, samples, etc.)
   * @returns The trained port bank (for inspection)
   */
  async learn<S, C = {}>(
    name: string,
    spec: DomainSpec<S, C>,
    training?: TrainingConfig
  ): Promise<GeometricPortBank<S, C>> {
    // Route to appropriate learning method
    if (spec.embedder) {
      const specWithEmbedder = spec as DomainSpec<S, C> & {
        embedder: (s: S) => Vec;
      };
      return this.learnWithEmbedder(name, specWithEmbedder, training);
    }

    if (spec.stateToRaw && spec.rawDim !== undefined && spec.composeWith) {
      const specWithComposition = spec as DomainSpec<S, C> & {
        stateToRaw: (s: S) => Vec;
        rawDim: number;
        composeWith: string[];
      };
      return this.learnWithComposition(name, specWithComposition, training);
    }

    throw new Error(
      "DomainSpec must provide either `embedder` or (`stateToRaw`, `rawDim`, and `composeWith`)"
    );
  }

  /**
   * Learn with a hand-crafted embedder.
   */
  private learnWithEmbedder<S, C = {}>(
    name: string,
    spec: DomainSpec<S, C> & { embedder: (s: S) => Vec },
    training?: TrainingConfig
  ): GeometricPortBank<S, C> {
    const config = { ...defaultTrainingConfig, ...training };
    const { epochs, samplesPerEpoch, flushFrequency } = config;

    // Create encoders from embedder if not provided
    const encoders = spec.encoders ?? createEncoders<S, C>(spec.embedder);

    // Create port bank
    const port = new GeometricPortBank<S, C>(encoders, spec.portConfig);

    // Context factory (defaults to empty object)
    const getContext = spec.context ?? (() => ({} as C));

    // Training loop
    for (let e = 0; e < epochs; e++) {
      for (let i = 0; i < samplesPerEpoch; i++) {
        const before = spec.randomState();
        const after = spec.simulate(before);
        const context = getContext();

        port.observe({
          before: { state: before, context },
          after: { state: after, context },
          action: "default",
        });
      }

      // Flush periodically to train networks
      if ((e + 1) % flushFrequency === 0) {
        port.flush();
      }
    }

    // Final flush
    port.flush();

    // Register with TAM
    this.register(name, {
      port,
      embedder: spec.embedder,
      randomState: spec.randomState,
      simulate: spec.simulate,
      embeddingDim: spec.embeddingDim,
    });

    return port;
  }

  /**
   * Learn with composition-based encoder learning.
   *
   * The encoder learns to represent the new domain in terms of
   * the dynamics of the specified target domains.
   */
  private async learnWithComposition<S, C = {}>(
    name: string,
    spec: DomainSpec<S, C> & {
      stateToRaw: (s: S) => Vec;
      rawDim: number;
      composeWith: string[];
    },
    training?: TrainingConfig
  ): Promise<GeometricPortBank<S, C>> {
    const config = { ...defaultTrainingConfig, ...training };

    // Validate targets exist
    const targetConfigs: Array<{
      name: string;
      config: DomainConfig<unknown, unknown>;
    }> = [];
    for (const targetName of spec.composeWith) {
      const targetConfig = this.domains.get(targetName);
      if (!targetConfig) {
        throw new Error(
          `Composition target "${targetName}" not registered. Register it first with learn().`
        );
      }
      targetConfigs.push({ name: targetName, config: targetConfig });
    }

    // Create learnable encoder
    const encoder = new LearnableEncoder(spec.rawDim, spec.embeddingDim);

    // Create functors for each target (mapping from new domain embedding to target)
    const functors: Array<{
      name: string;
      functor: TFFunctorNetwork;
      targetConfig: DomainConfig<unknown, unknown>;
    }> = [];

    for (const { name: targetName, config: targetConfig } of targetConfigs) {
      const functor = new TFFunctorNetwork(
        spec.embeddingDim,
        targetConfig.embeddingDim,
        this.config.hiddenSizes,
        this.config.learningRate
      );
      functors.push({ name: targetName, functor, targetConfig });
    }

    // Joint training loop
    const tolerance = 0.3;

    for (let epoch = 0; epoch < config.epochs; epoch++) {
      for (let i = 0; i < config.samplesPerEpoch; i++) {
        // Sample transition from new domain
        const before = spec.randomState();
        const after = spec.simulate(before);

        // Get raw features
        const rawBefore = spec.stateToRaw(before);
        const rawAfter = spec.stateToRaw(after);

        // Collect target deltas from each target domain
        const targetDeltas: Vec[] = [];

        for (const { targetConfig } of functors) {
          // Sample a paired transition from target domain
          const targetBefore = targetConfig.randomState();
          const targetAfter = targetConfig.simulate(targetBefore);

          // Compute target delta
          const targetBeforeEmb = targetConfig.embedder(targetBefore);
          const targetAfterEmb = targetConfig.embedder(targetAfter);
          const targetDelta = targetAfterEmb.map(
            (a, j) => a - targetBeforeEmb[j]!
          );

          targetDeltas.push(targetDelta);
        }

        // Train encoder to match average target delta structure
        encoder.trainStep(rawBefore, rawAfter, targetDeltas);

        // Also train the functors with current encoder output
        const embBefore = encoder.encode(rawBefore);
        const embAfter = encoder.encode(rawAfter);
        const embDelta = embAfter.map((a, j) => a - embBefore[j]!);

        for (const { functor, targetConfig } of functors) {
          // Get fresh target sample for functor training
          const targetBefore = targetConfig.randomState();
          const targetAfter = targetConfig.simulate(targetBefore);
          const targetBeforeEmb = targetConfig.embedder(targetBefore);
          const targetAfterEmb = targetConfig.embedder(targetAfter);
          const targetDelta = targetAfterEmb.map(
            (a, j) => a - targetBeforeEmb[j]!
          );

          functor.trainStep(embDelta, targetDelta, tolerance);
        }
      }
    }

    // Create fixed embedder from learned encoder
    const embedder = (s: S): Vec => encoder.encode(spec.stateToRaw(s));

    // Now train the port with the learned embedder
    const port = this.learnWithEmbedder(name, { ...spec, embedder }, training);

    // Clean up functors (they were just for encoder learning)
    for (const { functor } of functors) {
      functor.dispose();
    }

    // Keep encoder alive (it's captured in the embedder closure)
    // Note: In production, we'd want to manage encoder lifecycle

    return port;
  }

  /**
   * Find a composition path between two domains.
   * May trigger functor discovery for untested pairs.
   */
  async findPath(
    source: string,
    target: string,
    maxHops: number = 3
  ): Promise<CompositionPath | null> {
    return this.graph.findPath(source, target, maxHops);
  }

  /**
   * Compose a path into a usable ComposedPort.
   */
  compose(path: CompositionPath): ComposedPort {
    if (path.steps.length === 0) {
      throw new Error("Cannot compose empty path");
    }

    const sourceDomain = path.steps[0]!.from;
    const sourceConfig = this.domains.get(sourceDomain);
    if (!sourceConfig) {
      throw new Error(`Source domain "${sourceDomain}" not found`);
    }

    return createComposedPort(path, this.registry, sourceConfig.embedder);
  }

  /**
   * Convenience method: find path and compose in one step.
   */
  async getComposedPort(
    source: string,
    target: string,
    maxHops: number = 3
  ): Promise<ComposedPort | null> {
    const path = await this.findPath(source, target, maxHops);
    if (!path || path.steps.length === 0) {
      return null;
    }
    return this.compose(path);
  }

  /**
   * Retry functor discovery with different config.
   */
  async retryDiscovery(
    source: string,
    target: string,
    config: Partial<FunctorDiscoveryConfig>
  ): Promise<FunctorResult> {
    return this.cache.retry(source, target, config);
  }

  /**
   * Get cached functor result (without triggering discovery).
   */
  getCachedResult(source: string, target: string): FunctorResult | null {
    return this.cache.getCached(source, target);
  }

  /**
   * List all registered domains.
   */
  listDomains(): string[] {
    return this.registry.list();
  }

  /**
   * Get metadata for a domain.
   */
  getDomainMetadata(name: string): PortMetadata {
    return this.registry.getMetadata(name);
  }

  /**
   * Get all attempted domain pairs with their status.
   */
  getAttemptedPairs(): Array<{
    source: string;
    target: string;
    status: string;
  }> {
    return this.graph.getAttemptedPairs();
  }

  /**
   * Get failed discovery attempts for a pair.
   */
  getFailedAttempts(source: string, target: string): FunctorResult[] {
    return this.graph.getFailedAttempts(source, target);
  }

  /**
   * Get the successful composition graph.
   */
  getSuccessfulGraph(): Map<string, string[]> {
    return this.graph.getSuccessfulGraph();
  }

  /**
   * Dispose all resources.
   */
  dispose(): void {
    this.cache.clear();
    this.registry.clear();
    this.domains.clear();
  }

  /**
   * Internal: Perform functor discovery between two domains.
   */
  private async performDiscovery(
    source: string,
    target: string,
    config: Partial<FunctorDiscoveryConfig>
  ): Promise<FunctorResult> {
    const sourceConfig = this.domains.get(source);
    const targetConfig = this.domains.get(target);

    if (!sourceConfig) {
      return {
        status: "not_found",
        bindingRate: 0,
        epochs: 0,
        reason: "timeout",
        source,
        target,
      };
    }

    if (!targetConfig) {
      return {
        status: "not_found",
        bindingRate: 0,
        epochs: 0,
        reason: "timeout",
        source,
        target,
      };
    }

    return discoverFunctor(
      source,
      target,
      sourceConfig.randomState,
      sourceConfig.simulate,
      sourceConfig.embedder,
      targetConfig.randomState,
      targetConfig.simulate,
      targetConfig.embedder,
      config
    );
  }
}

// Re-export types and utilities
export * from "./types";
export { PortRegistry } from "./registry";
export { FunctorCache } from "./cache";
export { CompositionGraph } from "./graph";
export { ComposedPort, createComposedPort } from "./composed";
export { TFFunctorNetwork, discoverFunctor } from "./functor";
export { LearnableEncoder, defaultEncoderConfig } from "./encoder";
export type { EncoderConfig } from "./encoder";
