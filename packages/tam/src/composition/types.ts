/**
 * Types for the TAM Composition API.
 *
 * Enables compositional world models through functor discovery
 * and arbitrary composition graphs.
 */

import type { Vec } from "../vec";
import type { Encoders, GeometricPortConfigInput } from "../types";

// ============================================================================
// Domain Learning Types
// ============================================================================

/**
 * Configuration for domain training.
 */
export interface TrainingConfig {
  /** Number of training epochs (default: 100) */
  epochs?: number;

  /** Samples to observe per epoch (default: 50) */
  samplesPerEpoch?: number;

  /** How often to flush/train neural nets, in epochs (default: 20) */
  flushFrequency?: number;
}

export const defaultTrainingConfig: Required<TrainingConfig> = {
  epochs: 100,
  samplesPerEpoch: 50,
  flushFrequency: 20,
};

/**
 * Domain specification for TAM.learn().
 * Defines a domain's dynamics, encoding, and optional configuration.
 *
 * Two modes:
 * 1. Hand-crafted encoder: provide `embedder`
 * 2. Composition-learned encoder: provide `stateToRaw`, `rawDim`, and `composeWith`
 */
export interface DomainSpec<S, C = {}> {
  /** Generate a random state for training */
  randomState: () => S;

  /** Simulate one step of dynamics */
  simulate: (state: S) => S;

  /** Dimension of state embeddings */
  embeddingDim: number;

  /** Generate context (optional, defaults to {} as C) */
  context?: () => C;

  /** Custom encoders (optional, auto-generated from embedder if not provided) */
  encoders?: Encoders<S, C>;

  /** Port configuration overrides (optional) */
  portConfig?: GeometricPortConfigInput;

  // ---- Option A: Hand-crafted encoder ----

  /** Embed a state into a vector (required if not using composition-learning) */
  embedder?: (state: S) => Vec;

  // ---- Option B: Composition-learned encoder ----

  /** Extract raw numerical features from state (required for composition-learning) */
  stateToRaw?: (state: S) => Vec;

  /** Dimension of raw state features (required if stateToRaw is provided) */
  rawDim?: number;

  /** Names of registered domains to use as composition targets for encoder learning */
  composeWith?: string[];
}

// ============================================================================
// Functor Discovery Types
// ============================================================================

/**
 * Configuration for functor discovery.
 *
 * Acknowledges that we cannot prove a functor doesn't exist (undecidable),
 * only that we haven't found one within budget.
 */
export interface FunctorDiscoveryConfig {
  /** Maximum epochs before giving up */
  maxEpochs: number;

  /** Binding success rate to declare "found" (0-1) */
  successThreshold: number;

  /** Epochs without improvement before declaring "stalled" */
  patience: number;

  /** Minimum epochs before early stopping */
  minEpochs: number;

  /** Network hidden layer sizes */
  hiddenSizes: number[];

  /** Learning rate for optimizer */
  learningRate: number;

  /** Samples per epoch during discovery */
  samplesPerEpoch: number;
}

export const defaultFunctorDiscoveryConfig: FunctorDiscoveryConfig = {
  maxEpochs: 200,
  successThreshold: 0.8,
  patience: 50,
  minEpochs: 20,
  hiddenSizes: [32, 32],
  learningRate: 0.01,
  samplesPerEpoch: 50,
};

/**
 * Status of functor discovery.
 *
 * - "found": Functor discovered, binding rate above threshold
 * - "not_found": No functor found within budget (heuristic, not proof)
 * - "inconclusive": Some progress but didn't meet threshold
 */
export type FunctorStatus = "found" | "not_found" | "inconclusive";

/**
 * Reason for discovery termination.
 */
export type FunctorReason = "converged" | "timeout" | "stalled" | "threshold";

/**
 * Result of functor discovery.
 */
export interface FunctorResult {
  /** Honest status: we can't prove non-existence */
  status: FunctorStatus;

  /** The discovered functor (if found) */
  functor?: FunctorNetwork;

  /** Final binding success rate */
  bindingRate: number;

  /** Number of epochs run */
  epochs: number;

  /** Why discovery terminated */
  reason: FunctorReason;

  /** Source domain name */
  source: string;

  /** Target domain name */
  target: string;
}

/**
 * Interface for a functor network.
 * Maps state embeddings from source domain to target domain.
 */
export interface FunctorNetwork {
  /** Apply the functor to transform a state embedding */
  apply(stateEmb: Vec): Vec;

  /** Get the input dimension */
  inputDim: number;

  /** Get the output dimension */
  outputDim: number;

  /** Dispose of resources */
  dispose(): void;
}

/**
 * A step in a composition path.
 */
export interface CompositionStep {
  /** Source domain name */
  from: string;

  /** Target domain name */
  to: string;

  /** The functor connecting them */
  functor: FunctorNetwork;

  /** Binding rate for this step */
  bindingRate: number;
}

/**
 * A path through the composition graph.
 */
export interface CompositionPath {
  /** The steps in the path */
  steps: CompositionStep[];

  /** Product of individual binding rates */
  totalBindingRate: number;

  /** Human-readable description */
  describe(): string;
}

/**
 * Metadata about a registered port.
 */
export interface PortMetadata {
  /** Unique name for the port */
  name: string;

  /** Dimension of state embeddings */
  embeddingDim: number;

  /** When the port was registered */
  registeredAt: Date;
}
