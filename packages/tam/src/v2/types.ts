/**
 * TAM v2: Minimal Core Types
 *
 * Focus on geometric essentials:
 * - Causal manifold (trajectories)
 * - Commitment manifold (affordance cones)
 * - Port manifold (specialist embeddings)
 */

export type Vec = number[];

/**
 * A situation is a state + any context.
 * For simplicity, v2 focuses on state-only.
 */
export interface Situation<S> {
  state: S;
}

/**
 * A transition is a before → after pair.
 */
export interface Transition<S> {
  before: Situation<S>;
  after: Situation<S>;
}

/**
 * A prediction from the actor.
 */
export interface Prediction {
  /** Predicted trajectory (delta in state space) */
  delta: Vec;

  /** Affordance cone around the prediction */
  cone: Cone;

  /** Agency: commitment specificity (0-1, higher = more confident) */
  agency: number;
}

/**
 * Affordance cone: center + anisotropic radius.
 */
export interface Cone {
  center: Vec;
  radius: Vec;
}

/**
 * Configuration for the Actor.
 */
export interface ActorConfig {
  /** Initial port embeddings (if seeding) */
  initialPorts?: Vec[];

  /** Causal network config */
  causal?: CausalNetConfig;

  /** Proliferation config */
  proliferation?: ProliferationConfig;

  /** Exploration config (active inference / curiosity) - only used by fixed policy */
  exploration?: ExplorationConfig;

  /** Port selection strategy (defaults to fixed curiosity-driven policy) */
  portSelectionPolicy?: PortSelectionPolicy;

  /** Refinement strategy (defaults to fixed soft-boundary policy) */
  refinementPolicy?: RefinementPolicy;
}

export interface ExplorationConfig {
  /** Probability of exploration vs exploitation (default: 0.1) */
  rate?: number;
}

/**
 * Context for port selection decisions
 */
export interface PortSelectionContext {
  stateEmb: Vec;
  portEmbs: Vec[];
  concentrations: number[];
  portSamples: number[];
  bindingRates?: number[];  // Recent binding success rate per port (prevents selecting failing ports)
}

/**
 * Context for refinement decisions
 */
export interface RefinementContext {
  stateEmb: Vec;
  portEmb: Vec;
  concentration: number;
  bindingStrength: number;
  normalizedDistance: number;
  predictionError: number;
  nearbyPortDistances?: number[];  // Distances to k-nearest ports in embedding space (for overlap detection)
}

/**
 * Strategy for selecting which port to use for a given state
 */
export interface PortSelectionPolicy {
  selectPort(context: PortSelectionContext): number;
}

/**
 * Strategy for deciding refinement actions (concentrate/disperse)
 */
export interface RefinementPolicy {
  /**
   * Returns [concentratePressure, dispersePressure] in range [0, 1]
   */
  decideRefinement(context: RefinementContext): { concentrate: number; disperse: number };
}

export interface CausalNetConfig {
  hiddenSizes?: number[];
  learningRate?: number;
  batchSize?: number;
  minVariance?: number; // Minimum variance floor for one-sided penalty (default: 5.0)
}

export interface ProliferationConfig {
  /** Enable automatic proliferation */
  enabled?: boolean;

  /** Window size for statistical analysis of errors and agencies */
  windowSize?: number;

  /** Minimum samples before allowing proliferation (wait for initial learning to stabilize) */
  minSamplesBeforeProliferation?: number;
}

export const defaultActorConfig = {
  initialPorts: [],
  causal: {
    hiddenSizes: [32, 32],
    learningRate: 0.001,
    batchSize: 10,
  },
  proliferation: {
    enabled: true,
    windowSize: 50,
    minSamplesBeforeProliferation: 500,
  },
  exploration: {
    rate: 0.1,  // 10% exploration rate
  },
  // Policies are initialized in Actor constructor to avoid circular dependency
  portSelectionPolicy: undefined,
  refinementPolicy: undefined,
} as const;
