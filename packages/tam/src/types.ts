/**
 * Core TAM types and interfaces.
 *
 * These are implementation-agnostic abstractions for:
 * - Situations and transitions (what the agent observes)
 * - Encoders (how to embed situations and compute deltas)
 * - Ports (how to model p(Δ | situation, action))
 * - Predictions (what the port outputs)
 */

import type { Vec } from "./vec";

// ============================================================================
// Situations & Transitions
// ============================================================================

/**
 * A situation is a state with optional context.
 * The state is the core observable, context provides additional info.
 */
export interface Situation<S, C = unknown> {
  state: S;
  context: C;
}

/**
 * A transition records before/after situations and the action taken.
 */
export interface Transition<S, C = unknown> {
  before: Situation<S, C>;
  after: Situation<S, C>;
  action: string;
}

// ============================================================================
// Encoders
// ============================================================================

/**
 * Encoders transform domain-specific situations into vectors.
 * This is the bridge between raw state and learned representations.
 */
export interface Encoders<S, C = unknown> {
  /** Embed situation into a vector for the port's internal processing */
  embedSituation: (sit: Situation<S, C>) => Vec;

  /** Compute the delta (change) between two situations */
  delta: (before: Situation<S, C>, after: Situation<S, C>) => Vec;
}

// ============================================================================
// Predictions
// ============================================================================

/**
 * A prediction from a port: expected delta, confidence, and agency.
 */
export interface Prediction {
  /** The predicted change in state */
  delta: Vec;

  /** Confidence/probability score (log-scale or linear, implementation-dependent) */
  score: number;

  /** Agency: how specific/committed this prediction is (0 = uncertain, 1 = certain) */
  agency: number;

  /** Optional: the full cone for this prediction mode */
  cone?: Cone;

  /** Optional: mode index for multi-modal predictions */
  modeIndex?: number;
}

/**
 * Multi-modal prediction: a mixture of cones representing "either-or" outcomes.
 * For ARC-style tasks where one action might have distinct valid outcomes
 * (e.g., object could move OR change color).
 */
export interface MultiModalPrediction {
  /** Individual modes, each with a weight (mixing coefficient) */
  modes: Array<{
    prediction: Prediction;
    weight: number; // Mixing coefficient, sums to 1 across modes
  }>;

  /** Total agency across all modes (weighted average) */
  totalAgency: number;

  /** Entropy of the mode distribution (high = uncertain which mode) */
  modeEntropy: number;
}

// ============================================================================
// Port Interface
// ============================================================================

/**
 * A Port models p(Δ | situation, action) - the distribution of outcomes
 * given a situation and action.
 *
 * This is the core abstraction that can be implemented by:
 * - Gaussian mixture models (analytical, interpretable)
 * - Neural networks (flexible, learns non-linear patterns)
 * - Discrete/categorical models (for grid worlds)
 */
export interface Port<S, C = unknown> {
  /** The action this port represents */
  readonly action: string;

  /** Human-readable name for debugging */
  readonly name: string;

  /**
   * Predict top-k outcomes for a situation.
   * Returns predictions sorted by score (best first).
   */
  predict(sit: Situation<S, C>, k?: number): Prediction[];

  /**
   * Observe a transition and update the port's model.
   * Only processes transitions matching this port's action.
   */
  observe(tr: Transition<S, C>): void;

  /**
   * Get the overall agency of this port (how specific its predictions are).
   */
  getAgency(): number;

  /**
   * Get diagnostic information for inspection.
   */
  snapshot(): unknown;

  /**
   * Clean up resources (e.g., dispose TensorFlow models).
   */
  dispose?(): void;
}

// ============================================================================
// PortBank Interface
// ============================================================================

/**
 * A PortBank manages multiple ports, one per action.
 * Lazily creates ports as needed.
 */
export interface PortBank<S, C = unknown> {
  /** Get or create a port for an action */
  get(action: string): Port<S, C>;

  /** Observe a transition (routes to appropriate port) */
  observe(tr: Transition<S, C>): void;

  /** Get predictions for an action in a situation */
  predict(action: string, sit: Situation<S, C>, k?: number): Prediction[];

  /** Get average agency across all ports */
  getAgency(): number;

  /** Get diagnostic snapshot of all ports */
  snapshot(): unknown;

  /** Clean up all ports */
  dispose?(): void;
}

// ============================================================================
// Geometric Port Types (Commitment-Based Architecture)
// ============================================================================

/**
 * A cone represents a committed tolerance region in trajectory space.
 * The actor commits to accepting trajectories within this cone.
 *
 * Supports both isotropic (scalar radius) and anisotropic (covariance) forms.
 */
export interface Cone {
  /** Center of the cone (predicted trajectory from CausalNet) */
  center: Vec;

  /** Per-dimension radius (tolerance from CommitmentNet) - isotropic version */
  radius: Vec;

  /**
   * Optional: Full covariance matrix for ellipsoidal cones.
   * Allows agent to be "committed" on some axes while "uncommitted" on others.
   * Stored as flattened lower-triangular Cholesky factor for efficiency.
   * If present, radius is ignored and covariance is used.
   */
  covariance?: Vec; // Flattened lower-triangular Cholesky factor

  /**
   * Optional: Precision matrix (inverse covariance) for efficient Mahalanobis distance.
   * If present, used directly instead of inverting covariance.
   */
  precision?: Vec; // Flattened precision matrix
}

/**
 * Result of evaluating the binding predicate χ_p.
 * Determines whether the actual trajectory falls within the committed cone.
 */
export type BindingOutcome =
  | { success: true; margin: number } // Distance inside cone (positive)
  | { success: false; violation: number }; // Distance outside cone (positive)

/**
 * Refinement actions that modify port structure.
 * - narrow: tighten cone (trajectory space, increase agency)
 * - widen: loosen cone (trajectory space, decrease agency for safety)
 * - proliferate: spawn new port (situation space, create specialist)
 * - noop: no change (at equilibrium)
 */
export type RefinementAction = "narrow" | "widen" | "proliferate" | "noop";

/**
 * A geometric port is identified by its embedding in the action manifold.
 * The embedding is a learned vector that conditions the shared networks.
 *
 * Geometric interpretation:
 * - Port Space (P): Where port embeddings live (the "blank wall")
 * - Trajectory Space (T): Possible outcomes (the "outside world")
 * - Situation Space (X): Agent's context (viewing position)
 * - Fibration π(p,x): Maps (port, situation) → cone in T
 *
 * The cone is computed as:
 *   center: π(p, x)                           ← CausalNet (fibration)
 *   radius: aperture(p) × α(p,x) / (1 + d)    ← scaled by geometry
 *
 * Where:
 *   α(p, x) = max(0, p̂ · x̂) = alignment (cosine similarity)
 *   d(p, x) = viewing distance (commitment level from CommitmentNet)
 */
export interface GeometricPortId {
  /** Unique identifier */
  id: string;
  /** Base action name (multiple ports can share the same action) */
  action: string;
  /** Learned position in action manifold (also defines orientation for alignment) */
  embedding: Vec;
  /** Intrinsic window size (base aperture) */
  aperture: number;
}

/**
 * History tracking for refinement decisions.
 * Tracks binding outcomes and agency for fiber-based proliferation.
 */
export interface BindingHistory {
  /** Record a binding outcome for a port in a situation */
  record(portId: string, situationKey: string, outcome: BindingOutcome): void;

  /** Get recent success rate for a port in similar situations */
  recentSuccessRate(portId: string, situationKey: string): number;

  /** Check if a situation is familiar (seen many times) */
  isFamiliar(situationKey: string): boolean;

  /** Get failure trajectories for a port (for proliferation) */
  getFailureTrajectories(portId: string): Vec[];

  // --- Agency Tracking for Fiber-Based Proliferation ---

  /** Record an agency observation for a port */
  recordAgency(portId: string, agency: number): void;

  /** Get average agency for a port (rolling window) */
  getAverageAgency(portId: string): number;

  /** Get total sample count for a port */
  getSampleCount(portId: string): number;

  /** Record last proliferation time for cooldown tracking */
  recordProliferation(portId: string): void;

  /** Check if port is within proliferation cooldown period */
  isInCooldown(portId: string, cooldownPeriod: number): boolean;

  // --- Legacy (deprecated) ---

  /** @deprecated Use agency-based proliferation instead */
  detectBimodal(portId: string): boolean;
}

/**
 * Refinement policy decides how to update ports after binding.
 */
export interface RefinementPolicy {
  /** Decide what action to take based on binding outcome */
  decide(
    outcome: BindingOutcome,
    portId: string,
    situationKey: string,
    history: BindingHistory
  ): RefinementAction;
}

/**
 * Configuration for CausalNet (trajectory prediction).
 */
export interface CausalNetConfig {
  /** Hidden layer sizes */
  hiddenSizes: number[];
  /** Learning rate for MSE training */
  learningRate: number;
  /** Batch size for training */
  batchSize: number;
}

export const defaultCausalNetConfig: CausalNetConfig = {
  hiddenSizes: [64, 32],
  learningRate: 0.01,
  batchSize: 32,
};

/**
 * Configuration for CommitmentNet (tolerance prediction).
 */
export interface CommitmentNetConfig {
  /** Hidden layer sizes */
  hiddenSizes: number[];
  /** Learning rate for asymmetric training */
  learningRate: number;
  /** Batch size for training */
  batchSize: number;
  /** Initial radius (wide = low agency, safe) */
  initialRadius: number;
  /** Minimum radius (prevents overconfidence) */
  minRadius: number;
  /** Threshold for "near center" when deciding to narrow */
  narrowThreshold: number;
}

export const defaultCommitmentNetConfig: CommitmentNetConfig = {
  hiddenSizes: [64, 32],
  learningRate: 0.01,
  batchSize: 32,
  initialRadius: 2.0, // Wide initial cone
  minRadius: 0.01, // Prevent collapse
  narrowThreshold: 0.5, // Only narrow if trajectory is in inner 50% of cone
};

/**
 * Configuration for the geometric port architecture.
 */
export interface GeometricPortConfig {
  /** Dimension of port embeddings (action manifold) */
  embeddingDim: number;
  /** Configuration for CausalNet */
  causal: CausalNetConfig;
  /** Configuration for CommitmentNet */
  commitment: CommitmentNetConfig;
  /** Reference volume for agency computation (dynamically updated if useDynamicReference=true) */
  referenceVolume: number;
  /** Whether to dynamically calibrate reference volume from observed deltas */
  useDynamicReference: boolean;
  /** Success rate threshold for equilibrium (stop narrowing) */
  equilibriumRate: number;
  /** Minimum samples before situation is considered familiar */
  familiarityThreshold: number;

  // --- Geometric Port Properties ---
  /** Default aperture for new ports (intrinsic window size) */
  defaultAperture: number;
  /** Minimum alignment (α) for a port to be considered applicable */
  minAlignmentThreshold: number;

  // --- Fiber Consistency ---
  /**
   * Weight for fiber consistency regularization loss.
   * Penalizes when states with similar embeddings have different dynamics.
   * Higher values = stronger push for consistent fibers.
   */
  fiberConsistencyWeight: number;
  /**
   * Distance threshold for considering two embeddings in the same fiber.
   * Embeddings within this distance are expected to have similar dynamics.
   */
  fiberThreshold: number;

  // --- Fiber-Based Proliferation ---
  /**
   * Enable port proliferation (creating specialist ports for inconsistent fibers).
   * Default: false. Only enable for genuinely multi-modal domains.
   */
  enableProliferation: boolean;
  /**
   * Agency threshold below which a port is considered to have an inconsistent fiber.
   * If a port's average agency falls below this, proliferation may be triggered.
   */
  proliferationAgencyThreshold: number;
  /**
   * Minimum samples before making a proliferation decision.
   * Prevents premature proliferation from noise.
   */
  proliferationMinSamples: number;
  /**
   * Cooldown period (in samples) between proliferation events.
   * Prevents rapid cascading proliferation.
   */
  proliferationCooldown: number;

  // --- Port Functor Discovery (Intra-Domain Composition) ---
  /**
   * Enable port functor discovery for structured proliferation.
   * When true, attempts to learn systematic transformations between ports
   * instead of using random perturbation. Falls back to random if discovery fails.
   * Default: false (use random perturbation).
   */
  enablePortFunctors: boolean;
  /**
   * Error tolerance for accepting a discovered port functor.
   * Functor is accepted if RMSE < tolerance when mapping parent → target embedding.
   */
  portFunctorTolerance: number;
  /**
   * Maximum training epochs for port functor discovery.
   * Since ports are already in learned space, this can be relatively small.
   */
  portFunctorMaxEpochs: number;

  // --- Intra-Domain Encoder Learning ---
  /**
   * Enable learnable encoder within domain.
   * When true and rawDim is specified, learns encoder via end-to-end backprop
   * through binding objective. Otherwise uses static hand-crafted encoder.
   * Default: false (use static encoder).
   */
  enableEncoderLearning: boolean;
  /**
   * Dimension of raw state features (before encoding).
   * Required if enableEncoderLearning is true. If not specified, encoder learning disabled.
   */
  encoderRawDim?: number;
  /**
   * Hidden layer sizes for learnable encoder.
   */
  encoderHiddenSizes: number[];
  /**
   * Learning rate for encoder (typically lower than port networks).
   */
  encoderLearningRate: number;
  /**
   * Weight for agency regularization in encoder loss.
   * Encourages encoder to produce embeddings that enable narrow cones.
   */
  encoderAgencyWeight: number;
  /**
   * Weight for temporal smoothness in encoder loss.
   * Encourages consecutive states to have similar embeddings.
   */
  encoderSmoothnessWeight: number;

  // --- Legacy (deprecated, kept for backward compatibility) ---
  /** @deprecated Use proliferationMinSamples instead */
  minFailuresForBimodal: number;
  /** @deprecated Replaced by agency-based proliferation */
  bimodalRatioThreshold: number;
}

/**
 * Partial config input - allows partial nested configs.
 */
export type GeometricPortConfigInput = Partial<
  Omit<GeometricPortConfig, "causal" | "commitment">
> & {
  causal?: Partial<CausalNetConfig>;
  commitment?: Partial<CommitmentNetConfig>;
};

export const defaultGeometricPortConfig: GeometricPortConfig = {
  embeddingDim: 8,
  causal: defaultCausalNetConfig,
  commitment: defaultCommitmentNetConfig,
  referenceVolume: 1.0,
  useDynamicReference: true, // Dynamically calibrate from observed deltas
  equilibriumRate: 0.95, // Stop narrowing at 95% success rate
  familiarityThreshold: 10,

  // Geometric port properties
  defaultAperture: 1.0, // Base window size
  minAlignmentThreshold: 0.1, // Ports with α < 0.1 are considered inapplicable

  // Fiber consistency
  fiberConsistencyWeight: 0.1, // Mild regularization
  fiberThreshold: 0.1, // Embeddings within 0.1 should have similar dynamics

  // Fiber-based proliferation
  enableProliferation: false, // Off by default; enable for multi-modal domains
  proliferationAgencyThreshold: 0.3, // Proliferate if agency consistently below 30%
  proliferationMinSamples: 20, // Need enough data before deciding
  proliferationCooldown: 50, // Samples between proliferations

  // Port functor discovery
  enablePortFunctors: false, // Off by default; use random perturbation
  portFunctorTolerance: 0.3, // Accept if RMSE < 0.3
  portFunctorMaxEpochs: 50, // Quick search in learned space

  // Intra-domain encoder learning
  enableEncoderLearning: false, // Off by default; use static encoder
  encoderRawDim: undefined, // Must be specified to enable
  encoderHiddenSizes: [32, 16], // Encoder network architecture
  encoderLearningRate: 0.001, // Lower than port networks (0.01)
  encoderAgencyWeight: 0.1, // Encourage narrow cones
  encoderSmoothnessWeight: 0.05, // Temporal coherence

  // Legacy (deprecated)
  minFailuresForBimodal: 20,
  bimodalRatioThreshold: 0.25,
};
