/**
 * TAM - Trajectory-Affordance Model
 *
 * A framework for learning and composing world models through
 * commitment-based ports and functor discovery.
 *
 * Core concepts:
 * - Ports model p(Δ | situation, action) - the distribution of outcomes
 * - Agency measures prediction specificity: narrow cones = high agency
 * - Binding: commit to a cone, evaluate if trajectory falls within
 * - Refinement: narrow on success, widen on failure, proliferate for new regimes
 * - Composition: discover functors to map between domain embeddings
 */

// ============================================================================
// Core Types
// ============================================================================

export {
  type Situation,
  type Transition,
  type Encoders,
  type Prediction,
  type MultiModalPrediction,
  type Port,
  type PortBank,
  // Geometric port types
  type Cone,
  type BindingOutcome,
  type RefinementAction,
  type GeometricPortId,
  type BindingHistory,
  type RefinementPolicy,
  type CausalNetConfig,
  type CommitmentNetConfig,
  type GeometricPortConfig,
  type GeometricPortConfigInput,
  defaultCausalNetConfig,
  defaultCommitmentNetConfig,
  defaultGeometricPortConfig,
} from "./types";

// ============================================================================
// Vector Utilities
// ============================================================================

export {
  type Vec,
  zeros,
  dot,
  add,
  sub,
  scale,
  l2,
  l2Sq,
  norm,
  mean,
  sum,
  prod,
  clamp,
  softmax,
  logSumExp,
  cosineSimilarity,
  magnitude,
} from "./vec";

// ============================================================================
// Agency Computation
// ============================================================================

export {
  type AgencyConfig,
  defaultAgencyConfig,
  agencyFromVariance,
  agencyFromEntropy,
  weightedAgency,
  isWithinBounds,
  // Cone-based agency
  type ConeAgencyConfig,
  defaultConeAgencyConfig,
  agencyFromCone,
  agencyFromLogVolume,
} from "./agency";

// ============================================================================
// Geometric Port Architecture
// ============================================================================

export {
  // Core networks
  CausalNet,
  CommitmentNet,
  // Fibration utilities
  assembleCone,
  assembleConeWithCovariance,
  evaluateBinding,
  isInNarrowZone,
  coneVolume,
  coneLogVolume,
  // History and policy
  DefaultBindingHistory,
  FixedRefinementPolicy,
  // Port and bank
  GeometricPort,
  GeometricPortBank,
  // Port functor discovery
  PortFunctor,
  discoverPortFunctor,
  generateSpecialistEmbedding,
  type PortFunctorDiscoveryConfig,
  defaultPortFunctorConfig,
  // Intra-domain encoder learning
  IntraDomainEncoder,
  createLearnableEncoder,
  type IntraDomainEncoderConfig,
  defaultIntraDomainEncoderConfig,
  EncoderBridge,
  createEncoderBridge,
  type EncoderBridgeConfig,
} from "./geometric";

// ============================================================================
// Composition API
// ============================================================================

export {
  // Main API
  TAM,
  type DomainConfig,
  type DomainSpec,
  type TrainingConfig,
  defaultTrainingConfig,
  // Types
  type FunctorDiscoveryConfig,
  type FunctorResult,
  type FunctorStatus,
  type FunctorReason,
  type FunctorNetwork,
  type CompositionPath,
  type CompositionStep,
  type PortMetadata,
  defaultFunctorDiscoveryConfig,
  // Components
  PortRegistry,
  FunctorCache,
  CompositionGraph,
  ComposedPort,
  createComposedPort,
  TFFunctorNetwork,
  discoverFunctor,
} from "./composition";
