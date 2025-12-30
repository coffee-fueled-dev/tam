/**
 * Geometric TAM Architecture
 *
 * Exports for the commitment-based port architecture with:
 * - Shared CausalNet (trajectory prediction)
 * - Shared CommitmentNet (tolerance prediction with asymmetric training)
 * - Ports as embeddings in the action manifold
 * - Discrete binding predicate evaluation
 * - Principled refinement policy (narrow/widen/proliferate/noop)
 */

// Core networks
export { CausalNet } from "./causal";
export { CommitmentNet } from "./commitment";

// Fibration utilities
export {
  assembleCone,
  assembleConeWithCovariance,
  evaluateBinding,
  isInNarrowZone,
  coneVolume,
  coneLogVolume,
} from "./fibration";

// History tracking
export { DefaultBindingHistory } from "./history";

// Refinement policy
export { FixedRefinementPolicy } from "./refinement";

// Port and port bank
export { GeometricPort } from "./port";
export { GeometricPortBank } from "./bank";

// Port functor discovery (intra-domain composition)
export {
  PortFunctor,
  discoverPortFunctor,
  generateSpecialistEmbedding,
  type PortFunctorDiscoveryConfig,
  defaultPortFunctorConfig,
} from "./port-functor";

