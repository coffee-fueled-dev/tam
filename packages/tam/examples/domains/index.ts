/**
 * Domain exports.
 *
 * Domains are DomainSpec objects compatible with TAM.learn().
 * Each domain provides:
 * - randomState: Generate training samples
 * - simulate: Step dynamics
 * - embedder: Encode states to vectors
 * - embeddingDim: Dimension of embeddings
 *
 * Domains can be:
 * - Learned with TAM.learn()
 * - Composed via functor discovery
 * - Used as building blocks for higher-order compositions
 */

// Shared physics utilities
export {
  type PhysicsState,
  physicsEncoders,
  physicsEmbedder,
  physicsPortConfig,
  WORLD_SIZE,
  GRAVITY,
  MAX_VEL,
  FRICTION,
  BOUNCE,
} from "./physics";

// Physics domains
export { gravity, simulateGravity, randomGravityState } from "./gravity";
export { bounce, simulateBounce, randomBounceState } from "./bounce";

// Primitive 1D domains for composition experiments
export { shift, scale, power, affine, shiftDeterministic } from "./primitives";
