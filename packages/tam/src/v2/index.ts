/**
 * TAM v2: Minimal Core
 *
 * Clean reimplementation focusing on:
 * - Latent port proliferation (ports as embeddings, not objects)
 * - Unified geometric system (dual-head CausalNet)
 * - Agency-based selection
 * - Geometric affordances
 */

export { Actor } from "./actor";
export { CausalNet } from "./causal";
export {
  FixedPortSelectionPolicy,
  FixedRefinementPolicy,
  ProportionalRefinementPolicy,
  LearnedRefinementPolicy,
  TAMPortSelectionPolicy,
  TAMRefinementPolicy,
  embedUnifiedQuery,
  type UnifiedQuery,
} from "./policies";
export * from "./types";
export * from "./harness";
