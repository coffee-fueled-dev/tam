/**
 * Shared physics constants and types for physics-based domains.
 */

import type { Situation, Encoders } from "../../src/types";

// Physics constants
export const WORLD_SIZE = 100;
export const GRAVITY = 0.5;
export const MAX_VEL = 15;
export const FRICTION = 0.99;
export const BOUNCE = 0.8;

/**
 * 2D physics state with position and velocity.
 */
export interface PhysicsState {
  x: number;
  y: number;
  vx: number;
  vy: number;
}

/**
 * Standard encoders for physics states.
 */
export const physicsEncoders: Encoders<PhysicsState, {}> = {
  embedSituation(sit: Situation<PhysicsState, {}>) {
    const { x, y, vx, vy } = sit.state;
    return [
      (x / WORLD_SIZE) * 2 - 1,
      (y / WORLD_SIZE) * 2 - 1,
      vx / MAX_VEL,
      vy / MAX_VEL,
    ];
  },
  delta(before, after) {
    return [
      (after.state.x - before.state.x) / WORLD_SIZE,
      (after.state.y - before.state.y) / WORLD_SIZE,
      (after.state.vx - before.state.vx) / MAX_VEL,
      (after.state.vy - before.state.vy) / MAX_VEL,
    ];
  },
};

/**
 * Embedder function for physics states.
 */
export function physicsEmbedder(state: PhysicsState): number[] {
  return physicsEncoders.embedSituation({ state, context: {} });
}

/**
 * Standard port configuration for physics domains.
 */
export const physicsPortConfig = {
  embeddingDim: 4,
  causal: { hiddenSizes: [32, 16] },
  commitment: { initialRadius: 0.5 },
};

