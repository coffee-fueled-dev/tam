/**
 * Gravity Domain
 *
 * Freefall dynamics in open space (no walls).
 * Objects accelerate downward due to gravity.
 */

import type { DomainSpec } from "../../src";
import {
  type PhysicsState,
  physicsEmbedder,
  physicsEncoders,
  physicsPortConfig,
  GRAVITY,
  FRICTION,
} from "./physics";

/**
 * Simulate gravity physics (no wall collisions).
 */
export function simulateGravity(s: PhysicsState): PhysicsState {
  let { x, y, vx, vy } = s;
  for (let i = 0; i < 10; i++) {
    vy += GRAVITY;
    x += vx;
    y += vy;
    vx *= FRICTION;
    vy *= FRICTION;
  }
  return { x, y, vx, vy };
}

/**
 * Generate random state in open space (away from walls).
 */
export function randomGravityState(): PhysicsState {
  return {
    x: 30 + Math.random() * 40,
    y: 20 + Math.random() * 40,
    vx: (Math.random() - 0.5) * 4,
    vy: (Math.random() - 0.5) * 4,
  };
}

/**
 * Gravity domain specification.
 */
export const gravity: DomainSpec<PhysicsState> = {
  randomState: randomGravityState,
  simulate: simulateGravity,
  embedder: physicsEmbedder,
  embeddingDim: 4,
  encoders: physicsEncoders,
  portConfig: physicsPortConfig,
};
