/**
 * Bounce Domain
 *
 * Wall collision dynamics (no gravity).
 * Objects bounce off walls with damping.
 */

import type { DomainSpec } from "../../src";
import {
  type PhysicsState,
  physicsEmbedder,
  physicsEncoders,
  physicsPortConfig,
  WORLD_SIZE,
  FRICTION,
  BOUNCE,
} from "./physics";

/**
 * Simulate bounce physics (wall collisions, no gravity).
 */
export function simulateBounce(s: PhysicsState): PhysicsState {
  let { x, y, vx, vy } = s;
  for (let i = 0; i < 10; i++) {
    x += vx;
    y += vy;
    if (x < 0) {
      x = 0;
      vx = -vx * BOUNCE;
    }
    if (x > WORLD_SIZE) {
      x = WORLD_SIZE;
      vx = -vx * BOUNCE;
    }
    if (y < 0) {
      y = 0;
      vy = -vy * BOUNCE;
    }
    if (y > WORLD_SIZE) {
      y = WORLD_SIZE;
      vy = -vy * BOUNCE;
    }
    vx *= FRICTION;
    vy *= FRICTION;
  }
  return { x, y, vx, vy };
}

/**
 * Generate random state near walls with incoming velocity.
 */
export function randomBounceState(): PhysicsState {
  const wall = Math.floor(Math.random() * 4);
  const near = 10;
  switch (wall) {
    case 0:
      return { x: 50, y: WORLD_SIZE - near, vx: 0, vy: 5 + Math.random() * 5 };
    case 1:
      return { x: 50, y: near, vx: 0, vy: -5 - Math.random() * 5 };
    case 2:
      return { x: near, y: 50, vx: -5 - Math.random() * 5, vy: 0 };
    default:
      return { x: WORLD_SIZE - near, y: 50, vx: 5 + Math.random() * 5, vy: 0 };
  }
}

/**
 * Bounce domain specification.
 */
export const bounce: DomainSpec<PhysicsState> = {
  randomState: randomBounceState,
  simulate: simulateBounce,
  embedder: physicsEmbedder,
  embeddingDim: 4,
  encoders: physicsEncoders,
  portConfig: physicsPortConfig,
};
