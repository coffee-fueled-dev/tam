/**
 * Standard Domains for Experiments
 *
 * Reusable domain definitions that implement the Domain interface.
 * Organized by complexity and purpose.
 */

import type { Domain } from "./types";

// ============================================================================
// Simple 1D Domains (Good for basic testing)
// ============================================================================

/**
 * 1D Damped Spring: x'' = -kx - bx'
 * Simple, interpretable, single-mode dynamics.
 */
export const dampedSpring1D: Domain<{ x: number; v: number }> = {
  name: "1D Damped Spring",
  embeddingDim: 2,

  randomState: () => ({
    x: (Math.random() - 0.5) * 2,
    v: (Math.random() - 0.5) * 2,
  }),

  simulate: (state, steps = 1) => {
    const k = 1.0;
    const b = 0.1;
    const dt = 0.1;

    let current = state;
    for (let i = 0; i < steps; i++) {
      const ax = -k * current.x - b * current.v;
      const newV = current.v + ax * dt;
      const newX = current.x + newV * dt;
      current = { x: newX, v: newV };
    }
    return current;
  },

  embed: (state) => [state.x, state.v],
};

// ============================================================================
// 2D Domains (Multi-mode, good for specialists)
// ============================================================================

/**
 * 2D Point with Drift Modes
 * Multiple distinct behaviors that need specialization.
 */
export type DriftMode = "leftward" | "rightward" | "upward" | "downward";

export interface DriftState {
  x: number;
  y: number;
  mode: DriftMode;
}

export const driftModes: Domain<DriftState> = {
  name: "2D Drift Modes",
  embeddingDim: 6, // 2 position + 4 mode one-hot
  rawDim: 6,

  randomState: () => {
    const modes: DriftMode[] = ["leftward", "rightward", "upward", "downward"];
    return {
      x: (Math.random() - 0.5) * 2,
      y: (Math.random() - 0.5) * 2,
      mode: modes[Math.floor(Math.random() * modes.length)]!,
    };
  },

  simulate: (state, steps = 1) => {
    const drift = 0.2;
    let current = { ...state };

    for (let i = 0; i < steps; i++) {
      switch (current.mode) {
        case "leftward":
          current.x -= drift;
          break;
        case "rightward":
          current.x += drift;
          break;
        case "upward":
          current.y += drift;
          break;
        case "downward":
          current.y -= drift;
          break;
      }

      // Keep in bounds
      current.x = Math.max(-2, Math.min(2, current.x));
      current.y = Math.max(-2, Math.min(2, current.y));
    }

    return current;
  },

  embed: (state) => {
    const modeOneHot = {
      leftward: [1, 0, 0, 0],
      rightward: [0, 1, 0, 0],
      upward: [0, 0, 1, 0],
      downward: [0, 0, 0, 1],
    }[state.mode];
    return [state.x, state.y, ...modeOneHot];
  },

  extractRaw: (state) => {
    // Same as embed for this domain
    const modeOneHot = {
      leftward: [1, 0, 0, 0],
      rightward: [0, 1, 0, 0],
      upward: [0, 0, 1, 0],
      downward: [0, 0, 0, 1],
    }[state.mode];
    return [state.x, state.y, ...modeOneHot];
  },
};

// ============================================================================
// 2D Pendulum with Noise (For encoder learning)
// ============================================================================

export interface PendulumState {
  x: number;
  y: number;
  vx: number;
  vy: number;
  noise: number[]; // Irrelevant features
}

export const noisyPendulum = (noiseFeatures: number = 6): Domain<PendulumState> => ({
  name: `2D Pendulum (${noiseFeatures} noise features)`,
  embeddingDim: 4, // Only relevant: x, y, vx, vy
  rawDim: 4 + noiseFeatures,

  randomState: () => ({
    x: (Math.random() - 0.5) * 2,
    y: (Math.random() - 0.5) * 2,
    vx: (Math.random() - 0.5) * 2,
    vy: (Math.random() - 0.5) * 2,
    noise: Array.from({ length: noiseFeatures }, () => Math.random() - 0.5),
  }),

  simulate: (state, steps = 1) => {
    const dt = 0.1;
    const gravity = 0.5;
    const damping = 0.95;

    let current = { ...state };

    for (let i = 0; i < steps; i++) {
      const ax = 0;
      const ay = gravity;

      const newVx = (current.vx + ax * dt) * damping;
      const newVy = (current.vy + ay * dt) * damping;

      current = {
        x: current.x + newVx * dt,
        y: current.y + newVy * dt,
        vx: newVx,
        vy: newVy,
        noise: Array.from({ length: noiseFeatures }, () => Math.random() - 0.5),
      };
    }

    return current;
  },

  embed: (state) => [state.x, state.y, state.vx, state.vy],

  extractRaw: (state) => [state.x, state.y, state.vx, state.vy, ...state.noise],
});

// ============================================================================
// Domain Variants (For testing generalization)
// ============================================================================

/**
 * Variant of damped spring with different parameters.
 * Useful for testing compositional generalization.
 */
export const dampedSpringVariant = (
  k: number,
  b: number,
  name?: string
): Domain<{ x: number; v: number }> => ({
  name: name || `1D Spring (k=${k}, b=${b})`,
  embeddingDim: 2,

  randomState: () => ({
    x: (Math.random() - 0.5) * 2,
    v: (Math.random() - 0.5) * 2,
  }),

  simulate: (state, steps = 1) => {
    const dt = 0.1;
    let current = state;

    for (let i = 0; i < steps; i++) {
      const ax = -k * current.x - b * current.v;
      const newV = current.v + ax * dt;
      const newX = current.x + newV * dt;
      current = { x: newX, v: newV };
    }

    return current;
  },

  embed: (state) => [state.x, state.v],
});

// ============================================================================
// Composite Domains (For multi-horizon experiments)
// ============================================================================

/**
 * Create a multi-horizon wrapper around any domain.
 * Returns a map of horizon → domain variant.
 */
export function createMultiHorizonDomain<S>(
  baseDomain: Domain<S>,
  horizons: number[]
): Map<number, Domain<S>> {
  const domains = new Map<number, Domain<S>>();

  for (const h of horizons) {
    domains.set(h, {
      ...baseDomain,
      name: `${baseDomain.name} (${h}-step)`,
      simulate: (state) => baseDomain.simulate(state, h),
    });
  }

  return domains;
}
