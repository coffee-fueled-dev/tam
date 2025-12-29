/**
 * Primitive Mathematical Domains
 *
 * Simple 1D operations for hierarchical composition experiments:
 * - Shift: x → x + c
 * - Scale: x → x * k
 * - Power: x → x^n
 * - Affine: x → ax + b (composed from shift + scale)
 *
 * Each domain is a DomainSpec compatible with TAM.learn().
 */

import type { DomainSpec } from "../../src";

// ============================================================================
// Deterministic Shift: x → x + 1 (predictable)
// ============================================================================

const DET_SHIFT_MIN = -10;
const DET_SHIFT_MAX = 10;

export const shiftDeterministic: DomainSpec<number> = {
  randomState: () =>
    DET_SHIFT_MIN + Math.random() * (DET_SHIFT_MAX - DET_SHIFT_MIN),
  simulate: (x) => x + 1, // Always +1, fully predictable
  embedder: (x) => [
    (x - (DET_SHIFT_MIN + DET_SHIFT_MAX) / 2) /
      ((DET_SHIFT_MAX - DET_SHIFT_MIN) / 2),
  ],
  embeddingDim: 1,
};

// ============================================================================
// Shift Domain: x → x + c (stochastic)
// ============================================================================

const SHIFT_MIN = -10;
const SHIFT_MAX = 10;
const SHIFT_DELTA = 2;

export const shift: DomainSpec<number> = {
  randomState: () => SHIFT_MIN + Math.random() * (SHIFT_MAX - SHIFT_MIN),
  simulate: (x) => x + (Math.random() - 0.5) * 2 * SHIFT_DELTA,
  embedder: (x) => [
    (x - (SHIFT_MIN + SHIFT_MAX) / 2) / ((SHIFT_MAX - SHIFT_MIN) / 2),
  ],
  embeddingDim: 1,
};

// ============================================================================
// Scale Domain: x → x * k
// ============================================================================

const SCALE_MIN = 0.5;
const SCALE_MAX = 5;
const SCALE_FACTOR_MIN = 0.5;
const SCALE_FACTOR_MAX = 2;

export const scale: DomainSpec<number> = {
  randomState: () => SCALE_MIN + Math.random() * (SCALE_MAX - SCALE_MIN),
  simulate: (x) => {
    const k =
      SCALE_FACTOR_MIN + Math.random() * (SCALE_FACTOR_MAX - SCALE_FACTOR_MIN);
    return Math.max(SCALE_MIN, Math.min(SCALE_MAX, x * k));
  },
  embedder: (x) => [
    (x - (SCALE_MIN + SCALE_MAX) / 2) / ((SCALE_MAX - SCALE_MIN) / 2),
  ],
  embeddingDim: 1,
};

// ============================================================================
// Power Domain: x → x^n
// ============================================================================

const POWER_MIN = 0.5;
const POWER_MAX = 3;
const POWER_EXP_MIN = 0.5;
const POWER_EXP_MAX = 2;

export const power: DomainSpec<number> = {
  randomState: () => POWER_MIN + Math.random() * (POWER_MAX - POWER_MIN),
  simulate: (x) => {
    const n = POWER_EXP_MIN + Math.random() * (POWER_EXP_MAX - POWER_EXP_MIN);
    return Math.max(POWER_MIN, Math.min(POWER_MAX, Math.pow(x, n)));
  },
  embedder: (x) => [
    (x - (POWER_MIN + POWER_MAX) / 2) / ((POWER_MAX - POWER_MIN) / 2),
  ],
  embeddingDim: 1,
};

// ============================================================================
// Affine Domain: x → ax + b
// ============================================================================

const AFFINE_MIN = -10;
const AFFINE_MAX = 10;
const AFFINE_SCALE_MIN = 0.5;
const AFFINE_SCALE_MAX = 2;
const AFFINE_SHIFT_MAX = 2;

export const affine: DomainSpec<number> = {
  randomState: () => AFFINE_MIN + Math.random() * (AFFINE_MAX - AFFINE_MIN),
  simulate: (x) => {
    const a =
      AFFINE_SCALE_MIN + Math.random() * (AFFINE_SCALE_MAX - AFFINE_SCALE_MIN);
    const b = (Math.random() - 0.5) * 2 * AFFINE_SHIFT_MAX;
    return Math.max(AFFINE_MIN, Math.min(AFFINE_MAX, a * x + b));
  },
  embedder: (x) => [
    (x - (AFFINE_MIN + AFFINE_MAX) / 2) / ((AFFINE_MAX - AFFINE_MIN) / 2),
  ],
  embeddingDim: 1,
};
