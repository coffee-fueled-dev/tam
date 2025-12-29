/**
 * Vector utilities for TA implementation.
 * All operations are pure functions on number arrays.
 */

export type Vec = number[];

export function assertSameDim(a: Vec, b: Vec): void {
  if (a.length !== b.length)
    throw new Error(`Dim mismatch: ${a.length} vs ${b.length}`);
}

export function zeros(n: number): Vec {
  return new Array(n).fill(0);
}

export function dot(a: Vec, b: Vec): number {
  assertSameDim(a, b);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += (a[i] ?? 0) * (b[i] ?? 0);
  }
  return isNaN(s) ? 0 : s;
}

export function add(a: Vec, b: Vec): Vec {
  assertSameDim(a, b);
  return a.map((v, i) => v + b[i]!);
}

export function sub(a: Vec, b: Vec): Vec {
  assertSameDim(a, b);
  return a.map((v, i) => v - b[i]!);
}

export function scale(a: Vec, s: number): Vec {
  return a.map((v) => v * s);
}

export function l2Sq(a: Vec, b: Vec): number {
  assertSameDim(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return sum;
}

export function l2(a: Vec, b: Vec): number {
  return Math.sqrt(l2Sq(a, b));
}

export function norm(a: Vec): number {
  return Math.sqrt(a.reduce((s, v) => s + v * v, 0));
}

export function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

export function mean(vecs: Vec[]): Vec {
  if (vecs.length === 0) return [];
  const first = vecs[0]!;
  const dim = first.length;
  const s = zeros(dim);
  for (const v of vecs) {
    for (let i = 0; i < dim; i++) s[i]! += v[i] ?? 0;
  }
  return s.map((x) => (x ?? 0) / vecs.length);
}

/** Product of array elements */
export function prod(xs: number[]): number {
  return xs.reduce((a, b) => a * b, 1);
}

/** Sum of array elements */
export function sum(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0);
}

/** Log-sum-exp for numerical stability */
export function logSumExp(xs: number[]): number {
  if (xs.length === 0) return Number.NEGATIVE_INFINITY;
  const m = Math.max(...xs);
  if (!isFinite(m)) return m;
  let s = 0;
  for (const x of xs) s += Math.exp(x - m);
  return m + Math.log(s || 1e-12);
}

/** Softmax normalization */
export function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((v) => Math.exp(v - m));
  const s = sum(exps) || 1;
  return exps.map((e) => e / s);
}

/**
 * Cosine similarity between two vectors.
 * Returns value in [-1, 1] where 1 = same direction, 0 = orthogonal, -1 = opposite.
 * Returns 0 for zero vectors.
 */
export function cosineSimilarity(a: Vec, b: Vec): number {
  const normA = norm(a);
  const normB = norm(b);
  if (normA < 1e-8 || normB < 1e-8) return 0;
  return dot(a, b) / (normA * normB);
}

/**
 * Magnitude (L2 norm) of a vector.
 */
export function magnitude(a: Vec): number {
  return Math.sqrt(a.reduce((s, v) => s + v * v, 0));
}
