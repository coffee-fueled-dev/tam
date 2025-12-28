/* ============================================================================
Contextual Ports (TA-style) with a Gating Model in TypeScript (no deps)

What you get:
- A Port models p(Δ | situation, action) as a mixture of diagonal Gaussians:
      p(Δ|s) = Σ_k π_k(s)  N(Δ; μ_k, diag(σ²_k))
- The mixture weights π_k(s) are produced by a learned gating model:
      π(s) = softmax( W * φ(s) + b )
  where φ(s) is your situation embedding.

Refinement operators:
- NARROW: good fit => tighten expert (reduce variance) & pull mean toward delta
- WIDEN: unexpected but plausible => increase variance & pull mean a bit
- PROLIFERATE: novel delta => add a new expert and extend gating params

You plug in environment specifics via Encoders:
- embedSituation(sit) -> φ(s) ∈ R^m
- delta(before, after) -> Δ ∈ R^d

This is a general “world dynamics” predictor: it does NOT assume Markov truth,
only learns conditional delta patterns that repeat across contexts.
============================================================================ */

//////////////////////////////
// Types + Interfaces
//////////////////////////////

export type Vec = number[];

export interface Situation<S, C = unknown> {
  state: S;
  context: C;
}

export interface Transition<S, C = unknown> {
  before: Situation<S, C>;
  after: Situation<S, C>;
  action: string;
}

export interface Encoders<S, C = unknown> {
  embedSituation: (sit: Situation<S, C>) => Vec; // φ(s)
  delta: (before: Situation<S, C>, after: Situation<S, C>) => Vec; // Δ
}

export interface RefinementConfig {
  maxComponents: number;

  // Thresholds in "log space-ish"; tune per your delta scaling/dim.
  noveltyLogLikeThreshold: number; // below => proliferate
  goodFitLogLikeThreshold: number; // above => narrow, else widen

  meanLr: number; // expert mean update
  varLr: number; // expert variance update
  gateLr: number; // gating gradient step

  minVar: number;
  maxVar: number;

  // Proliferation guard: don't spawn if too close to an existing mean
  minComponentSeparationL2: number;

  // Regularize gating weights to prevent explosion
  gateWeightDecay: number; // small, e.g. 1e-4

  // Optional reward shaping: positive => narrow more, negative => widen more
  rewardScale: number;

  // If you want deterministic "top-k" delta predictions: return component means
  defaultTopK: number;

  // Split heuristic: reservoir size per component for bimodality detection
  reservoirSize: number;
  // Split when k=2 SSE is this fraction of k=1 SSE (lower = stricter)
  splitSseRatio: number;
  // Minimum samples before considering a split
  splitMinSamples: number;
  // How often to check for splits (every N observations to this component)
  splitCheckInterval: number;
}

export const defaultRefinementConfig: RefinementConfig = {
  maxComponents: 8,
  noveltyLogLikeThreshold: -80,
  goodFitLogLikeThreshold: -25,

  meanLr: 0.2,
  varLr: 0.15,
  gateLr: 0.05,

  minVar: 1e-3,
  maxVar: 1e2,

  minComponentSeparationL2: 0.5,

  gateWeightDecay: 1e-4,
  rewardScale: 0.5,

  defaultTopK: 3,

  reservoirSize: 30,
  splitSseRatio: 0.5,
  splitMinSamples: 8,
  splitCheckInterval: 5,
};

export interface PortPredictor<S, C = unknown> {
  predictDeltas: (
    sit: Situation<S, C>,
    k?: number
  ) => Array<{ delta: Vec; score: number; component: number }>;
  scoreDelta: (sit: Situation<S, C>, delta: Vec) => number; // log p(Δ|s)
}

export interface PortRefiner<S, C = unknown> {
  observe: (tr: Transition<S, C>, reward?: number) => void;
}

export interface Port<S, C = unknown>
  extends PortPredictor<S, C>,
    PortRefiner<S, C> {
  readonly action: string;
  readonly name: string;
  snapshot: () => unknown;
}

//////////////////////////////
// Vector / Math helpers
//////////////////////////////

function assertSameDim(a: Vec, b: Vec): void {
  if (a.length !== b.length)
    throw new Error(`Dim mismatch: ${a.length} vs ${b.length}`);
}

function dot(a: Vec, b: Vec): number {
  assertSameDim(a, b);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const ai = a[i] ?? 0;
    const bi = b[i] ?? 0;
    s += ai * bi;
  }
  return isNaN(s) ? 0 : s;
}

function add(a: Vec, b: Vec): Vec {
  assertSameDim(a, b);
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! + b[i]!;
  return out;
}

function sub(a: Vec, b: Vec): Vec {
  assertSameDim(a, b);
  const out = new Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i]! - b[i]!;
  return out;
}

function mulScalar(a: Vec, s: number): Vec {
  return a.map((v) => v * s);
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, x));
}

function l2Sq(a: Vec, b: Vec): number {
  assertSameDim(a, b);
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return sum;
}

function logSumExp(xs: number[]): number {
  const m = Math.max(...xs);
  let sum = 0;
  for (const x of xs) sum += Math.exp(x - m);
  return m + Math.log(sum || 1e-12);
}

function softmax(xs: number[]): number[] {
  const m = Math.max(...xs);
  const exps = xs.map((v) => Math.exp(v - m));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

/** Diagonal Gaussian log-likelihood (up to additive constant). */
function logGaussianDiag(x: Vec, mean: Vec, varDiag: Vec): number {
  assertSameDim(x, mean);
  assertSameDim(x, varDiag);
  let md2 = 0;
  let logdet = 0;
  for (let i = 0; i < x.length; i++) {
    const v = Math.max(varDiag[i]!, 1e-9);
    const d = x[i]! - mean[i]!;
    md2 += (d * d) / v;
    logdet += Math.log(v);
  }
  return -0.5 * (md2 + logdet);
}

//////////////////////////////
// Contextual Mixture Port
//////////////////////////////

interface Component {
  mean: Vec;
  varDiag: Vec;
  seen: number;
  // Gating parameters for this component: logits_k(s) = w_k · φ(s) + b_k
  gateW: Vec; // length = embedDim
  gateB: number;
  // Reservoir of recent deltas + contexts for split detection
  reservoir: Array<{ delta: Vec; phi: Vec }>;
  // Cooldown: step number when this component was last split (prevents re-splitting)
  lastSplitStep: number;
}

export class ContextualMixtureDeltaPort<S, C = unknown> implements Port<S, C> {
  public readonly action: string;
  public readonly name: string;

  private readonly enc: Encoders<S, C>;
  private readonly cfg: RefinementConfig;

  private comps: Component[] = [];
  private steps = 0;

  private embedDim: number | null = null;
  private deltaDim: number | null = null;

  constructor(opts: {
    action: string;
    name?: string;
    encoders: Encoders<S, C>;
    config?: Partial<RefinementConfig>;
    init?: { mean?: Vec; varDiag?: Vec };
  }) {
    this.action = opts.action;
    this.name = opts.name ?? `CtxPort(${opts.action})`;
    this.enc = opts.encoders;
    this.cfg = { ...defaultRefinementConfig, ...(opts.config ?? {}) };

    // Optional initial seed (only valid after we know dims from first observation)
    void opts.init;
  }

  snapshot(): unknown {
    return {
      action: this.action,
      name: this.name,
      steps: this.steps,
      embedDim: this.embedDim,
      deltaDim: this.deltaDim,
      components: this.comps.map((c, i) => ({
        i,
        mean: c.mean,
        varDiag: c.varDiag,
        seen: c.seen,
        gateW: c.gateW,
        gateB: c.gateB,
      })),
    };
  }

  scoreDelta(sit: Situation<S, C>, delta: Vec): number {
    if (this.comps.length === 0) return Number.NEGATIVE_INFINITY;
    const phi = this.enc.embedSituation(sit);
    this.ensureDims(phi, delta);

    const logPis = this.logGatingProbs(phi); // log π_k(s)
    const terms = this.comps.map(
      (c, k) => logPis[k]! + logGaussianDiag(delta, c.mean, c.varDiag)
    );
    return logSumExp(terms);
  }

  predictDeltas(
    sit: Situation<S, C>,
    k = this.cfg.defaultTopK
  ): Array<{ delta: Vec; score: number; component: number }> {
    if (this.comps.length === 0) return [];
    const phi = this.enc.embedSituation(sit);
    if (this.embedDim == null) this.embedDim = phi.length;

    const logPis = this.logGatingProbs(phi);
    const scored = this.comps.map((c, idx) => {
      // Score "at the mean" as a proxy peak: log π + log N(mean; mean, var)
      // log N(mean; mean, var) = -0.5 * logdet(var) (the MD2 term is 0)
      let logdet = 0;
      for (const v of c.varDiag) logdet += Math.log(Math.max(v, 1e-9));
      const score = logPis[idx]! - 0.5 * logdet;
      return { delta: [...c.mean], score, component: idx };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, clamp(k, 1, scored.length));
  }

  observe(tr: Transition<S, C>, reward?: number): void {
    if (tr.action !== this.action) return;
    this.steps += 1;

    const phi = this.enc.embedSituation(tr.before);
    const deltaObs = this.enc.delta(tr.before, tr.after);
    this.ensureDims(phi, deltaObs);

    // Seed if empty
    if (this.comps.length === 0) {
      this.spawnComponent(deltaObs, phi, "seed@first");
      return;
    }

    // E-step responsibilities r_k ∝ π_k(s) * N(Δ; μ_k, Σ_k)
    const logPis = this.logGatingProbs(phi);
    const logLikes = this.comps.map(
      (c, k) => logPis[k]! + logGaussianDiag(deltaObs, c.mean, c.varDiag)
    );
    const logNorm = logSumExp(logLikes);
    const resp = logLikes.map((ll) => Math.exp(ll - logNorm)); // r_k

    // Identify best component + overall log-likelihood
    let best = 0;
    for (let i = 1; i < resp.length; i++) if (resp[i]! > resp[best]!) best = i;
    const logp = logNorm;

    // Reward factor (optional)
    const r = reward ?? 0;
    const rewardFactor = 1 + this.cfg.rewardScale * r; // can be <1 if negative reward

    // Choose refinement op based on logp
    if (logp < this.cfg.noveltyLogLikeThreshold) {
      this.proliferate(deltaObs, phi);
      // Still do a small gating update on the new set (optional)
      this.updateGating(phi, resp);
      return;
    }

    if (logp >= this.cfg.goodFitLogLikeThreshold) {
      this.narrow(best, deltaObs, phi, clamp(rewardFactor, 0.25, 4));
    } else {
      this.widen(
        best,
        deltaObs,
        phi,
        clamp(1 / Math.max(rewardFactor, 1e-6), 0.25, 4)
      );
    }

    // Always: update gating params to increase likelihood of this delta in this context
    this.updateGating(phi, resp);

    // Prune if too many
    this.pruneIfNeeded();
  }

  /////////////////////////////
  // Gating model internals
  /////////////////////////////

  /** logits_k = w_k · phi + b_k; return log softmax. */
  private logGatingProbs(phi: Vec): number[] {
    const logits = this.comps.map((c) => {
      let logit = dot(c.gateW, phi) + c.gateB;
      // Clamp to prevent overflow
      return clamp(logit, -50, 50);
    });
    const logZ = logSumExp(logits);
    return logits.map((z) => z - logZ);
  }

  /**
   * Online gradient ascent on log p(Δ|s):
   * For mixture models, the gradient of log-likelihood wrt logits is (r_k - π_k).
   * We do: w_k += lr * (r_k - π_k) * phi
   */
  private updateGating(phi: Vec, resp: number[]): void {
    const pis = softmax(this.comps.map((c) => dot(c.gateW, phi) + c.gateB)); // π_k
    const lr = this.cfg.gateLr;

    for (let k = 0; k < this.comps.length; k++) {
      const c = this.comps[k]!;
      const gradLogit = resp[k]! - pis[k]!; // r_k - π_k

      // Weight update with decay and clamping
      for (let i = 0; i < c.gateW.length; i++) {
        const w = c.gateW[i] ?? 0;
        const newW =
          (1 - this.cfg.gateWeightDecay) * w + lr * gradLogit * (phi[i] ?? 0);
        c.gateW[i] = clamp(isNaN(newW) ? 0 : newW, -10, 10);
      }
      const newB = (1 - this.cfg.gateWeightDecay) * c.gateB + lr * gradLogit;
      c.gateB = clamp(isNaN(newB) ? 0 : newB, -10, 10);
    }
  }

  /////////////////////////////
  // Expert refinement ops
  /////////////////////////////

  private narrow(k: number, deltaObs: Vec, phi: Vec, strength: number): void {
    const c = this.comps[k]!;
    c.seen += 1;

    // Add to reservoir for split detection
    this.addToReservoir(k, deltaObs, phi);

    // Mean update
    const lrM = clamp(this.cfg.meanLr * strength, 0, 1);
    c.mean = add(c.mean, mulScalar(sub(deltaObs, c.mean), lrM));

    // Variance update: shrink toward small error
    const err = sub(deltaObs, c.mean);
    const targetVar = err.map((e) => Math.max(this.cfg.minVar, 0.25 * e * e));
    const lrV = clamp(this.cfg.varLr * strength, 0, 1);
    c.varDiag = c.varDiag.map((v, i) =>
      clamp(v + lrV * (targetVar[i]! - v), this.cfg.minVar, this.cfg.maxVar)
    );

    // Check for bimodality and potentially split
    this.maybeSplit(k, phi);
  }

  private widen(k: number, deltaObs: Vec, phi: Vec, strength: number): void {
    const c = this.comps[k]!;
    c.seen += 1;

    // Add to reservoir for split detection
    this.addToReservoir(k, deltaObs, phi);

    // Mean moves a bit (optional)
    const lrM = clamp(this.cfg.meanLr * 0.5 * strength, 0, 1);
    c.mean = add(c.mean, mulScalar(sub(deltaObs, c.mean), lrM));

    // Variance update: increase toward squared error
    const err = sub(deltaObs, c.mean);
    const targetVar = err.map((e) => e * e);
    const lrV = clamp(this.cfg.varLr * strength, 0, 1);
    c.varDiag = c.varDiag.map((v, i) =>
      clamp(v + lrV * (targetVar[i]! - v), this.cfg.minVar, this.cfg.maxVar)
    );

    // Check for bimodality and potentially split
    this.maybeSplit(k, phi);
  }

  private proliferate(deltaObs: Vec, phi: Vec): void {
    // If near an existing mean, just widen the nearest
    let nearest = 0;
    let bestDist = Infinity;
    for (let i = 0; i < this.comps.length; i++) {
      const d = Math.sqrt(l2Sq(deltaObs, this.comps[i]!.mean));
      if (d < bestDist) {
        bestDist = d;
        nearest = i;
      }
    }
    if (bestDist < this.cfg.minComponentSeparationL2) {
      this.widen(nearest, deltaObs, phi, 1.0);
      return;
    }
    this.spawnComponent(deltaObs, phi, `spawn@${this.steps}`);
  }

  private spawnComponent(deltaObs: Vec, phi: Vec, name?: string): void {
    void name;
    const m = phi.length;

    const initVar = deltaObs.map(() => 1);
    const initW = new Array(m).fill(0);
    const initB = 0;

    this.comps.push({
      mean: [...deltaObs],
      varDiag: initVar,
      seen: 1,
      gateW: initW,
      gateB: initB,
      reservoir: [{ delta: [...deltaObs], phi: [...phi] }],
      lastSplitStep: 0,
    });

    // Bias the new component to be likely in the current context
    const boost = 0.5;
    const last = this.comps[this.comps.length - 1]!;
    for (let i = 0; i < last.gateW.length; i++) {
      const w = (last.gateW[i] ?? 0) + boost * (phi[i] ?? 0);
      last.gateW[i] = clamp(isNaN(w) ? 0 : w, -10, 10);
    }

    this.pruneIfNeeded();
  }

  private pruneIfNeeded(): void {
    if (this.comps.length <= this.cfg.maxComponents) return;

    // Heuristic: keep components with highest "seen" (usage) + smaller variance (more specific)
    const scored = this.comps.map((c, idx) => {
      const varSum = c.varDiag.reduce((a, b) => a + b, 0);
      const score = c.seen / Math.sqrt(varSum + 1e-9);
      return { idx, score };
    });

    scored.sort((a, b) => b.score - a.score);
    const keep = new Set(
      scored.slice(0, this.cfg.maxComponents).map((s) => s.idx)
    );
    this.comps = this.comps.filter((_, i) => keep.has(i));
  }

  /////////////////////////////
  // Split heuristic
  /////////////////////////////

  /** Add a sample to component's reservoir (ring buffer). */
  private addToReservoir(k: number, delta: Vec, phi: Vec): void {
    const c = this.comps[k]!;
    c.reservoir.push({ delta: [...delta], phi: [...phi] });
    if (c.reservoir.length > this.cfg.reservoirSize) {
      c.reservoir.shift();
    }
  }

  /** Check if component k should split; if so, perform the split. */
  private maybeSplit(k: number, phi: Vec): void {
    const c = this.comps[k]!;
    if (c.reservoir.length < this.cfg.splitMinSamples) return;
    if (c.seen % this.cfg.splitCheckInterval !== 0) return;
    if (this.comps.length >= this.cfg.maxComponents) return;

    // Cooldown: don't split a component that was recently split
    const cooldown = this.cfg.reservoirSize; // wait until reservoir refills
    if (this.steps - c.lastSplitStep < cooldown) return;

    const deltas = c.reservoir.map((r) => r.delta);
    const phis = c.reservoir.map((r) => r.phi);

    // Compute SSE for k=1 (current mean)
    const sse1 = this.computeSse(deltas, [c.mean]);
    if (sse1 < 1e-9) return; // already converged, no need to split

    // Run k-means with k=2
    const { centroids, assignments } = this.kMeans2(deltas);
    const sse2 = this.computeSse(deltas, centroids, assignments);

    // Split if k=2 significantly reduces SSE
    if (sse2 < this.cfg.splitSseRatio * sse1) {
      this.performSplit(k, centroids, assignments, phis);
    }
  }

  /** Compute SSE given deltas and centroids. */
  private computeSse(
    deltas: Vec[],
    centroids: Vec[],
    assignments?: number[]
  ): number {
    let sse = 0;
    for (let i = 0; i < deltas.length; i++) {
      const d = deltas[i]!;
      const cIdx = assignments ? assignments[i]! : 0;
      const c = centroids[cIdx]!;
      sse += l2Sq(d, c);
    }
    return sse;
  }

  /** Simple k-means with k=2, few iterations. */
  private kMeans2(deltas: Vec[]): { centroids: Vec[]; assignments: number[] } {
    const n = deltas.length;
    const dim = deltas[0]!.length;

    // Initialize: pick two furthest points
    let maxDist = -1;
    let c0Idx = 0,
      c1Idx = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = l2Sq(deltas[i]!, deltas[j]!);
        if (d > maxDist) {
          maxDist = d;
          c0Idx = i;
          c1Idx = j;
        }
      }
    }

    let c0 = [...deltas[c0Idx]!];
    let c1 = [...deltas[c1Idx]!];
    let assignments = new Array(n).fill(0);

    // Run a few iterations
    for (let iter = 0; iter < 5; iter++) {
      // Assign
      for (let i = 0; i < n; i++) {
        const d0 = l2Sq(deltas[i]!, c0);
        const d1 = l2Sq(deltas[i]!, c1);
        assignments[i] = d0 <= d1 ? 0 : 1;
      }

      // Recompute centroids
      const sum0 = new Array(dim).fill(0);
      const sum1 = new Array(dim).fill(0);
      let cnt0 = 0,
        cnt1 = 0;

      for (let i = 0; i < n; i++) {
        const target = assignments[i] === 0 ? sum0 : sum1;
        for (let d = 0; d < dim; d++) target[d] += deltas[i]![d]!;
        if (assignments[i] === 0) cnt0++;
        else cnt1++;
      }

      if (cnt0 > 0) c0 = sum0.map((v) => v / cnt0);
      if (cnt1 > 0) c1 = sum1.map((v) => v / cnt1);
    }

    return { centroids: [c0, c1], assignments };
  }

  /** Split component k into two components based on k-means result. */
  private performSplit(
    k: number,
    centroids: Vec[],
    assignments: number[],
    phis: Vec[]
  ): void {
    const old = this.comps[k]!;
    const embedDim = old.gateW.length;

    // Partition reservoir samples
    const samples0: Array<{ delta: Vec; phi: Vec }> = [];
    const samples1: Array<{ delta: Vec; phi: Vec }> = [];
    for (let i = 0; i < assignments.length; i++) {
      const sample = { delta: old.reservoir[i]!.delta, phi: phis[i]! };
      if (assignments[i] === 0) samples0.push(sample);
      else samples1.push(sample);
    }

    // Need at least 2 samples in each cluster for a meaningful split
    if (samples0.length < 2 || samples1.length < 2) return;

    // Compute variance for each new cluster
    const var0 = this.computeClusterVar(
      samples0.map((s) => s.delta),
      centroids[0]!
    );
    const var1 = this.computeClusterVar(
      samples1.map((s) => s.delta),
      centroids[1]!
    );

    // Compute average phi for gating initialization
    const avgPhi0 = this.averageVecs(samples0.map((s) => s.phi));
    const avgPhi1 = this.averageVecs(samples1.map((s) => s.phi));

    // Compute phi difference to find discriminative dimensions
    const phiDiff = sub(avgPhi0, avgPhi1);

    // Replace old component with first cluster
    old.mean = [...centroids[0]!];
    old.varDiag = var0;
    old.seen = samples0.length;
    old.reservoir = []; // Clear reservoir to start fresh
    old.lastSplitStep = this.steps; // Set cooldown
    // Set gating to prefer contexts similar to cluster 0
    for (let i = 0; i < embedDim; i++) {
      old.gateW[i] = phiDiff[i]!;
    }
    old.gateB = 0;

    // Create new component for second cluster
    const newGateW = new Array(embedDim).fill(0);
    for (let i = 0; i < embedDim; i++) {
      newGateW[i] = -phiDiff[i]!;
    }

    this.comps.push({
      mean: [...centroids[1]!],
      varDiag: var1,
      seen: samples1.length,
      gateW: newGateW,
      gateB: 0,
      reservoir: [], // Clear reservoir
      lastSplitStep: this.steps, // Set cooldown
    });
  }

  /** Compute diagonal variance for a cluster. */
  private computeClusterVar(deltas: Vec[], centroid: Vec): Vec {
    const dim = centroid.length;
    const varDiag = new Array(dim).fill(0);

    for (const d of deltas) {
      for (let i = 0; i < dim; i++) {
        const diff = d[i]! - centroid[i]!;
        varDiag[i] += diff * diff;
      }
    }

    const n = Math.max(deltas.length, 1);
    // Ensure minimum variance of 0.1 to prevent log(0) issues
    const minInitVar = Math.max(this.cfg.minVar, 0.1);
    return varDiag.map((v) =>
      clamp(v / n + minInitVar, minInitVar, this.cfg.maxVar)
    );
  }

  /** Compute average of vectors. */
  private averageVecs(vecs: Vec[]): Vec {
    if (vecs.length === 0) return [];
    const dim = vecs[0]!.length;
    const sum = new Array(dim).fill(0);
    for (const v of vecs) {
      for (let i = 0; i < dim; i++) sum[i] += v[i]!;
    }
    return sum.map((s) => s / vecs.length);
  }

  /////////////////////////////
  // Dim checks
  /////////////////////////////

  private ensureDims(phi: Vec, delta: Vec): void {
    if (this.embedDim == null) this.embedDim = phi.length;
    if (this.deltaDim == null) this.deltaDim = delta.length;

    if (phi.length !== this.embedDim) {
      throw new Error(
        `embedSituation dim changed: expected ${this.embedDim}, got ${phi.length}`
      );
    }
    if (delta.length !== this.deltaDim) {
      throw new Error(
        `delta dim changed: expected ${this.deltaDim}, got ${delta.length}`
      );
    }

    // Ensure existing components have consistent dims (useful if you seed differently)
    for (const c of this.comps) {
      if (c.mean.length !== this.deltaDim)
        throw new Error("Component mean dim mismatch");
      if (c.varDiag.length !== this.deltaDim)
        throw new Error("Component var dim mismatch");
      if (c.gateW.length !== this.embedDim)
        throw new Error("Component gateW dim mismatch");
    }
  }
}

//////////////////////////////
// PortBank for many actions
//////////////////////////////

export class ContextualPortBank<S, C = unknown> {
  private ports = new Map<string, ContextualMixtureDeltaPort<S, C>>();

  constructor(
    private readonly enc: Encoders<S, C>,
    private readonly cfg?: Partial<RefinementConfig>
  ) {}

  get(action: string): ContextualMixtureDeltaPort<S, C> {
    const p = this.ports.get(action);
    if (p) return p;

    const created = new ContextualMixtureDeltaPort<S, C>({
      action,
      encoders: this.enc,
      config: this.cfg,
    });
    this.ports.set(action, created);
    return created;
  }

  observe(tr: Transition<S, C>, reward?: number): void {
    this.get(tr.action).observe(tr, reward);
  }

  predict(action: string, sit: Situation<S, C>, k?: number) {
    return this.get(action).predictDeltas(sit, k);
  }

  score(action: string, sit: Situation<S, C>, delta: Vec) {
    return this.get(action).scoreDelta(sit, delta);
  }

  snapshot(): unknown {
    return {
      ports: [...this.ports.entries()].map(([a, p]) => ({
        action: a,
        snapshot: p.snapshot(),
      })),
    };
  }
}

//////////////////////////////
// Example: tiny grid "UP"
//////////////////////////////

export type SimpleGridState = { x: number; y: number; blockedAbove?: boolean };

export const simpleGridEncoders: Encoders<SimpleGridState, {}> = {
  // Situation embedding φ(s): include features that affect action semantics/feasibility
  // In real boards: object centroids, selected-id, cursor pos, phase flags, etc.
  embedSituation: (sit) => [
    sit.state.x,
    sit.state.y,
    sit.state.blockedAbove ? 1 : 0,
  ],
  // Δ = (dx, dy)
  delta: (before, after) => [
    after.state.x - before.state.x,
    after.state.y - before.state.y,
  ],
};

/*
USAGE:

const bank = new ContextualPortBank(simpleGridEncoders, {
  goodFitLogLikeThreshold: -15,
  noveltyLogLikeThreshold: -60,
  maxComponents: 4,
});

bank.observe({
  action: "UP",
  before: { state: { x: 3, y: 5, blockedAbove: false }, context: {} },
  after:  { state: { x: 3, y: 4, blockedAbove: false }, context: {} },
});

bank.observe({
  action: "UP",
  before: { state: { x: 8, y: 1, blockedAbove: false }, context: {} },
  after:  { state: { x: 8, y: 0, blockedAbove: false }, context: {} },
});

// Blocked case: same action yields Δ=(0,0) in contexts with blockedAbove=1
bank.observe({
  action: "UP",
  before: { state: { x: 2, y: 0, blockedAbove: true }, context: {} },
  after:  { state: { x: 2, y: 0, blockedAbove: true }, context: {} },
});

// Now the gating model can learn: when blockedAbove=1, favor the (0,0) component;
// otherwise favor (0,-1).
console.log(bank.predict("UP", { state: { x: 7, y: 10, blockedAbove: false }, context: {} }, 2));
console.log(bank.predict("UP", { state: { x: 7, y: 0, blockedAbove: true },  context: {} }, 2));

console.log(JSON.stringify(bank.snapshot(), null, 2));
*/
