/**
 * GeometricPort: A port in the geometric TAM architecture.
 *
 * Geometric interpretation (window analogy):
 * - Port Space (P): Where ports live (the "blank wall")
 * - Trajectory Space (T): Possible outcomes (the "outside world")
 * - Situation Space (X): Agent's context (viewing position)
 *
 * A port is like a window in the wall. The situation determines:
 * - Viewing angle (α): alignment between port and situation embeddings
 * - Viewing distance (d): commitment level (from CommitmentNet)
 *
 * The cone (what you can see through the window) is computed as:
 *   center: π(p, x)                           ← CausalNet (fibration)
 *   radius: aperture × α / (1 + d)            ← scaled by geometry
 *
 * This implements polysemy: one port has different cones for different situations.
 *
 * The full binding cycle:
 * 1. Bind: commit to a cone Φ_p(situation)
 * 2. World responds with actual trajectory
 * 3. Infer: interpret episode as trajectory (done by encoders)
 * 4. Evaluate: check if trajectory ∈ cone
 * 5. Refine: update based on binding outcome
 */

import type {
  Port,
  Prediction,
  Situation,
  Transition,
  Encoders,
  Cone,
  BindingOutcome,
  RefinementAction,
  GeometricPortConfig,
  GeometricPortId,
} from "../types";
import { defaultGeometricPortConfig } from "../types";
import type { Vec } from "../vec";
import { cosineSimilarity } from "../vec";
import { CausalNet } from "./causal";
import { CommitmentNet } from "./commitment";
import { assembleCone, evaluateBinding, coneLogVolume } from "./fibration";
import { DefaultBindingHistory } from "./history";
import { FixedRefinementPolicy } from "./refinement";

/**
 * GeometricPort: combines embedding with shared network references.
 */
export class GeometricPort<S, C = unknown> implements Port<S, C> {
  public readonly action: string;
  public readonly name: string;
  public readonly id: string;
  public readonly embedding: Vec;
  public readonly aperture: number;

  private readonly enc: Encoders<S, C>;
  private readonly causalNet: CausalNet;
  private readonly commitmentNet: CommitmentNet;
  private readonly history: DefaultBindingHistory;
  private readonly policy: FixedRefinementPolicy;
  private readonly cfg: GeometricPortConfig;

  // Reference volume for agency calculation (shared across bank)
  private referenceVolume: number;

  private steps = 0;
  private totalSuccesses = 0;
  private lastCone: Cone | null = null;
  private lastOutcome: BindingOutcome | null = null;
  private lastSituation: Situation<S, C> | null = null;
  private lastAlignment: number = 0;
  private lastDistance: number = 0;

  constructor(opts: {
    action: string;
    name?: string;
    id?: string;
    embedding?: Vec;
    aperture?: number;
    encoders: Encoders<S, C>;
    causalNet: CausalNet;
    commitmentNet: CommitmentNet;
    history?: DefaultBindingHistory;
    policy?: FixedRefinementPolicy;
    config?: Partial<GeometricPortConfig>;
    referenceVolume?: number;
  }) {
    this.action = opts.action;
    this.id = opts.id ?? `port_${Math.random().toString(36).slice(2, 8)}`;
    this.name = opts.name ?? `GeometricPort(${this.action})`;

    this.cfg = { ...defaultGeometricPortConfig, ...(opts.config ?? {}) };
    this.aperture = opts.aperture ?? this.cfg.defaultAperture;
    this.referenceVolume = opts.referenceVolume ?? this.cfg.referenceVolume;

    // Initialize embedding (random if not provided)
    this.embedding =
      opts.embedding ??
      Array.from(
        { length: this.cfg.embeddingDim },
        () => (Math.random() - 0.5) * 0.1
      );

    this.enc = opts.encoders;
    this.causalNet = opts.causalNet;
    this.commitmentNet = opts.commitmentNet;
    this.history =
      opts.history ??
      new DefaultBindingHistory({
        familiarityThreshold: this.cfg.familiarityThreshold,
      });
    this.policy = opts.policy ?? new FixedRefinementPolicy(this.cfg);
  }

  /**
   * Update reference volume (called by bank when dynamically calibrating).
   */
  setReferenceVolume(ref: number): void {
    this.referenceVolume = ref;
  }

  /**
   * Compute alignment between port and situation.
   * α = max(0, cos(θ)) where θ is the angle between port and situation embeddings.
   * Returns 0 when orthogonal (port doesn't apply), 1 when perfectly aligned.
   *
   * When port and state have different dimensions, we project the larger
   * to the smaller using average pooling. This preserves directional structure
   * without information loss for distance/similarity computation.
   */
  private computeAlignment(stateEmb: Vec): number {
    const portDim = this.embedding.length;
    const stateDim = stateEmb.length;

    // If dimensions match, use direct cosine similarity
    if (portDim === stateDim) {
      return Math.max(0, cosineSimilarity(this.embedding, stateEmb));
    }

    // Project the larger to match the smaller via average pooling
    let portProj: Vec;
    let stateProj: Vec;

    if (portDim > stateDim) {
      // Project port down to state dimension
      portProj = this.projectToSize(this.embedding, stateDim);
      stateProj = stateEmb;
    } else {
      // Project state down to port dimension
      portProj = this.embedding;
      stateProj = this.projectToSize(stateEmb, portDim);
    }

    return Math.max(0, cosineSimilarity(portProj, stateProj));
  }

  /**
   * Project a vector to a target size using average pooling.
   * Divides the vector into targetSize chunks and averages each.
   * Preserves directional structure for cosine similarity.
   */
  private projectToSize(vec: Vec, targetSize: number): Vec {
    if (vec.length === targetSize) return vec;
    if (vec.length < targetSize) {
      // Can't expand - pad with zeros (shouldn't happen in normal flow)
      return [...vec, ...new Array(targetSize - vec.length).fill(0)];
    }

    // Average pooling: divide into targetSize segments
    const result: Vec = [];
    const chunkSize = vec.length / targetSize;

    for (let i = 0; i < targetSize; i++) {
      const start = Math.floor(i * chunkSize);
      const end = Math.floor((i + 1) * chunkSize);
      let sum = 0;
      for (let j = start; j < end; j++) {
        sum += vec[j] ?? 0;
      }
      result.push(sum / (end - start));
    }

    return result;
  }

  /**
   * Get the cone for a state embedding (polysemous: depends on situation).
   * PRIMARY INTERFACE - operates in embedding space.
   *
   * Uses the geometric formula:
   *   radius = aperture × α / (1 + d)
   *
   * Where:
   *   α = alignment (cosine similarity between port and situation)
   *   d = distance (commitment level from CommitmentNet)
   */
  getCone(stateEmb: Vec): Cone {
    // Fibration: project to trajectory space (center of cone)
    const center = this.causalNet.predict(stateEmb, this.embedding);

    // Alignment: viewing angle factor
    const alignment = this.computeAlignment(stateEmb);

    // Distance: commitment level (higher = more committed = narrower cone)
    const distance = this.commitmentNet.predictDistance(
      stateEmb,
      this.embedding
    );

    // Cache for diagnostics
    this.lastAlignment = alignment;
    this.lastDistance = distance;

    // Effective radius = aperture × alignment / (1 + distance)
    // When alignment = 0, radius = 0 (empty cone, port doesn't apply)
    // When distance → ∞, radius → 0 (very narrow cone, high agency)
    const scale = alignment / (1 + distance);
    const effectiveRadius = center.map(() => this.aperture * scale);

    return assembleCone(center, effectiveRadius);
  }

  /**
   * Get the cone for a situation (convenience wrapper).
   * Encodes the situation then calls getCone(embedding).
   */
  getConeFromState(sit: Situation<S, C>): Cone {
    const stateEmb = this.enc.embedSituation(sit);
    return this.getCone(stateEmb);
  }

  /**
   * Predict outcomes from a state embedding.
   * PRIMARY INTERFACE - operates in embedding space.
   * Returns the predicted trajectory as a single prediction with agency.
   */
  predict(stateEmb: Vec, _k?: number): Prediction[] {
    const cone = this.getCone(stateEmb);
    const agency = this.computeAgency(cone);

    return [
      {
        delta: cone.center,
        score: 0, // Single mode
        agency,
        cone,
      },
    ];
  }

  /**
   * Predict outcomes for a situation (convenience wrapper).
   * Encodes the situation then calls predict(embedding).
   */
  predictFromState(sit: Situation<S, C>, k?: number): Prediction[] {
    const stateEmb = this.enc.embedSituation(sit);
    return this.predict(stateEmb, k);
  }

  /**
   * Observe a transition: full binding cycle.
   */
  observe(tr: Transition<S, C>): void {
    if (tr.action !== this.action) return;

    this.steps++;

    const stateEmb = this.enc.embedSituation(tr.before);
    const actualDelta = this.enc.delta(tr.before, tr.after);
    const situationKey = this.situationToKey(stateEmb);

    // 1. Get cone (commitment) - use embedding directly
    const cone = this.getCone(stateEmb);
    this.lastCone = cone;
    this.lastSituation = tr.before;

    // 2. Evaluate binding predicate
    const outcome = evaluateBinding(actualDelta, cone);
    this.lastOutcome = outcome;

    if (outcome.success) {
      this.totalSuccesses++;
    } else {
      // Record failure trajectory for bimodal detection
      this.history.recordFailureTrajectory(this.id, actualDelta);
    }

    // 3. Record outcome in history
    this.history.record(this.id, situationKey, outcome);

    // 4. Decide refinement action
    const action = this.policy.decide(
      outcome,
      this.id,
      situationKey,
      this.history
    );

    // 5. Apply refinement
    this.applyRefinement(stateEmb, actualDelta, outcome, action);
  }

  /**
   * Apply refinement action to update networks.
   */
  private applyRefinement(
    stateEmb: Vec,
    actualDelta: Vec,
    outcome: BindingOutcome,
    action: RefinementAction
  ): void {
    // Always train CausalNet toward actual trajectory (learns the causal manifold)
    this.causalNet.observe(stateEmb, this.embedding, actualDelta);

    // Train CommitmentNet based on refinement action
    if (action === "narrow") {
      this.commitmentNet.queueRefinement(stateEmb, this.embedding, "narrow");
    } else if (action === "widen") {
      const violation = outcome.success ? 0 : outcome.violation;
      this.commitmentNet.queueRefinement(
        stateEmb,
        this.embedding,
        "widen",
        violation
      );
    }
    // "noop" and "proliferate" don't train CommitmentNet for this port
  }

  /**
   * Get overall agency of this port (using last observed situation).
   * Computes a fresh cone from the current network state.
   */
  getAgency(): number {
    if (!this.lastSituation) return 0;
    return this.computeAgencyForState(this.lastSituation);
  }

  /**
   * Compute agency for an embedding (not just last observed).
   * PRIMARY INTERFACE - operates in embedding space.
   *
   * Agency = 1 - (cone volume / reference volume)
   * High agency = narrow cone = specific commitment
   * Low agency = wide cone = vague commitment
   * Zero agency = empty cone = port doesn't apply
   */
  computeAgencyFor(stateEmb: Vec): number {
    const cone = this.getCone(stateEmb);
    return this.computeAgency(cone);
  }

  /**
   * Compute agency for a situation (convenience wrapper).
   * Encodes the situation then calls computeAgencyFor(embedding).
   */
  computeAgencyForState(sit: Situation<S, C>): number {
    const stateEmb = this.enc.embedSituation(sit);
    return this.computeAgencyFor(stateEmb);
  }

  /**
   * Compute agency from a cone.
   * Agency = 1 - volume/referenceVolume (clamped to [0, 1])
   *
   * Uses dynamic reference volume from bank (total trajectory space).
   */
  private computeAgency(cone: Cone): number {
    // Check for empty cone (alignment = 0 → all radii = 0)
    const anyNonZero = cone.radius.some((r) => r > 1e-8);
    if (!anyNonZero) return 0; // Empty cone = no agency (port doesn't apply)

    const logVolume = coneLogVolume(cone);
    const refLogVolume = cone.radius.length * Math.log(this.referenceVolume);
    const logRatio = logVolume - refLogVolume;
    const ratio = Math.exp(logRatio / cone.radius.length); // Geometric mean
    return Math.max(0, Math.min(1, 1 - ratio));
  }

  /**
   * Convert state embedding to a discretized key for history tracking.
   */
  private situationToKey(stateEmb: Vec): string {
    // Quantize to reduce key space
    return stateEmb.map((v) => Math.round(v * 10) / 10).join(",");
  }

  /**
   * Get diagnostic snapshot.
   */
  snapshot(): unknown {
    return {
      id: this.id,
      action: this.action,
      name: this.name,
      embedding: this.embedding,
      aperture: this.aperture,
      steps: this.steps,
      successRate: this.steps > 0 ? this.totalSuccesses / this.steps : 0,
      lastCone: this.lastCone,
      lastOutcome: this.lastOutcome,
      lastAlignment: this.lastAlignment,
      lastDistance: this.lastDistance,
      agency: this.getAgency(),
      referenceVolume: this.referenceVolume,
    };
  }

  /**
   * Get the port's identity for proliferation.
   */
  getPortId(): GeometricPortId {
    return {
      id: this.id,
      action: this.action,
      embedding: [...this.embedding],
      aperture: this.aperture,
    };
  }

  /**
   * Check if this port is applicable (non-empty cone) for an embedding.
   * PRIMARY INTERFACE - operates in embedding space.
   */
  isApplicable(stateEmb: Vec): boolean {
    const alignment = this.computeAlignment(stateEmb);
    return alignment >= this.cfg.minAlignmentThreshold;
  }

  /**
   * Check if this port is applicable for a situation (convenience wrapper).
   */
  isApplicableForState(sit: Situation<S, C>): boolean {
    const stateEmb = this.enc.embedSituation(sit);
    return this.isApplicable(stateEmb);
  }

  /**
   * Get the step count for this port.
   */
  getSteps(): number {
    return this.steps;
  }

  /**
   * Flush any buffered training data.
   */
  flush(): void {
    this.causalNet.flush();
    this.commitmentNet.flush();
  }

  /**
   * No resources owned directly - shared networks are managed by PortBank.
   */
  dispose(): void {
    // Nothing to dispose - shared networks are managed by PortBank
  }
}
