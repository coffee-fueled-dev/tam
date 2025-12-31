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
 * Sigmoid activation function: σ(x) = 1 / (1 + e^(-x))
 * Maps real values to [0, 1] range.
 */
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * GeometricPort: combines embedding with shared network references.
 */
export class GeometricPort<S, C = unknown> implements Port<S, C> {
  public readonly action: string;
  public readonly name: string;
  public readonly id: string;
  public embedding: Vec; // Mutable - trained via binding outcomes
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
   * Compute per-dimension alignment between port and situation.
   * Returns a vector of alignment values [0,1] for each dimension.
   *
   * This enables anisotropic cones where the port attends more strongly
   * to some dimensions than others, implementing fiber-based dimensional attention.
   *
   * Uses element-wise product with sigmoid activation:
   *   alignment[i] = sigmoid(port.embedding[i] × state[i])
   *
   * High embedding value → high sensitivity to that dimension
   * Low embedding value → dimension is ignored (wide cone on that axis)
   */
  private computePerDimAlignment(stateEmb: Vec): Vec {
    const portDim = this.embedding.length;
    const stateDim = stateEmb.length;

    // Handle dimension mismatch via projection
    if (portDim !== stateDim) {
      // Project to common dimension
      const portProj = portDim > stateDim
        ? this.projectToSize(this.embedding, stateDim)
        : this.embedding;

      const dim = Math.min(portDim, stateDim);
      // Use softmax for relative attention (prevents saturation, enforces competition)
      return this.softmax(portProj.slice(0, dim));
    }

    // Dimensions match - compute per-dimension alignment via softmax
    // Softmax enforces relative attention: increasing attention to some dimensions
    // automatically decreases attention to others (zero-sum game)
    // Prevents sigmoid saturation where all alignments → 1.0
    return this.softmax(this.embedding);
  }

  /**
   * Softmax activation for computing relative dimensional attention.
   * Returns a probability distribution over dimensions.
   *
   * Uses temperature scaling to control concentration:
   * - High temperature → more uniform distribution (less extreme)
   * - Low temperature → more peaked distribution (more extreme)
   */
  private softmax(vec: Vec): Vec {
    // Temperature parameter: higher = smoother attention distribution
    // With 10D, we want moderate differentiation, not extreme concentration
    const temperature = 2.0;

    // Scale by temperature before softmax
    const scaled = vec.map((x) => x / temperature);

    // Numerical stability: subtract max before exp
    const maxVal = Math.max(...scaled);
    const exps = scaled.map((x) => Math.exp(x - maxVal));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sumExps);
  }

  /**
   * Compute global alignment between port and situation (scalar).
   * α = max(0, cos(θ)) where θ is the angle between port and situation embeddings.
   * Returns 0 when orthogonal (port doesn't apply), 1 when perfectly aligned.
   *
   * When port and state have different dimensions, we project the larger
   * to the smaller using average pooling. This preserves directional structure
   * without information loss for distance/similarity computation.
   *
   * @deprecated Use computePerDimAlignment() for anisotropic cones
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
   * Uses the geometric formula with anisotropic (per-dimension) radii:
   *   radius[i] = aperture × α[i] / (1 + d)
   *
   * Where:
   *   α[i] = per-dimension alignment (fiber-based dimensional attention)
   *   d = distance (scalar commitment level from CommitmentNet)
   *
   * This creates elliptical cones that attend more strongly to some dimensions.
   */
  getCone(stateEmb: Vec): Cone {
    // Fibration: project to trajectory space (center of cone)
    const center = this.causalNet.predict(stateEmb, this.embedding);

    // Per-dimension alignment: dimensional attention
    const alignments = this.computePerDimAlignment(stateEmb);

    // Distance: scalar commitment level (higher = more committed = narrower cone)
    const distance = this.commitmentNet.predictDistance(
      stateEmb,
      this.embedding
    );

    // Cache for diagnostics (use mean alignment for backward compatibility)
    this.lastAlignment = alignments.reduce((sum, a) => sum + a, 0) / alignments.length;
    this.lastDistance = distance;

    // Anisotropic radii = aperture × alignment[i] / (1 + distance)
    // Each dimension gets its own radius based on how much the port attends to it
    // High alignment[i] → narrow radius on dimension i (port cares about this dim)
    // Low alignment[i] → wide radius on dimension i (port ignores this dim)
    const effectiveRadius = alignments.map(a => this.aperture * a / (1 + distance));

    // Include alignment in cone for alignment-weighted binding evaluation
    return { center, radius: effectiveRadius, alignment: alignments };
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

    // Train embedding based on per-dimension binding performance
    this.trainEmbedding(stateEmb, actualDelta);

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
   * Train port embedding based on per-dimension binding outcomes.
   *
   * Goal: Learn which dimensions this port should attend to.
   * - Increase embedding[i] for dimensions where predictions are good
   * - Decrease embedding[i] for dimensions where predictions are bad
   *
   * With state-independent alignment (alignment[i] = sigmoid(embedding[i])):
   * - To increase attention: embedding[i] += learningRate
   * - To decrease attention: embedding[i] -= learningRate
   */
  private trainEmbedding(stateEmb: Vec, actualDelta: Vec): void {
    // Get current cone to evaluate per-dimension performance
    const cone = this.getCone(stateEmb);
    const prediction = cone.center;
    const radii = cone.radius;

    const learningRate = this.cfg.embeddingLearningRate;
    const dim = Math.min(this.embedding.length, stateEmb.length);

    // Update embedding based on per-dimension success/failure
    for (let i = 0; i < dim; i++) {
      // Per-dimension error
      const error = Math.abs((actualDelta[i] ?? 0) - (prediction[i] ?? 0));
      const radius = radii[i] ?? 1.0;
      const success = error < radius;

      if (success) {
        // Prediction was good → increase attention to this dimension
        this.embedding[i] = (this.embedding[i] ?? 0) + learningRate;
      } else {
        // Prediction was bad → decrease attention to this dimension
        // Use smaller step for decreases to bias toward attending
        this.embedding[i] = (this.embedding[i] ?? 0) - learningRate * 0.5;
      }
    }
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
   * Compute agency from commitment level (distance), not cone volume.
   *
   * Agency measures COMMITMENT SPECIFICITY (learned confidence),
   * NOT affordance specificity (viewing angle/alignment).
   *
   * Alignment still affects cone radius and binding success,
   * but doesn't inflate agency. This prevents geometric mismatch
   * from being confused with epistemic certainty.
   *
   * Agency = distance / (1 + distance)
   * - distance = 0: agency = 0 (no commitment, maximally uncertain)
   * - distance → ∞: agency → 1 (maximum commitment)
   */
  private computeAgency(cone: Cone): number {
    // Check for empty cone (alignment = 0 → all radii = 0)
    const anyNonZero = cone.radius.some((r) => r > 1e-8);
    if (!anyNonZero) return 0; // Empty cone = no agency (port doesn't apply)

    // Agency based on distance (commitment level), not final volume
    const distance = this.lastDistance ?? 0;
    return Math.min(1, distance / (1 + distance));
  }

  /**
   * Convert state embedding to a discretized key for history tracking.
   */
  private situationToKey(stateEmb: Vec): string {
    // Quantize to reduce key space
    return stateEmb.map((v) => Math.round(v * 10) / 10).join(",");
  }

  /**
   * Get last computed distance (for diagnostics).
   */
  getLastDistance(): number {
    return this.lastDistance;
  }

  /**
   * Get last computed alignment (for diagnostics).
   */
  getLastAlignment(): number {
    return this.lastAlignment;
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
