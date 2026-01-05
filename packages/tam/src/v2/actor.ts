/**
 * Actor: Main TAM v2 class with latent port proliferation
 *
 * Key simplification: Ports are just embeddings (Vec) in port manifold,
 * not objects with methods. Port selection, proliferation, and management
 * all operate directly on embedding vectors.
 *
 * Self-organizing homeostatic architecture (unified geometric system):
 * - CausalNet predicts both trajectory mean AND uncertainty (dual-head)
 * - Concentration derived geometrically from prediction variance
 * - Soft conditional NLL (one-sided bounded) creates homeostatic feedback loop
 *
 * Complete negative feedback loop creates equilibrium:
 *   variance ↓ → concentration ↑ → cones narrow
 *                                       ↓
 *                                   binding ↓
 *                                       ↓
 *                         min variance penalty activated
 *                         (push above floor, one-sided)
 *                                       ↓
 *                                  variance ↑
 *
 * The system self-stabilizes at domain-dependent equilibrium:
 * - Low-noise domains: variance shrinks freely, high binding rate, high agency
 * - High-noise/chaotic domains: variance floor prevents overconfidence, moderate binding
 * - MinVariance parameter (default 5.0) can be tuned per domain
 * - One-sided penalty: prevents overconfidence without limiting calibration
 *
 * Core loop:
 * 1. Select port embedding for current state
 * 2. Predict trajectory + uncertainty via CausalNet
 * 3. Compute concentration geometrically from variance
 * 4. Evaluate binding strength
 * 5. Train CausalNet with soft conditional NLL (one-sided bounded)
 * 6. Update port embeddings via gradient descent
 * 7. Proliferate new port embeddings when needed
 */

import type {
  Vec,
  Situation,
  Transition,
  Prediction,
  Cone,
  ActorConfig,
  PortSelectionPolicy,
  RefinementPolicy,
  CausalNetConfig,
} from "./types";
import { defaultActorConfig } from "./types";
import { CausalNet } from "./causal";
import { FixedPortSelectionPolicy, FixedRefinementPolicy } from "./policies";
import Queue from "queue";

/**
 * Vector utilities
 */
function sub(a: Vec, b: Vec): Vec {
  return a.map((val, i) => val - b[i]!);
}

function magnitude(v: Vec): number {
  return Math.sqrt(v.reduce((sum, val) => sum + val * val, 0));
}

function dot(a: Vec, b: Vec): number {
  return a.reduce((sum, val, i) => sum + val * b[i]!, 0);
}

/**
 * Port tracking for trajectory clustering and loss plateau detection
 */
interface PortStats {
  samples: number;
  bindings: number;             // Count of successful bindings (binary)
  weightedBindings: number;     // Agency-weighted binding accumulator
  recentTrajectories: Vec[];    // Actual deltas (for clustering)
  recentErrors: number[];       // Prediction MSE (for plateau detection)
  recentAgencies: number[];     // Agency values when this port is selected
  lastProliferationSample: number;  // Sample count at last proliferation
  creationSample: number;       // Total samples when this port was created
}

export class Actor<S> {
  // Port manifold: just embeddings, no objects
  private portEmbeddings: Vec[] = [];
  private portStats = new Map<number, PortStats>(); // index → stats

  // Global sample counter (across all ports, for pruning decisions)
  private totalSamples: number = 0;

  // Neural network
  private causalNet: CausalNet;

  // Shared training queue
  private trainingQueue: Queue;

  // State embedding function
  private embedState: (s: S) => Vec;

  // Configuration
  private config: ActorConfig;

  // Policies (strategy pattern)
  private portSelectionPolicy: PortSelectionPolicy;
  private refinementPolicy: RefinementPolicy;

  constructor(embedState: (s: S) => Vec, config?: Partial<ActorConfig>) {
    this.embedState = embedState;

    // Deep merge configuration to ensure nested objects are properly combined
    this.config = {
      initialPorts: config?.initialPorts ?? ([] as Vec[]),
      causal: { ...defaultActorConfig.causal, ...config?.causal } as CausalNetConfig,
      proliferation: { ...defaultActorConfig.proliferation, ...config?.proliferation },
      exploration: { ...defaultActorConfig.exploration, ...config?.exploration },
      portSelectionPolicy: config?.portSelectionPolicy,
      refinementPolicy: config?.refinementPolicy,
    };

    // Initialize policies with fixed defaults if not provided
    const explorationRate = this.config.exploration?.rate ?? 0.1;
    this.portSelectionPolicy =
      config?.portSelectionPolicy ?? new FixedPortSelectionPolicy(explorationRate);
    this.refinementPolicy = config?.refinementPolicy ?? new FixedRefinementPolicy();

    // Create shared training queue
    this.trainingQueue = new Queue({ autostart: true, concurrency: 1 });

    this.causalNet = new CausalNet(this.trainingQueue, this.config.causal);

    // Initialize with seed ports if provided
    if (this.config.initialPorts && this.config.initialPorts.length > 0) {
      this.portEmbeddings = this.config.initialPorts.map(p => [...p]);
      for (let i = 0; i < this.portEmbeddings.length; i++) {
        this.portStats.set(i, {
          samples: 0,
          bindings: 0,
          weightedBindings: 0,
          recentTrajectories: [],
          recentErrors: [],
          recentAgencies: [],
          lastProliferationSample: 0,
          creationSample: 0,
        });
      }
    }
  }

  /**
   * Predict for a given state.
   */
  predict(state: S): Prediction {
    const stateEmb = this.embedState(state);

    // Select best port
    const portIdx = this.selectPort(stateEmb);
    let portEmb: Vec;

    if (portIdx === -1) {
      // No ports yet - create one
      portEmb = this.createPort(stateEmb);
    } else {
      portEmb = this.portEmbeddings[portIdx]!;
    }

    // Get prediction with uncertainty
    const {mean: delta, variance} = this.causalNet.predict(stateEmb, portEmb);

    // Compute concentration geometrically from prediction uncertainty
    const concentration = this.concentrationFromVariance(variance);

    // Build affordance cone (anisotropic: per-dimension variance → per-dimension radius)
    const cone = this.buildCone(delta, variance, concentration, stateEmb);

    // Compute agency (monotonic transform of concentration)
    const agency = concentration / (1 + concentration);

    return { delta, cone, agency };
  }

  /**
   * Observe a transition and train.
   */
  async observe(transition: Transition<S>): Promise<void> {
    const beforeEmb = this.embedState(transition.before.state);
    const afterEmb = this.embedState(transition.after.state);
    const actualDelta = sub(afterEmb, beforeEmb);

    // Select or create port
    let portIdx = this.selectPort(beforeEmb);
    if (portIdx === -1) {
      this.createPort(beforeEmb);
      portIdx = this.portEmbeddings.length - 1;
    }

    const portEmb = this.portEmbeddings[portIdx]!;

    // Get prediction with uncertainty
    const {mean: delta, variance} = this.causalNet.predict(beforeEmb, portEmb);

    // Compute concentration geometrically from prediction uncertainty
    const concentration = this.concentrationFromVariance(variance);

    const cone = this.buildCone(delta, variance, concentration, beforeEmb);

    // Evaluate binding with soft boundaries
    const bindingResult = this.evaluateSoftBinding(actualDelta, cone);
    const bound = bindingResult.bound; // Binary for stats
    const bindingStrength = bindingResult.strength; // Continuous [0, 1]
    const normalizedDistance = bindingResult.normalizedDistance;

    // Compute prediction error for plateau detection
    const predictionError = magnitude(sub(actualDelta, delta));

    // Compute agency for this port (monotonic transform of concentration)
    const agency = concentration / (1 + concentration);

    // Update stats
    const stats = this.portStats.get(portIdx)!;
    stats.samples++;
    if (bound) stats.bindings++;

    // Agency-weighted binding: only count bindings when model is confident
    // Low agency early bindings (lucky wide cones) don't signal understanding
    stats.weightedBindings += bound ? agency : 0;

    // Increment global sample counter
    this.totalSamples++;

    // Track recent trajectories, errors, and agencies (rolling window)
    const windowSize = this.config.proliferation?.windowSize ?? 50;
    stats.recentTrajectories.push([...actualDelta]);
    stats.recentErrors.push(predictionError);
    stats.recentAgencies.push(agency);

    if (stats.recentTrajectories.length > windowSize) {
      stats.recentTrajectories.shift();
      stats.recentErrors.shift();
      stats.recentAgencies.shift();
    }

    // Train causal network with binding-gated learning
    // Soft conditional NLL creates the homeostatic feedback loop:
    //   narrow cones → low binding → uncertainty penalty → variance increases → wider cones
    await this.causalNet.observe(beforeEmb, portEmb, actualDelta, bindingStrength);

    // Adaptive port embeddings: gradient-based update to maximize reward
    this.updatePortEmbedding(portIdx, beforeEmb, bindingStrength, normalizedDistance, predictionError);

    // Proliferate if needed
    if (this.shouldProliferate(portIdx)) {
      this.proliferate(beforeEmb, portEmb, portIdx);
    }
  }

  /**
   * Flush all buffered training data.
   * Also runs port pruning to remove underused ports.
   */
  async flush(): Promise<void> {
    await this.causalNet.flush();

    // Run usage-based pruning: natural selection removes underperforming ports
    this.pruneUnderperformingPorts();
  }

  /**
   * Get number of ports.
   */
  getPortCount(): number {
    return this.portEmbeddings.length;
  }

  /**
   * Export port embeddings for composition.
   */
  exportPorts(): Vec[] {
    return this.portEmbeddings.map(p => [...p]);
  }

  /**
   * Import port embeddings from another actor.
   */
  importPorts(ports: Vec[]): void {
    for (const port of ports) {
      const idx = this.portEmbeddings.length;
      this.portEmbeddings.push([...port]);
      this.portStats.set(idx, {
        samples: 0,
        bindings: 0,
        weightedBindings: 0,
        recentTrajectories: [],
        recentErrors: [],
        recentAgencies: [],
        lastProliferationSample: 0,
        creationSample: this.totalSamples,
      });
    }
  }

  /**
   * Get commitment concentrations for all ports at a given state.
   * Useful for latent space visualization.
   */
  getPortCommitments(state: S): Array<{
    portIdx: number;
    embedding: Vec;
    concentration: number;
    agency: number;
    coneRadius: Vec;
    selected: boolean;
  }> {
    const stateEmb = this.embedState(state);
    const selectedPort = this.selectPort(stateEmb);
    const results = [];

    for (let i = 0; i < this.portEmbeddings.length; i++) {
      const portEmb = this.portEmbeddings[i]!;
      const {variance} = this.causalNet.predict(stateEmb, portEmb);
      const concentration = this.concentrationFromVariance(variance);
      const agency = concentration / (1 + concentration);

      // Compute anisotropic cone radius (same as buildCone)
      const scaleFactor = 1.0;
      const coneRadius = variance.map(v => scaleFactor * Math.sqrt(v));

      results.push({
        portIdx: i,
        embedding: [...portEmb],
        concentration,
        agency,
        coneRadius,
        selected: i === selectedPort,
      });
    }

    return results;
  }

  /**
   * Get detailed analytics for all ports.
   * Returns per-port statistics for logging and analysis.
   */
  getAnalytics(): {
    portCount: number;
    ports: Array<{
      portIdx: number;
      samples: number;
      bindings: number;
      bindingRate: number;
      avgAgency: number;
      agencyStdDev: number;
      avgError: number;
      errorStdDev: number;
    }>;
  } {
    const ports = [];

    for (let i = 0; i < this.portEmbeddings.length; i++) {
      const stats = this.portStats.get(i);
      if (!stats) continue;

      const bindingRate = stats.samples > 0 ? stats.bindings / stats.samples : 0;

      // Compute average and std dev of recent agencies
      let avgAgency = 0;
      let agencyStdDev = 0;

      if (stats.recentAgencies.length > 0) {
        avgAgency = stats.recentAgencies.reduce((a, b) => a + b, 0) / stats.recentAgencies.length;

        const agencyVariance =
          stats.recentAgencies.reduce((sum, a) => sum + (a - avgAgency) ** 2, 0) /
          stats.recentAgencies.length;
        agencyStdDev = Math.sqrt(agencyVariance);
      }

      // Compute average and std dev of recent errors
      let avgError = 0;
      let errorStdDev = 0;

      if (stats.recentErrors.length > 0) {
        avgError = stats.recentErrors.reduce((a, b) => a + b, 0) / stats.recentErrors.length;

        const variance =
          stats.recentErrors.reduce((sum, e) => sum + (e - avgError) ** 2, 0) /
          stats.recentErrors.length;
        errorStdDev = Math.sqrt(variance);
      }

      ports.push({
        portIdx: i,
        samples: stats.samples,
        bindings: stats.bindings,
        bindingRate,
        avgAgency,
        agencyStdDev,
        avgError,
        errorStdDev,
      });
    }

    return {
      portCount: this.portEmbeddings.length,
      ports,
    };
  }

  /**
   * Clean up resources.
   */
  dispose(): void {
    this.causalNet.dispose();
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  /**
   * Select a port for the given state using the configured policy.
   * Returns port index, or -1 if no ports exist.
   */
  private selectPort(stateEmb: Vec): number {
    if (this.portEmbeddings.length === 0) {
      return -1;
    }

    // Gather context for policy
    const concentrations: number[] = [];
    const portSamples: number[] = [];
    const bindingRates: number[] = [];

    for (let i = 0; i < this.portEmbeddings.length; i++) {
      const portEmb = this.portEmbeddings[i]!;
      const {variance} = this.causalNet.predict(stateEmb, portEmb);
      const concentration = this.concentrationFromVariance(variance);
      const stats = this.portStats.get(i);
      const samples = stats?.samples ?? 0;
      const bindings = stats?.bindings ?? 0;
      const bindingRate = samples > 0 ? bindings / samples : 1.0; // Default 1.0 for new ports

      concentrations.push(concentration);
      portSamples.push(samples);
      bindingRates.push(bindingRate);
    }

    // Delegate to policy
    return this.portSelectionPolicy.selectPort({
      stateEmb,
      portEmbs: this.portEmbeddings,
      concentrations,
      portSamples,
      bindingRates,
    });
  }

  /**
   * Convert prediction variance to concentration geometrically.
   *
   * The dual-head CausalNet outputs variance per dimension.
   * We convert this to concentration (inverse of cone size):
   *
   * 1. Cone radius = k * sqrt(avg variance)
   *    - k = 2 for 95% confidence (±2σ)
   * 2. Concentration ≈ d / radius²
   *    - Higher variance → larger radius → lower concentration
   *    - Lower variance → smaller radius → higher concentration
   *
   * This creates a direct geometric link between prediction uncertainty
   * and affordance cone geometry.
   */
  private concentrationFromVariance(variance: Vec): number {
    // Compute average variance across dimensions
    const avgVariance = variance.reduce((a, b) => a + b, 0) / variance.length;

    // Cone radius covers ±2σ for 95% confidence
    const k = 2;
    const coneRadius = k * Math.sqrt(avgVariance);

    // Convert radius to concentration: concentration ≈ d / radius²
    const embeddingDim = variance.length;
    const concentration = embeddingDim / (coneRadius * coneRadius + 1e-6);

    return concentration;
  }

  /**
   * Create a new port embedding.
   * Returns the new embedding and updates internal state.
   */
  private createPort(stateEmb: Vec): Vec {
    // Initialize port embedding near state or randomly
    const portEmb = this.initializePortEmbedding(stateEmb);
    const idx = this.portEmbeddings.length;

    this.portEmbeddings.push(portEmb);
    this.portStats.set(idx, {
      samples: 0,
      bindings: 0,
      weightedBindings: 0,
      recentTrajectories: [],
      recentErrors: [],
      recentAgencies: [],
      lastProliferationSample: 0,
      creationSample: this.totalSamples,
    });

    return portEmb;
  }

  /**
   * Initialize a new port embedding.
   * Ports live in the same manifold as state embeddings.
   * Start near the state embedding with small random perturbation.
   */
  private initializePortEmbedding(stateEmb: Vec): Vec {
    // Ports have same dimensionality as state embeddings
    return stateEmb.map(val => val + (Math.random() - 0.5) * 0.1);
  }

  /**
   * Build affordance cone from prediction.
   *
   * Anisotropic cone construction:
   * - Radius per dimension is proportional to sqrt(variance[i])
   * - High variance dimension → wide cone (model uncertain)
   * - Low variance dimension → narrow cone (model confident)
   * - Context-dependent: same port can have different anisotropy for different states
   *
   * This allows the cone to naturally reflect which dimensions the model
   * is uncertain about for this specific prediction, without requiring
   * global dimensional importance weights on ports.
   */
  private buildCone(delta: Vec, variance: Vec, concentration: number, stateEmb: Vec): Cone {
    // Anisotropic radius: proportional to standard deviation per dimension
    // Scale factor ensures reasonable cone sizes (tunable per domain)
    const scaleFactor = 1.0;
    const radius = variance.map(v => scaleFactor * Math.sqrt(v));

    return {
      center: delta,
      radius,
    };
  }

  /**
   * Find nearby ports in embedding space.
   * Used by learned policies to detect and penalize overlap.
   *
   * Returns distances in embedding space, not concentrations, because:
   * - Concentrations are state-dependent and don't reflect true overlap
   * - Ports close in embedding space respond to similar states (true overlap)
   */
  private findNearbyPorts(portEmb: Vec, currentPortIdx: number, k: number): Array<{ portIdx: number; distance: number; embedding: Vec }> {
    if (this.portEmbeddings.length <= 1) return [];

    // Compute distances to all other ports in embedding space
    const distances: Array<{ portIdx: number; distance: number; embedding: Vec }> = [];

    for (let portIdx = 0; portIdx < this.portEmbeddings.length; portIdx++) {
      if (portIdx === currentPortIdx) continue;

      const otherPortEmb = this.portEmbeddings[portIdx]!;

      // Euclidean distance in embedding space
      let sumSq = 0;
      for (let i = 0; i < portEmb.length; i++) {
        const diff = portEmb[i]! - otherPortEmb[i]!;
        sumSq += diff * diff;
      }
      const distance = Math.sqrt(sumSq);

      distances.push({ portIdx, distance, embedding: otherPortEmb });
    }

    // Sort by distance and take k nearest
    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, k);
  }

  /**
   * Update port embedding via gradient ascent on multi-objective reward.
   * This is the principled approach: port embeddings optimize the same objectives as refinement.
   *
   * Objectives:
   * - Binding success: Move toward states that bind successfully
   * - Binding failure: Move away from states that fail to bind
   * - Error minimization: Move toward states with low prediction error
   *
   * Learning rate decays with maturity for stability (learned predictions become valid).
   */
  private updatePortEmbedding(
    portIdx: number,
    stateEmb: Vec,
    bindingStrength: number,
    normalizedDistance: number,
    predictionError: number
  ): void {
    const stats = this.portStats.get(portIdx);
    if (!stats) return;

    // Decay learning rate with maturity
    const samples = stats.samples;
    const maturityThreshold = 200;
    const youthThreshold = 50;

    let learningRate = 0.01; // Base learning rate
    if (samples >= maturityThreshold) {
      learningRate *= 0.1; // Very small updates for mature ports
    } else if (samples > youthThreshold) {
      const maturityFactor = 1.0 - (samples - youthThreshold) / (maturityThreshold - youthThreshold);
      learningRate *= maturityFactor;
    }

    const portEmb = this.portEmbeddings[portIdx]!;

    // Compute gradient direction based on binding outcome
    const bound = normalizedDistance < 1.0;

    if (bound) {
      // SUCCESS: Pull port embedding toward this state
      // Strength proportional to binding strength (how well inside cone)
      const attractionStrength = bindingStrength * (1.0 - predictionError); // Scaled by prediction quality

      for (let i = 0; i < portEmb.length; i++) {
        const gradient = stateEmb[i]! - portEmb[i]!; // Direction toward state
        portEmb[i] = portEmb[i]! + learningRate * attractionStrength * gradient;
      }
    } else {
      // FAILURE: Push port embedding away from this state
      // This state is outside our territory, let another port handle it
      const repulsionStrength = (1.0 - bindingStrength); // Stronger when clearly outside

      for (let i = 0; i < portEmb.length; i++) {
        const gradient = portEmb[i]! - stateEmb[i]!; // Direction away from state
        portEmb[i] = portEmb[i]! + learningRate * repulsionStrength * gradient * 0.5; // Weaker repulsion than attraction
      }
    }
  }

  /**
   * Evaluate binding with soft boundaries (differentiable pressure signal).
   *
   * Uses sigmoid to create smooth transition:
   * - Deep inside cone: strength → 1.0 (strong concentrate signal)
   * - At boundary: strength → 0.5 (neutral)
   * - Far outside: strength → 0.0 (strong disperse signal)
   *
   * This allows learning from "near misses" and "near successes" without
   * requiring formal binding failures.
   */
  private evaluateSoftBinding(
    actualDelta: Vec,
    cone: Cone
  ): {
    bound: boolean;
    strength: number;
    normalizedDistance: number;
  } {
    // Compute normalized distance in cone space
    let sumSq = 0;
    for (let i = 0; i < actualDelta.length; i++) {
      const residual = actualDelta[i]! - cone.center[i]!;
      const normalized = cone.radius[i]! > 0 ? residual / cone.radius[i]! : residual;
      sumSq += normalized * normalized;
    }

    const normalizedDistance = Math.sqrt(sumSq);

    // Binary binding for stats (threshold at 1.0)
    const bound = normalizedDistance < 1.0;

    // Soft binding strength using sigmoid
    // sharpness controls transition steepness (higher = sharper boundary)
    const sharpness = 4.0;
    const strength = 1.0 / (1.0 + Math.exp(sharpness * (normalizedDistance - 1.0)));

    return { bound, strength, normalizedDistance };
  }

  /**
   * Compute violation magnitude for failed binding.
   */
  private computeViolation(actualDelta: Vec, cone: Cone): number {
    let sumSq = 0;
    for (let i = 0; i < actualDelta.length; i++) {
      const residual = actualDelta[i]! - cone.center[i]!;
      const normalized = cone.radius[i]! > 0 ? residual / cone.radius[i]! : residual;
      sumSq += normalized * normalized;
    }

    const normalizedDistance = Math.sqrt(sumSq);

    // Violation is how far outside the cone (0 if inside)
    return Math.max(0, normalizedDistance - 1.0);
  }

  /**
   * Simple k-means clustering for trajectory separation detection.
   * Returns cluster assignments and centroids.
   */
  private clusterTrajectories(
    trajectories: Vec[],
    k: number = 2
  ): { labels: number[]; centroids: Vec[]; separation: number } | null {
    if (trajectories.length < k * 5) return null; // Need enough samples per cluster

    const dim = trajectories[0]?.length ?? 0;
    if (dim === 0) return null;

    // Initialize centroids randomly
    const centroids: Vec[] = [];
    const used = new Set<number>();
    for (let i = 0; i < k; i++) {
      let idx: number;
      do {
        idx = Math.floor(Math.random() * trajectories.length);
      } while (used.has(idx));
      used.add(idx);
      centroids.push([...trajectories[idx]!]);
    }

    // K-means iterations
    const labels = Array(trajectories.length).fill(0);
    const maxIters = 20;

    for (let iter = 0; iter < maxIters; iter++) {
      // Assign to nearest centroid
      let changed = false;
      for (let i = 0; i < trajectories.length; i++) {
        const traj = trajectories[i]!;
        let minDist = Infinity;
        let bestCluster = 0;

        for (let c = 0; c < k; c++) {
          const dist = this.euclideanDistance(traj, centroids[c]!);
          if (dist < minDist) {
            minDist = dist;
            bestCluster = c;
          }
        }

        if (labels[i] !== bestCluster) {
          labels[i] = bestCluster;
          changed = true;
        }
      }

      if (!changed) break;

      // Update centroids
      const counts = Array(k).fill(0);
      const sums: Vec[] = Array(k).fill(0).map(() => Array(dim).fill(0));

      for (let i = 0; i < trajectories.length; i++) {
        const c = labels[i]!;
        counts[c]++;
        for (let d = 0; d < dim; d++) {
          sums[c]![d] += trajectories[i]![d]!;
        }
      }

      for (let c = 0; c < k; c++) {
        if (counts[c]! > 0) {
          for (let d = 0; d < dim; d++) {
            centroids[c]![d] = sums[c]![d]! / counts[c]!;
          }
        }
      }
    }

    // Compute separation: distance between centroids
    const separation = this.euclideanDistance(centroids[0]!, centroids[1]!);

    return { labels, centroids, separation };
  }

  private euclideanDistance(a: Vec, b: Vec): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      const diff = a[i]! - b[i]!;
      sum += diff * diff;
    }
    return Math.sqrt(sum);
  }

  /**
   * Check if proliferation is needed for this port.
   *
   * Error reduction test:
   *
   * Ports map situations (states) to affordances (predictions). A port should split
   * when it covers situations that need fundamentally different mappings - i.e.,
   * situations the model cannot reconcile with a single smooth function.
   *
   * Test procedure:
   * 1. Wait for port stability (agency + binding convergence)
   * 2. Cluster recent trajectories/situations into 2 groups
   * 3. Check if model has systematically different errors across clusters
   * 4. If yes → model biased toward one regime, can't reconcile → split
   * 5. If no → model handles all situations equally → single port suffices
   *
   * Uses F-statistic on prediction errors (between-cluster vs within-cluster variance).
   * All thresholds derived from statistical principles, not domain-specific tuning.
   */
  private shouldProliferate(portIdx: number): boolean {
    if (!this.config.proliferation?.enabled) {
      return false;
    }

    const stats = this.portStats.get(portIdx);
    if (!stats) return false;

    const windowSize = this.config.proliferation?.windowSize ?? 50;

    // Need sufficient trajectory data for meaningful analysis
    if (stats.recentTrajectories.length < windowSize) {
      return false;
    }

    // === STABILITY REQUIREMENT ===
    // Port must prove it understands its domain before claiming failure = new regime

    // 1. Agency stability: Has confidence converged?
    const recentAgencies = stats.recentAgencies;
    if (recentAgencies.length < windowSize / 2) {
      return false; // Need enough history
    }
    const agencyMean = recentAgencies.reduce((a, b) => a + b, 0) / recentAgencies.length;
    const agencyStdDev = Math.sqrt(
      recentAgencies.reduce((sum, a) => sum + (a - agencyMean) ** 2, 0) / recentAgencies.length
    );
    const agencyCoV = agencyMean > 0 ? agencyStdDev / agencyMean : Infinity;

    if (agencyCoV > 0.15) {
      if (stats.samples % 200 === 0) {
        console.log(`[Port ${portIdx}] Proliferation blocked: Agency unstable (CoV=${agencyCoV.toFixed(3)} > 0.15)`);
      }
      return false; // Agency still fluctuating, domain not understood
    }

    // 2. Binding convergence: Has agency-weighted binding rate plateaued?
    // Compute effective binding rate over sliding windows to detect slope
    // Effective binding = agency-weighted, so early low-agency bindings don't count
    const bindingWindowSize = Math.floor(windowSize / 5); // 5 windows
    const effectiveBindingRates: number[] = [];

    // We need to track cumulative weighted bindings over windows
    // Use recent agencies as a proxy for weighting since we have per-sample agency
    for (let i = 0; i < recentAgencies.length - bindingWindowSize; i += bindingWindowSize) {
      let weightedBindingSum = 0;
      let totalSamples = 0;

      for (let j = i; j < i + bindingWindowSize && j < recentAgencies.length; j++) {
        const agency = recentAgencies[j]!;
        const error = stats.recentErrors[j]!;
        // Approximate: bound if error low, weighted by agency
        const bound = error < 0.1; // Still approximate, but now agency-weighted
        weightedBindingSum += bound ? agency : 0;
        totalSamples++;
      }

      // Effective binding rate: sum of (agency when bound) / samples
      effectiveBindingRates.push(totalSamples > 0 ? weightedBindingSum / totalSamples : 0);
    }

    if (effectiveBindingRates.length < 3) {
      return false; // Need at least 3 windows for slope
    }

    // Compute linear regression slope
    const n = effectiveBindingRates.length;
    const xMean = (n - 1) / 2; // 0, 1, 2, ... n-1
    const yMean = effectiveBindingRates.reduce((a, b) => a + b, 0) / n;
    let numerator = 0;
    let denominator = 0;
    for (let i = 0; i < n; i++) {
      numerator += (i - xMean) * (effectiveBindingRates[i]! - yMean);
      denominator += (i - xMean) ** 2;
    }
    const slope = denominator > 0 ? numerator / denominator : 0;

    // Require slope near zero (converged) or negative (degrading)
    // Positive slope = still improving = not ready to proliferate
    if (slope > 0.02) {
      if (stats.samples % 200 === 0) {
        console.log(`[Port ${portIdx}] Proliferation blocked: Agency-weighted binding still improving (slope=${slope.toFixed(4)} > 0.02)`);
      }
      return false; // Still learning, performance improving
    }

    if (stats.samples % 200 === 0) {
      console.log(`[Port ${portIdx}] Stability check passed: Agency CoV=${agencyCoV.toFixed(3)}, Weighted binding slope=${slope.toFixed(4)}`);
    }

    // === CLUSTERING FOR ERROR ANALYSIS ===
    //
    // Cluster trajectories to test if different situation clusters have different errors.
    // K-means is used here just to partition data, not to measure spatial separation.

    const trajectories = stats.recentTrajectories;
    const numTrajectories = trajectories.length;

    // Run k-means with k=2 to partition trajectories
    const clustering = this.clusterTrajectories(trajectories, 2);
    if (!clustering) {
      return false; // Not enough data for clustering
    }
    const { labels } = clustering;

    // Compute cluster assignments
    const cluster0: number[] = [];
    const cluster1: number[] = [];
    for (let i = 0; i < numTrajectories; i++) {
      if (labels[i] === 0) cluster0.push(i);
      else cluster1.push(i);
    }

    // Check balance: both clusters need sufficient size
    const minClusterSize = Math.sqrt(numTrajectories / 2);
    const isBalanced = cluster0.length >= minClusterSize && cluster1.length >= minClusterSize;

    if (!isBalanced) {
      if (stats.samples % 200 === 0) {
        console.log(`[Port ${portIdx}] Cluster quality check: K-means split ${cluster0.length}/${cluster1.length} (imbalanced, skipping)`);
      }
      return false;
    }

    // === ERROR REDUCTION TEST ===
    //
    // The real question: "Does this port cover situations that need different mappings?"
    //
    // Test: Does the model have systematically different errors across clusters?
    // - If cluster 0 errors << cluster 1 errors (or vice versa)
    //   → Model biased toward one regime, failing to reconcile the other
    //   → Specialized ports would help
    // - If errors are uniform across clusters
    //   → Model handles all situations equally (smooth mapping)
    //   → No need to split
    //
    // We use recent prediction errors already tracked per sample.

    if (stats.recentErrors.length !== numTrajectories) {
      // Mismatch between trajectories and errors - shouldn't happen but be safe
      return false;
    }

    // Map clusters to their errors
    const cluster0Errors: number[] = [];
    const cluster1Errors: number[] = [];
    for (let i = 0; i < numTrajectories; i++) {
      const error = stats.recentErrors[i]!;
      if (labels[i] === 0) {
        cluster0Errors.push(error);
      } else {
        cluster1Errors.push(error);
      }
    }

    // Compute average error per cluster
    const avgError0 = cluster0Errors.reduce((sum, e) => sum + e, 0) / cluster0Errors.length;
    const avgError1 = cluster1Errors.reduce((sum, e) => sum + e, 0) / cluster1Errors.length;
    const overallAvgError = stats.recentErrors.reduce((sum, e) => sum + e, 0) / stats.recentErrors.length;

    // Compute error variance within each cluster
    const variance0 = cluster0Errors.reduce((sum, e) => sum + (e - avgError0) ** 2, 0) / cluster0Errors.length;
    const variance1 = cluster1Errors.reduce((sum, e) => sum + (e - avgError1) ** 2, 0) / cluster1Errors.length;
    const avgWithinClusterVariance = (variance0 + variance1) / 2;

    // Compute error variance between clusters (how different are cluster means?)
    const betweenClusterVariance =
      (cluster0Errors.length * (avgError0 - overallAvgError) ** 2 +
       cluster1Errors.length * (avgError1 - overallAvgError) ** 2) / numTrajectories;

    // F-statistic: between-cluster variance / within-cluster variance
    // High F: errors differ systematically between clusters → model can't reconcile
    // Low F: errors uniform → model handles both situations equally
    const errorFStat = avgWithinClusterVariance > 0 ? betweenClusterVariance / avgWithinClusterVariance : 0;

    // Also check absolute error difference (for cases where variance is low but means differ)
    const errorDifference = Math.abs(avgError0 - avgError1);
    const relativeErrorDifference = overallAvgError > 0 ? errorDifference / overallAvgError : 0;

    // Split only if BOTH conditions met:
    // 1. F-stat > 1.0: Error variance between clusters exceeds within-cluster variance
    //    (systematic pattern, not random noise)
    // 2. Relative error difference > 0.5: One cluster has 50%+ higher error than the other
    //    (meaningful practical difference)
    //
    // Requiring BOTH prevents splitting during early learning when errors are high everywhere
    // but forces split when model has converged yet shows systematic bias toward one regime.
    const hasSystematicErrorDifference = errorFStat > 1.0 && relativeErrorDifference > 0.5;

    // Debug logging
    if (stats.samples % 200 === 0 || hasSystematicErrorDifference) {
      console.log(`[Port ${portIdx}] Error reduction test (samples: ${stats.samples}):`);
      console.log(`  Stability: Agency CoV=${agencyCoV.toFixed(3)}, Binding slope=${slope.toFixed(4)}`);
      console.log(`  K-means split: ${cluster0.length}/${cluster1.length} trajectories`);
      console.log(`  Cluster 0 error: ${avgError0.toFixed(4)}, Cluster 1 error: ${avgError1.toFixed(4)}`);
      console.log(`  Overall error: ${overallAvgError.toFixed(4)}`);
      console.log(`  Error F-stat: ${errorFStat.toFixed(2)} (between/within variance ratio)`);
      console.log(`  Relative error diff: ${(relativeErrorDifference * 100).toFixed(1)}% (|cluster0 - cluster1| / overall)`);
      console.log(`  → Should split: ${hasSystematicErrorDifference} (need BOTH F>1.0 AND relDiff>50%)`);
    }

    return hasSystematicErrorDifference;
  }

  /**
   * Proliferate by splitting along discovered trajectory clusters.
   *
   * Uses clustering to find natural boundaries in the trajectory distribution,
   * then creates specialist ports for each cluster. This "atrophies" the unused
   * middle region between clusters.
   */
  private proliferate(stateEmb: Vec, parentPortEmb: Vec, parentIdx: number): void {
    const parentStats = this.portStats.get(parentIdx);
    if (!parentStats) return;

    // Re-cluster to get centroid directions
    const clustering = this.clusterTrajectories(parentStats.recentTrajectories, 2);
    if (!clustering) {
      console.log(`[Proliferation] Clustering failed for port ${parentIdx}, skipping`);
      return;
    }

    const { centroids } = clustering;

    // Create two new ports by shifting parent embedding toward each cluster centroid
    // This biases the ports toward their respective regions in trajectory space
    const shiftScale = 0.3; // How much to shift from parent toward cluster

    for (let c = 0; c < 2; c++) {
      const centroid = centroids[c]!;

      // Shift parent embedding in direction of this cluster's centroid
      // Note: centroid is in trajectory space (delta), we're shifting in state embedding space
      // Simple heuristic: use centroid direction as embedding shift
      const newPortEmb = parentPortEmb.map((val, i) => {
        const shift = (i < centroid.length) ? centroid[i]! * shiftScale : 0;
        return val + shift;
      });

      const idx = this.portEmbeddings.length;
      this.portEmbeddings.push(newPortEmb);
      this.portStats.set(idx, {
        samples: 0,
        bindings: 0,
        weightedBindings: 0,
        recentTrajectories: [],
        recentErrors: [],
        recentAgencies: [],
        lastProliferationSample: 0,
        creationSample: this.totalSamples,
      });

      console.log(`[Proliferation] Created port ${idx} from parent ${parentIdx}, cluster ${c}`);
    }

    // Retire the parent port by clearing all its stats and marking it as proliferated
    // This prevents it from continuing to see mixed data from both child regions
    // The parent should gracefully fade out as children take over
    parentStats.lastProliferationSample = parentStats.samples;
    parentStats.recentTrajectories = [];
    parentStats.recentErrors = [];
    parentStats.recentAgencies = [];
    // Reset binding stats so parent doesn't have unfair advantage over children
    parentStats.bindings = 0;
    parentStats.samples = 0;

    console.log(`[Proliferation] Split complete: ${this.portEmbeddings.length} total ports (parent ${parentIdx} reset)`);
  }

  /**
   * Prune ports that are consistently underused.
   *
   * Natural selection: ports compete for selection. Mature ports that consistently
   * receive less than half the fair share of selections are removed.
   *
   * All criteria derived from system state - no magic thresholds.
   */
  private pruneUnderperformingPorts(): void {
    if (this.portEmbeddings.length <= 1) {
      return; // Don't prune if only one port
    }

    const numPorts = this.portEmbeddings.length;
    const avgSelectionRate = 1.0 / numPorts; // Fair share
    const windowSize = this.config.proliferation?.windowSize ?? 50;

    const portsToRemove: number[] = [];

    for (const [portIdx, stats] of this.portStats) {
      const portAge = this.totalSamples - stats.creationSample;
      const mature = portAge > windowSize * 2;

      if (!mature) continue; // Too young to judge

      // Calculate selection rate over port's lifetime, not total samples
      // A port created at sample 700 shouldn't be penalized for not existing before sample 700
      const portLifetime = this.totalSamples - stats.creationSample;
      const selectionRate = portLifetime > 0 ? stats.samples / portLifetime : 0;
      const underused = selectionRate < avgSelectionRate * 0.5;

      if (underused) {
        portsToRemove.push(portIdx);
        console.log(`[Pruning] Port ${portIdx} underused: ${(selectionRate * 100).toFixed(1)}% vs fair share ${(avgSelectionRate * 100).toFixed(1)}% (age: ${portAge} samples)`);
      }
    }

    // Remove ports (from highest index to lowest to maintain indices during removal)
    for (const idx of portsToRemove.sort((a, b) => b - a)) {
      this.portEmbeddings.splice(idx, 1);
      this.portStats.delete(idx);

      // Renumber remaining ports
      const newStats = new Map<number, PortStats>();
      for (const [oldIdx, stats] of this.portStats) {
        const newIdx = oldIdx > idx ? oldIdx - 1 : oldIdx;
        newStats.set(newIdx, stats);
      }
      this.portStats = newStats;
    }

    if (portsToRemove.length > 0) {
      console.log(`[Pruning] Removed ${portsToRemove.length} underused ports, ${this.portEmbeddings.length} remaining`);
    }
  }
}
