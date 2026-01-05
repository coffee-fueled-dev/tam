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
  bindings: number;             // Count of successful bindings
  recentTrajectories: Vec[];    // Actual deltas (for clustering)
  recentErrors: number[];       // Prediction MSE (for plateau detection)
  recentAgencies: number[];     // Agency values when this port is selected
  lastProliferationSample: number;  // Sample count at last proliferation
}

export class Actor<S> {
  // Port manifold: just embeddings, no objects
  private portEmbeddings: Vec[] = [];
  private portStats = new Map<number, PortStats>(); // index → stats

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
          recentTrajectories: [],
          recentErrors: [],
          recentAgencies: [],
          lastProliferationSample: 0,
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
   */
  async flush(): Promise<void> {
    await this.causalNet.flush();
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
        recentTrajectories: [],
        recentErrors: [],
        recentAgencies: [],
        lastProliferationSample: 0,
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

      results.push({
        portIdx: i,
        embedding: [...portEmb],
        concentration,
        agency,
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
      recentTrajectories: [],
      recentErrors: [],
      recentAgencies: [],
      lastProliferationSample: 0,
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
   * Check if proliferation is needed for this port.
   *
   * Geometric criterion: Irreconcilable conflict AFTER initial learning
   *
   * The port should proliferate when it cannot accommodate its assigned trajectories
   * without sacrificing all specificity. This is derived geometrically from:
   *
   * 1. Minimum viable concentration (from prediction uncertainty):
   *    - Cone must be wide enough to cover ±2σ prediction error
   *    - Gives minimum concentration below which port is useless
   *
   * 2. Expected failure rate (from error statistics):
   *    - Some failures are expected even with perfect cone sizing (noise)
   *    - Actual failures >> expected failures → structural problem
   *
   * Proliferate if port is at minimum width AND still has excess failures.
   */
  private shouldProliferate(portIdx: number): boolean {
    if (!this.config.proliferation?.enabled) {
      return false;
    }

    const stats = this.portStats.get(portIdx);
    if (!stats) return false;

    const windowSize = this.config.proliferation?.windowSize ?? 50;
    const minSamples = this.config.proliferation?.minSamplesBeforeProliferation ?? 500;

    // Wait for initial causal learning to stabilize
    if (stats.samples < minSamples) {
      return false;
    }

    // Cooldown: prevent rapid successive proliferation
    const samplesSinceProliferation = stats.samples - stats.lastProliferationSample;
    if (samplesSinceProliferation < windowSize * 2) {
      return false;
    }

    // Need sufficient error statistics
    if (stats.recentErrors.length < windowSize) {
      return false;
    }

    // === GEOMETRIC DERIVATION ===

    // 1. Compute minimum viable agency from prediction uncertainty
    const avgError = stats.recentErrors.reduce((a, b) => a + b, 0) / stats.recentErrors.length;
    const errorVariance = stats.recentErrors.reduce((sum, e) => {
      const diff = e - avgError;
      return sum + diff * diff;
    }, 0) / stats.recentErrors.length;
    const stdError = Math.sqrt(errorVariance);

    // Cone radius must cover ±2σ for 95% confidence
    // For anisotropic cones, use average dimension behavior
    const embeddingDim = this.portEmbeddings[0]?.length ?? 2;
    const minConeRadius = 2 * stdError * Math.sqrt(embeddingDim);

    // Convert radius to concentration: radius ≈ sqrt(d / concentration)
    // So: concentration ≈ d / radius²
    const minConcentration = embeddingDim / (minConeRadius * minConeRadius + 1e-6);
    const minAgency = minConcentration / (1 + minConcentration);

    // 2. Compute current agency
    const avgAgency = stats.recentAgencies.reduce((a, b) => a + b, 0) / stats.recentAgencies.length;

    // 3. Compute expected vs actual failure rates
    const bindingRate = stats.samples > 0 ? stats.bindings / stats.samples : 1.0;
    const actualFailureRate = 1 - bindingRate;

    // Expected failures: based on prediction error distribution
    // If errors ~ Normal(0, σ), roughly 5% should exceed 2σ threshold
    const expectedFailureRate = 0.05; // From 95% confidence interval

    // === IRRECONCILABLE CONFLICT ===

    // If prediction error is negligible (minAgency > 95%), no geometric constraint
    // Port can be as narrow as needed, no proliferation
    if (minAgency > 0.95) {
      return false;
    }

    // Port is at minimum viable width (can't widen more without becoming useless)
    // AND failures significantly exceed what we'd expect from noise alone
    const isAtMinimumWidth = avgAgency < minAgency * 1.5; // 50% margin for safety
    const hasExcessFailures = actualFailureRate > expectedFailureRate * 3; // 3x noise level

    const hasConflict = isAtMinimumWidth && hasExcessFailures;

    // Debug logging
    if (stats.samples % 500 === 0 || hasConflict) {
      console.log(`[Port ${portIdx}] Proliferation check:`);
      console.log(`  Samples: ${stats.samples}, Since last: ${samplesSinceProliferation}`);
      console.log(`  Avg error: ${avgError.toFixed(4)}, Std: ${stdError.toFixed(4)}`);
      console.log(`  Min cone radius: ${minConeRadius.toFixed(3)} → min agency: ${(minAgency * 100).toFixed(1)}%`);
      console.log(`  Current agency: ${(avgAgency * 100).toFixed(1)}%`);
      console.log(`  Binding rate: ${(bindingRate * 100).toFixed(1)}% (failures: ${(actualFailureRate * 100).toFixed(1)}%)`);
      console.log(`  Expected failures: ${(expectedFailureRate * 100).toFixed(1)}%`);
      console.log(`  At minimum width: ${isAtMinimumWidth}, Excess failures: ${hasExcessFailures}`);
      console.log(`  → Irreconcilable conflict: ${hasConflict}`);
    }

    return hasConflict;
  }

  /**
   * Proliferate a new specialist from the parent port.
   *
   * Creates a new port embedding by perturbing the parent with small random offset.
   * The new port will naturally drift toward its own territory via gradient-based updates.
   */
  private proliferate(stateEmb: Vec, parentPortEmb: Vec, parentIdx: number): void {
    const parentStats = this.portStats.get(parentIdx);

    // Fixed perturbation scale - simple and principled
    // The new port will find its own territory via gradient updates
    const perturbationScale = 0.5;

    // Create new port by perturbing parent in random direction
    const newPortEmb = parentPortEmb.map(
      val => val + (Math.random() - 0.5) * 2 * perturbationScale
    );

    const idx = this.portEmbeddings.length;
    this.portEmbeddings.push(newPortEmb);
    this.portStats.set(idx, {
      samples: 0,
      bindings: 0,
      recentTrajectories: [],
      recentErrors: [],
      recentAgencies: [],
      lastProliferationSample: 0,
    });

    // Update parent's proliferation timestamp
    if (parentStats) {
      parentStats.lastProliferationSample = parentStats.samples;
    }

    console.log(`[Proliferation] Created port ${idx} from parent ${parentIdx} (${this.portEmbeddings.length} total ports)`);
  }
}
