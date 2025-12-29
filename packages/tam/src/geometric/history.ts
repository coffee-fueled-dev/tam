/**
 * BindingHistory: Tracks binding outcomes for refinement decisions.
 *
 * Used to:
 * - Determine if a situation is familiar (seen many times)
 * - Track success rate for a port in a situation region
 * - Detect bimodal failure patterns (port conflates distinct behaviors)
 * - Store failure trajectories for proliferation
 */

import type { Vec } from "../vec";
import type { BindingHistory, BindingOutcome } from "../types";

interface OutcomeRecord {
  outcome: BindingOutcome;
  timestamp: number;
}

interface PortHistory {
  outcomes: Map<string, OutcomeRecord[]>; // situationKey -> outcomes
  failureTrajectories: Vec[]; // For bimodal detection (legacy)
  // Agency tracking for fiber-based proliferation
  recentAgencies: number[]; // Rolling window of agency values
  totalSamples: number; // Total observations for this port
  lastProliferationSample: number; // Sample count at last proliferation
}

export class DefaultBindingHistory implements BindingHistory {
  private history = new Map<string, PortHistory>();
  private situationCounts = new Map<string, number>();

  private readonly maxOutcomesPerSituation: number;
  private readonly maxFailureTrajectories: number;
  private readonly familiarityThreshold: number;
  private readonly recentWindowMs: number;
  private readonly minFailuresForBimodal: number;
  private readonly bimodalRatioThreshold: number;
  private readonly agencyWindowSize: number; // Rolling window for agency

  constructor(opts?: {
    maxOutcomesPerSituation?: number;
    maxFailureTrajectories?: number;
    familiarityThreshold?: number;
    recentWindowMs?: number;
    minFailuresForBimodal?: number;
    bimodalRatioThreshold?: number;
    agencyWindowSize?: number;
  }) {
    this.maxOutcomesPerSituation = opts?.maxOutcomesPerSituation ?? 100;
    this.maxFailureTrajectories = opts?.maxFailureTrajectories ?? 50;
    this.familiarityThreshold = opts?.familiarityThreshold ?? 10;
    this.recentWindowMs = opts?.recentWindowMs ?? 60000; // 1 minute
    this.minFailuresForBimodal = opts?.minFailuresForBimodal ?? 10; // Conservative default
    this.bimodalRatioThreshold = opts?.bimodalRatioThreshold ?? 0.25;
    this.agencyWindowSize = opts?.agencyWindowSize ?? 50; // Rolling window
  }

  private getPortHistory(portId: string): PortHistory {
    let ph = this.history.get(portId);
    if (!ph) {
      ph = {
        outcomes: new Map(),
        failureTrajectories: [],
        recentAgencies: [],
        totalSamples: 0,
        lastProliferationSample: 0,
      };
      this.history.set(portId, ph);
    }
    return ph;
  }

  record(portId: string, situationKey: string, outcome: BindingOutcome): void {
    const ph = this.getPortHistory(portId);

    // Track situation visits globally
    this.situationCounts.set(
      situationKey,
      (this.situationCounts.get(situationKey) ?? 0) + 1
    );

    // Track outcome for this port/situation
    let outcomes = ph.outcomes.get(situationKey);
    if (!outcomes) {
      outcomes = [];
      ph.outcomes.set(situationKey, outcomes);
    }

    outcomes.push({ outcome, timestamp: Date.now() });

    // Trim to max size
    if (outcomes.length > this.maxOutcomesPerSituation) {
      outcomes.shift();
    }
  }

  /**
   * Record a failure trajectory for bimodal detection.
   */
  recordFailureTrajectory(portId: string, trajectory: Vec): void {
    const ph = this.getPortHistory(portId);
    ph.failureTrajectories.push([...trajectory]);

    if (ph.failureTrajectories.length > this.maxFailureTrajectories) {
      ph.failureTrajectories.shift();
    }
  }

  recentSuccessRate(portId: string, situationKey: string): number {
    const ph = this.history.get(portId);
    if (!ph) return 0;

    const outcomes = ph.outcomes.get(situationKey);
    if (!outcomes || outcomes.length === 0) return 0;

    const now = Date.now();
    const recentOutcomes = outcomes.filter(
      (o) => now - o.timestamp < this.recentWindowMs
    );

    if (recentOutcomes.length === 0) return 0;

    const successes = recentOutcomes.filter((o) => o.outcome.success).length;
    return successes / recentOutcomes.length;
  }

  isFamiliar(situationKey: string): boolean {
    const count = this.situationCounts.get(situationKey) ?? 0;
    return count >= this.familiarityThreshold;
  }

  /**
   * Detect if failure trajectories show bimodal pattern.
   * Uses simple clustering: if variance between clusters > variance within,
   * the failures are bimodal (port conflates distinct behaviors).
   *
   * Thresholds are configurable via constructor options.
   */
  detectBimodal(portId: string): boolean {
    const ph = this.history.get(portId);
    // Configurable minimum samples before checking bimodality
    if (!ph || ph.failureTrajectories.length < this.minFailuresForBimodal) {
      return false;
    }

    const trajectories = ph.failureTrajectories;
    const n = trajectories.length;

    // Compute centroid
    const dim = trajectories[0]!.length;
    const centroid = new Array(dim).fill(0);
    for (const t of trajectories) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += t[i]! / n;
      }
    }

    // Compute total variance
    let totalVar = 0;
    for (const t of trajectories) {
      for (let i = 0; i < dim; i++) {
        totalVar += (t[i]! - centroid[i]) ** 2;
      }
    }
    totalVar /= n;

    // Simple k-means with k=2
    // Initialize: first and last trajectory
    let c1 = [...trajectories[0]!];
    let c2 = [...trajectories[n - 1]!];

    for (let iter = 0; iter < 10; iter++) {
      const cluster1: Vec[] = [];
      const cluster2: Vec[] = [];

      // Assign
      for (const t of trajectories) {
        const d1 = this.distance(t, c1);
        const d2 = this.distance(t, c2);
        if (d1 < d2) {
          cluster1.push(t);
        } else {
          cluster2.push(t);
        }
      }

      // Update centroids
      if (cluster1.length > 0) {
        c1 = this.computeCentroid(cluster1);
      }
      if (cluster2.length > 0) {
        c2 = this.computeCentroid(cluster2);
      }
    }

    // Compute within-cluster variance
    let withinVar = 0;
    let count = 0;
    for (const t of trajectories) {
      const d1 = this.distance(t, c1);
      const d2 = this.distance(t, c2);
      withinVar += Math.min(d1, d2) ** 2;
      count++;
    }
    withinVar /= count;

    // If within-cluster variance is much smaller than total variance,
    // the data is bimodal. Threshold is configurable.
    const ratio = withinVar / (totalVar + 1e-8);
    return ratio < this.bimodalRatioThreshold;
  }

  getFailureTrajectories(portId: string): Vec[] {
    const ph = this.history.get(portId);
    return ph?.failureTrajectories ?? [];
  }

  // ============================================================================
  // Agency Tracking for Fiber-Based Proliferation
  // ============================================================================

  /**
   * Record an agency observation for a port.
   * Maintains a rolling window for averaging.
   */
  recordAgency(portId: string, agency: number): void {
    const ph = this.getPortHistory(portId);
    ph.recentAgencies.push(agency);
    ph.totalSamples++;

    // Trim to window size
    if (ph.recentAgencies.length > this.agencyWindowSize) {
      ph.recentAgencies.shift();
    }
  }

  /**
   * Get average agency for a port from rolling window.
   * Returns 1.0 (high agency) if no samples yet.
   */
  getAverageAgency(portId: string): number {
    const ph = this.history.get(portId);
    if (!ph || ph.recentAgencies.length === 0) {
      return 1.0; // Assume high agency until proven otherwise
    }

    const sum = ph.recentAgencies.reduce((s, a) => s + a, 0);
    return sum / ph.recentAgencies.length;
  }

  /**
   * Get total sample count for a port.
   */
  getSampleCount(portId: string): number {
    const ph = this.history.get(portId);
    return ph?.totalSamples ?? 0;
  }

  /**
   * Record that a proliferation event occurred for a port.
   * Used for cooldown tracking.
   */
  recordProliferation(portId: string): void {
    const ph = this.getPortHistory(portId);
    ph.lastProliferationSample = ph.totalSamples;
  }

  /**
   * Check if a port is within the proliferation cooldown period.
   */
  isInCooldown(portId: string, cooldownPeriod: number): boolean {
    const ph = this.history.get(portId);
    if (!ph) return false;

    const samplesSinceProliferation =
      ph.totalSamples - ph.lastProliferationSample;
    return samplesSinceProliferation < cooldownPeriod;
  }

  private distance(a: Vec, b: Vec): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += (a[i]! - b[i]!) ** 2;
    }
    return Math.sqrt(sum);
  }

  private computeCentroid(vecs: Vec[]): Vec {
    if (vecs.length === 0) return [];
    const dim = vecs[0]!.length;
    const centroid = new Array(dim).fill(0);
    for (const v of vecs) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += v[i]! / vecs.length;
      }
    }
    return centroid;
  }

  /**
   * Clear all history (useful for testing).
   */
  clear(): void {
    this.history.clear();
    this.situationCounts.clear();
  }

  /**
   * Get snapshot for diagnostics.
   */
  snapshot(): unknown {
    const portStats: Record<string, unknown> = {};
    for (const [portId, ph] of this.history.entries()) {
      portStats[portId] = {
        situationCount: ph.outcomes.size,
        failureTrajectories: ph.failureTrajectories.length,
        totalSamples: ph.totalSamples,
        recentAgencyWindow: ph.recentAgencies.length,
        averageAgency: this.getAverageAgency(portId),
      };
    }
    return {
      totalSituations: this.situationCounts.size,
      ports: portStats,
    };
  }
}
