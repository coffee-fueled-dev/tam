/**
 * GeometricPortBank: Manages multiple geometric ports with shared networks.
 *
 * Key features:
 * - Single shared CausalNet and CommitmentNet for all ports
 * - Ports are just embedding vectors in the port manifold
 * - Multiple ports can share the same action (specialists)
 * - Port selection based on agency for the current situation
 * - Supports proliferation (creating new specialist ports)
 * - Dynamic reference volume calibration
 *
 * Geometric interpretation:
 * - Each port is a "window" in port space
 * - The situation determines the viewing angle and distance
 * - Agency-based selection picks the port with the narrowest applicable cone
 */

import type {
  Port,
  PortBank,
  Prediction,
  Situation,
  Transition,
  Encoders,
  GeometricPortConfig,
  GeometricPortConfigInput,
  RefinementAction,
  BindingOutcome,
} from "../types";
import { defaultGeometricPortConfig } from "../types";
import type { Vec } from "../vec";
import { magnitude } from "../vec";
import { CausalNet } from "./causal";
import { CommitmentNet } from "./commitment";
import { DefaultBindingHistory } from "./history";
import type { RefinementPolicy } from "../types";
import { FixedRefinementPolicy } from "./refinement";
import { GeometricPort } from "./port";
import { evaluateBinding } from "./fibration";
import { generateSpecialistEmbedding } from "./port-functor";
import type { PortSelectionStrategy } from "./selection";
import { MaxAgencySelectionStrategy } from "./selection";

export class GeometricPortBank<S, C = unknown> implements PortBank<S, C> {
  // All ports by unique ID
  private allPorts = new Map<string, GeometricPort<S, C>>();

  // Ports grouped by base action (for selection)
  private portsByAction = new Map<string, GeometricPort<S, C>[]>();

  private readonly enc: Encoders<S, C>;
  private readonly cfg: GeometricPortConfig;

  // Shared networks
  private readonly causalNet: CausalNet;
  private readonly commitmentNet: CommitmentNet;

  // Shared history and policy
  private readonly history: DefaultBindingHistory;
  private readonly policy: FixedRefinementPolicy;

  // Port selection strategy
  private readonly selectionStrategy: PortSelectionStrategy<S, C>;

  // Dynamic reference volume (calibrated from observed deltas)
  private dynamicReferenceVolume: number;
  private observedDeltaMagnitudes: number[] = [];

  // Proliferation tracking
  private proliferationPending: Array<{
    parentId: string;
  }> = [];

  // Track recent failures per port for functor discovery
  private recentFailures = new Map<
    string,
    Array<{ situationEmb: Vec; trajectory: Vec }>
  >();
  private readonly maxFailuresPerPort = 20;

  // Port functor statistics (for diagnostics)
  private portFunctorStats = {
    totalProliferations: 0,
    functorBased: 0,
    randomBased: 0,
  };

  // Track last selected port for encoder training
  private lastSelectedPort: GeometricPort<S, C> | null = null;

  constructor(
    encoders: Encoders<S, C>,
    config?: GeometricPortConfigInput,
    selectionStrategy?: PortSelectionStrategy<S, C>,
    refinementPolicy?: RefinementPolicy
  ) {
    this.enc = encoders;

    // Deep merge nested configs
    const causal = {
      ...defaultGeometricPortConfig.causal,
      ...(config?.causal ?? {}),
    };
    const commitment = {
      ...defaultGeometricPortConfig.commitment,
      ...(config?.commitment ?? {}),
    };

    this.cfg = {
      ...defaultGeometricPortConfig,
      ...(config ?? {}),
      causal,
      commitment,
    };

    // Initialize reference volume (will be calibrated dynamically if enabled)
    this.dynamicReferenceVolume = this.cfg.referenceVolume;

    // Pass fiber config to CausalNet for consistency regularization
    this.causalNet = new CausalNet(this.cfg.causal, {
      fiberConsistencyWeight: this.cfg.fiberConsistencyWeight,
      fiberThreshold: this.cfg.fiberThreshold,
    });
    this.commitmentNet = new CommitmentNet(this.cfg.commitment);
    this.history = new DefaultBindingHistory({
      familiarityThreshold: this.cfg.familiarityThreshold,
      minFailuresForBimodal: this.cfg.minFailuresForBimodal,
      bimodalRatioThreshold: this.cfg.bimodalRatioThreshold,
      bindingRateDecay: this.cfg.bindingRateDecay,
    });
    this.policy = refinementPolicy ?? new FixedRefinementPolicy(this.cfg);

    // Initialize selection strategy (defaults to max-agency)
    this.selectionStrategy = selectionStrategy ?? new MaxAgencySelectionStrategy({
      minAlignmentThreshold: this.cfg.minAlignmentThreshold,
    });
  }

  /**
   * Get or create the first port for an action.
   * @deprecated Use selectPort() for agency-based selection
   */
  get(action: string): GeometricPort<S, C> {
    const ports = this.portsByAction.get(action);
    if (ports && ports.length > 0) {
      return ports[0]!;
    }
    return this.createPort(action);
  }

  /**
   * Create a new port for an action.
   */
  private createPort(action: string, embedding?: Vec, aperture?: number): GeometricPort<S, C> {
    const port = new GeometricPort({
      action,
      embedding,
      aperture: aperture ?? this.cfg.defaultAperture,
      encoders: this.enc,
      causalNet: this.causalNet,
      commitmentNet: this.commitmentNet,
      history: this.history,
      policy: this.policy,
      config: this.cfg,
      referenceVolume: this.dynamicReferenceVolume,
    });

    // Register in both maps
    this.allPorts.set(port.id, port);
    if (!this.portsByAction.has(action)) {
      this.portsByAction.set(action, []);
    }
    this.portsByAction.get(action)!.push(port);

    return port;
  }

  /**
   * Select the best port for an action given a state embedding.
   * Delegates to the configured selection strategy.
   *
   * Returns null if no port is applicable (all have empty cones).
   */
  selectPort(action: string, stateEmb: Vec): GeometricPort<S, C> | null {
    const candidates = this.portsByAction.get(action);

    if (!candidates || candidates.length === 0) {
      // No ports exist for this action - create one
      return this.createPort(action);
    }

    // Delegate to selection strategy
    const result = this.selectionStrategy.select({
      action,
      stateEmb,
      candidates,
      history: this.history,
    });

    return result.port;
  }

  /**
   * Get current reference volume (for agency computation).
   */
  getReferenceVolume(): number {
    return this.dynamicReferenceVolume;
  }

  /**
   * Update dynamic reference volume from observed delta.
   */
  private updateReferenceVolume(delta: Vec): void {
    if (!this.cfg.useDynamicReference) return;

    const mag = magnitude(delta);
    this.observedDeltaMagnitudes.push(mag);

    // Use max observed magnitude as reference (defines trajectory space extent)
    // Add small buffer to prevent exactly-at-boundary issues
    this.dynamicReferenceVolume = Math.max(
      this.cfg.referenceVolume, // Minimum floor
      Math.max(...this.observedDeltaMagnitudes) * 1.1
    );

    // Update all ports with new reference volume
    for (const port of this.allPorts.values()) {
      port.setReferenceVolume(this.dynamicReferenceVolume);
    }
  }

  /**
   * Observe a transition directly in embedding space (for composition).
   * Bypasses encoders and uses provided embeddings directly.
   *
   * Used by ComposedPort to train target domain with functored embeddings.
   */
  async observeEmbedding(params: {
    beforeEmb: Vec;
    afterEmb: Vec;
    action?: string;
  }): Promise<void> {
    const { beforeEmb, afterEmb, action = "default" } = params;
    const actualDelta = afterEmb.map((a, i) => a - beforeEmb[i]!);
    const situationKey = this.situationToKey(beforeEmb);

    // 1. Select best port for this situation
    let port = this.selectPort(action, beforeEmb);
    if (!port) {
      port = this.createPort(action);
    }
    this.lastSelectedPort = port;

    // 2. Update dynamic reference volume
    this.updateReferenceVolume(actualDelta);

    // 3. Get cone and evaluate binding
    const cone = port.getCone(beforeEmb);
    const outcome = evaluateBinding(actualDelta, cone);

    // 4. Compute and record agency
    const agency = port.computeAgencyFor(beforeEmb);
    this.history.recordAgency(port.id, agency);

    // 5. Record binding outcome
    this.history.record(port.id, situationKey, outcome);

    // 6. Record calibration diagnostics
    const normalizedDistance = this.computeNormalizedDistance(actualDelta, cone);
    this.history.recordNormalizedResidual(port.id, normalizedDistance);
    this.history.recordAgencyBindingPair(port.id, agency, outcome.success);

    if (!outcome.success) {
      this.history.recordFailureTrajectory(port.id, actualDelta);
      this.recordRecentFailure(port.id, beforeEmb, actualDelta);
    }

    // 7. Decide refinement action
    const refinementAction = this.policy.decide(
      outcome,
      port.id,
      situationKey,
      this.history
    );

    // 8. Train networks directly with embeddings
    if (refinementAction === "proliferate") {
      this.scheduleProliferation(port.id);
      this.history.recordProliferation(port.id);
      this.causalNet.observe(beforeEmb, port.embedding, actualDelta);
      (port as any).trainEmbedding(beforeEmb, actualDelta);
    } else {
      // Train both networks using embeddings
      this.causalNet.observe(beforeEmb, port.embedding, actualDelta);
      (port as any).trainEmbedding(beforeEmb, actualDelta);

      // Train CommitmentNet based on refinement action
      if (refinementAction === "narrow") {
        this.commitmentNet.queueRefinement(beforeEmb, port.embedding, "narrow");
      } else if (refinementAction === "widen") {
        const violation = outcome.success ? 0 : outcome.violation;
        this.commitmentNet.queueRefinement(
          beforeEmb,
          port.embedding,
          "widen",
          violation
        );
      }
    }

    // 9. Process pending proliferations
    await this.processProliferations();
  }

  /**
   * Observe a transition with agency-based port selection and potential proliferation.
   */
  async observe(tr: Transition<S, C>): Promise<void> {
    // Encode situation first
    const stateEmb = this.enc.embedSituation(tr.before);
    const actualDelta = this.enc.delta(tr.before, tr.after);
    const situationKey = this.situationToKey(stateEmb);

    // 1. Select best port for this situation (agency-based)
    let port = this.selectPort(tr.action, stateEmb);

    // Handle case where no port is applicable
    if (!port) {
      // Create a new port for this action with a fresh embedding
      port = this.createPort(tr.action);
    }

    // Track for encoder training
    this.lastSelectedPort = port;

    // 2. Update dynamic reference volume
    this.updateReferenceVolume(actualDelta);

    // 3. Get cone and evaluate binding
    const cone = port.getCone(stateEmb);
    const outcome = evaluateBinding(actualDelta, cone);

    // 4. Compute and record agency for fiber-based proliferation
    const agency = port.computeAgencyFor(stateEmb);
    this.history.recordAgency(port.id, agency);

    // 5. Record binding outcome in history
    this.history.record(port.id, situationKey, outcome);

    // 6. Record calibration diagnostics
    // Normalized residual: distance from center / radius
    const normalizedDistance = this.computeNormalizedDistance(actualDelta, cone);
    this.history.recordNormalizedResidual(port.id, normalizedDistance);
    this.history.recordAgencyBindingPair(port.id, agency, outcome.success);

    if (!outcome.success) {
      this.history.recordFailureTrajectory(port.id, actualDelta);
      // Also record for port functor discovery
      this.recordRecentFailure(port.id, stateEmb, actualDelta);
    }

    // 7. Decide refinement action
    const action = this.policy.decide(
      outcome,
      port.id,
      situationKey,
      this.history
    );

    // 8. Handle proliferation separately
    if (action === "proliferate") {
      this.scheduleProliferation(port.id);
      // Record that proliferation occurred (for cooldown)
      this.history.recordProliferation(port.id);
      // Still train CausalNet on the observation
      this.causalNet.observe(stateEmb, port.embedding, actualDelta);
      // Train port embedding even during proliferation
      (port as any).trainEmbedding(stateEmb, actualDelta);
    } else {
      // Normal observation (trains both networks AND embedding)
      port.observe(tr);
    }

    // 9. Process pending proliferations
    await this.processProliferations();
  }

  /**
   * Record a recent failure for port functor discovery.
   */
  private recordRecentFailure(
    portId: string,
    situationEmb: Vec,
    trajectory: Vec
  ): void {
    let failures = this.recentFailures.get(portId);
    if (!failures) {
      failures = [];
      this.recentFailures.set(portId, failures);
    }

    failures.push({ situationEmb: [...situationEmb], trajectory: [...trajectory] });

    // Keep only recent failures
    if (failures.length > this.maxFailuresPerPort) {
      failures.shift();
    }
  }

  /**
   * Schedule a new port to be created from proliferation.
   */
  private scheduleProliferation(parentId: string): void {
    this.proliferationPending.push({ parentId });
  }

  /**
   * Process pending proliferations (create new specialist ports).
   * New ports share the SAME action as parent, but have different embeddings.
   *
   * Uses hybrid approach:
   * 1. Try functor discovery if enabled and failures are structured
   * 2. Fall back to random perturbation otherwise
   */
  private async processProliferations(): Promise<void> {
    for (const { parentId } of this.proliferationPending) {
      const parent = this.allPorts.get(parentId);
      if (!parent) continue;

      // Get recent failures for this port
      const failures = this.recentFailures.get(parentId) ?? [];

      // Hybrid approach: try functor discovery, fallback to random
      const { embedding: newEmbedding, usedFunctor } =
        await generateSpecialistEmbedding(
          parent.embedding,
          failures,
          this.causalNet,
          this.cfg.enablePortFunctors,
          {
            tolerance: this.cfg.portFunctorTolerance,
            maxEpochs: this.cfg.portFunctorMaxEpochs,
            minSamples: 5,
          }
        );

      // Track statistics
      this.portFunctorStats.totalProliferations++;
      if (usedFunctor) {
        this.portFunctorStats.functorBased++;
      } else {
        this.portFunctorStats.randomBased++;
      }

      // New port shares same action (will compete with parent via agency)
      const newPort = this.createPort(parent.action, newEmbedding, parent.aperture);

      // Train the new port on recent failures
      for (const { situationEmb, trajectory } of failures.slice(0, 5)) {
        this.causalNet.observe(situationEmb, newEmbedding, trajectory);
      }

      // Clear failures for this port (new port will handle them)
      this.recentFailures.delete(parentId);
    }

    this.proliferationPending = [];
  }

  /**
   * Convert state embedding to key.
   */
  private situationToKey(stateEmb: Vec): string {
    return stateEmb.map((v) => Math.round(v * 10) / 10).join(",");
  }

  /**
   * Compute normalized distance for calibration tracking.
   * This is the same distance used in evaluateBinding but as a raw number.
   */
  private computeNormalizedDistance(trajectory: Vec, cone: { center: Vec; radius: Vec }): number {
    if (trajectory.length !== cone.center.length) return Infinity;

    let sumSq = 0;
    for (let i = 0; i < trajectory.length; i++) {
      const diff = trajectory[i]! - cone.center[i]!;
      const r = Math.max(cone.radius[i]!, 1e-8);
      sumSq += (diff / r) ** 2;
    }

    return Math.sqrt(sumSq / trajectory.length);
  }

  /**
   * Get predictions for an action from a state embedding.
   * PRIMARY INTERFACE - operates in embedding space.
   * Uses agency-based port selection.
   */
  predict(action: string, stateEmb: Vec, k?: number): Prediction[] {
    const port = this.selectPort(action, stateEmb);
    if (!port) {
      // No applicable port - return empty predictions
      return [];
    }
    return port.predict(stateEmb, k);
  }

  /**
   * Get predictions for an action in a situation (convenience wrapper).
   * Encodes the situation then calls predict(embedding).
   */
  predictFromState(action: string, sit: Situation<S, C>, k?: number): Prediction[] {
    const stateEmb = this.enc.embedSituation(sit);
    return this.predict(action, stateEmb, k);
  }

  /**
   * Get average agency across all observed ports.
   */
  getAgency(): number {
    if (this.allPorts.size === 0) return 0;

    // Only count ports that have been observed (have non-zero steps)
    let total = 0;
    let count = 0;
    for (const port of this.allPorts.values()) {
      if (port.getSteps() > 0) {
        total += port.getAgency();
        count++;
      }
    }
    return count > 0 ? total / count : 0;
  }

  /**
   * Get all port IDs.
   */
  getPortIds(): string[] {
    return Array.from(this.allPorts.keys());
  }

  /**
   * Get number of ports (total across all actions).
   */
  getPortCount(): number {
    return this.allPorts.size;
  }

  /**
   * Get number of ports for a specific action.
   */
  getPortCountForAction(action: string): number {
    return this.portsByAction.get(action)?.length ?? 0;
  }

  /**
   * Get all action names.
   */
  getActions(): string[] {
    return Array.from(this.portsByAction.keys());
  }

  /**
   * Get calibration diagnostics for all ports.
   * Useful for checking if agency is actually predictive of binding success.
   */
  getCalibrationDiagnostics(): Record<string, ReturnType<typeof this.history.getCalibrationDiagnostics>> {
    const result: Record<string, ReturnType<typeof this.history.getCalibrationDiagnostics>> = {};
    for (const portId of this.allPorts.keys()) {
      result[portId] = this.history.getCalibrationDiagnostics(portId);
    }
    return result;
  }

  /**
   * Get port functor statistics.
   * Shows how many proliferations used functor discovery vs random perturbation.
   */
  getPortFunctorStats(): typeof this.portFunctorStats {
    return { ...this.portFunctorStats };
  }

  /**
   * Get the CausalNet instance (for encoder training integration).
   */
  getCausalNet(): CausalNet {
    return this.causalNet;
  }

  /**
   * Get the CommitmentNet instance (for encoder training integration).
   */
  getCommitmentNet(): CommitmentNet {
    return this.commitmentNet;
  }

  /**
   * Get the last selected port (for encoder training integration).
   */
  getLastSelectedPort(): GeometricPort<S, C> | null {
    return this.lastSelectedPort;
  }

  /**
   * Flush all buffered training data.
   */
  flush(): void {
    this.causalNet.flush();
    this.commitmentNet.flush();
  }

  /**
   * Get diagnostic snapshot.
   */
  snapshot(): unknown {
    // Group ports by action for snapshot
    const portsByAction: Record<string, unknown[]> = {};
    for (const [action, ports] of this.portsByAction.entries()) {
      portsByAction[action] = ports.map((p) => p.snapshot());
    }

    return {
      totalPorts: this.allPorts.size,
      actions: Array.from(this.portsByAction.keys()),
      portsPerAction: Object.fromEntries(
        Array.from(this.portsByAction.entries()).map(([k, v]) => [k, v.length])
      ),
      totalAgency: this.getAgency(),
      referenceVolume: this.dynamicReferenceVolume,
      useDynamicReference: this.cfg.useDynamicReference,
      portFunctorStats: this.portFunctorStats,
      portsByAction,
      causalNet: this.causalNet.snapshot(),
      commitmentNet: this.commitmentNet.snapshot(),
      history: this.history.snapshot(),
    };
  }

  /**
   * Get all ports (for diagnostics and experiments).
   */
  getAllPorts(): GeometricPort<S, C>[] {
    return Array.from(this.allPorts.values());
  }

  /**
   * Export all ports for composition.
   * Returns serialized port data that can be imported into another bank.
   */
  exportPorts(): Array<{
    id: string;
    action: string;
    embedding: Vec;
  }> {
    return Array.from(this.allPorts.values()).map(port => ({
      id: port.id,
      action: port.action,
      embedding: [...port.embedding],
    }));
  }

  /**
   * Import a port from another bank.
   * Creates a new port with the given embedding as a candidate specialist.
   *
   * Used for emergent composition: imported ports compete with native ports
   * via agency-based selection and may be sidelined if they perform poorly.
   */
  importPort(data: {
    action: string;
    embedding: Vec;
  }): void {
    // Create port with imported embedding
    this.createPort(data.action, data.embedding);
    // Port is automatically registered in maps and ready for selection
  }

  /**
   * Clean up all resources.
   */
  dispose(): void {
    this.causalNet.dispose();
    this.commitmentNet.dispose();
    this.selectionStrategy.dispose?.();
    this.allPorts.clear();
    this.portsByAction.clear();
  }
}

