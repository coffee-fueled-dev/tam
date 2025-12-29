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
import { FixedRefinementPolicy } from "./refinement";
import { GeometricPort } from "./port";
import { evaluateBinding } from "./fibration";

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

  // Dynamic reference volume (calibrated from observed deltas)
  private dynamicReferenceVolume: number;
  private observedDeltaMagnitudes: number[] = [];

  // Proliferation tracking
  private proliferationPending: Array<{
    parentId: string;
    situation: Situation<S, C>;
    trajectory: Vec;
  }> = [];

  constructor(
    encoders: Encoders<S, C>,
    config?: GeometricPortConfigInput
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
    });
    this.policy = new FixedRefinementPolicy(this.cfg);
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
   * Select the best port for an action in a given situation.
   * Uses agency-based selection: the port with highest agency (narrowest cone)
   * among those with non-empty cones is chosen.
   *
   * Returns null if no port is applicable (all have empty cones).
   */
  selectPort(action: string, sit: Situation<S, C>): GeometricPort<S, C> | null {
    const candidates = this.portsByAction.get(action);

    if (!candidates || candidates.length === 0) {
      // No ports exist for this action - create one
      return this.createPort(action);
    }

    // Compute agency for each candidate
    const withAgency = candidates.map((port) => ({
      port,
      agency: port.computeAgencyFor(sit),
      applicable: port.isApplicable(sit),
    }));

    // Filter to applicable ports (non-empty cones)
    const eligible = withAgency.filter(({ applicable }) => applicable);

    if (eligible.length === 0) {
      // No port applies - return null (caller should handle, e.g., proliferate)
      return null;
    }

    // Select max agency (narrowest cone = most specific commitment)
    return eligible.reduce((best, curr) =>
      curr.agency > best.agency ? curr : best
    ).port;
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
   * Observe a transition with agency-based port selection and potential proliferation.
   */
  observe(tr: Transition<S, C>): void {
    // 1. Select best port for this situation (agency-based)
    let port = this.selectPort(tr.action, tr.before);

    // Handle case where no port is applicable
    if (!port) {
      // Create a new port for this action with a fresh embedding
      port = this.createPort(tr.action);
    }

    const stateEmb = this.enc.embedSituation(tr.before);
    const actualDelta = this.enc.delta(tr.before, tr.after);
    const situationKey = this.situationToKey(stateEmb);

    // 2. Update dynamic reference volume
    this.updateReferenceVolume(actualDelta);

    // 3. Get cone and evaluate binding
    const cone = port.getCone(tr.before);
    const outcome = evaluateBinding(actualDelta, cone);

    // 4. Compute and record agency for fiber-based proliferation
    const agency = port.computeAgencyFor(tr.before);
    this.history.recordAgency(port.id, agency);

    // 5. Record binding outcome in history
    this.history.record(port.id, situationKey, outcome);

    if (!outcome.success) {
      this.history.recordFailureTrajectory(port.id, actualDelta);
    }

    // 6. Decide refinement action
    const action = this.policy.decide(
      outcome,
      port.id,
      situationKey,
      this.history
    );

    // 7. Handle proliferation separately
    if (action === "proliferate") {
      this.scheduleProliferation(port.id, tr.before, actualDelta);
      // Record that proliferation occurred (for cooldown)
      this.history.recordProliferation(port.id);
      // Still train CausalNet on the observation
      this.causalNet.observe(stateEmb, port.embedding, actualDelta);
    } else {
      // Normal observation (trains both networks)
      port.observe(tr);
    }

    // 8. Process pending proliferations
    this.processProliferations();
  }

  /**
   * Schedule a new port to be created from proliferation.
   */
  private scheduleProliferation(
    parentId: string,
    situation: Situation<S, C>,
    trajectory: Vec
  ): void {
    this.proliferationPending.push({
      parentId,
      situation,
      trajectory,
    });
  }

  /**
   * Process pending proliferations (create new specialist ports).
   * New ports share the SAME action as parent, but have different embeddings.
   */
  private processProliferations(): void {
    for (const { parentId, situation, trajectory } of this.proliferationPending) {
      const parent = this.allPorts.get(parentId);
      if (!parent) continue;

      // Create new port with perturbed embedding (same action, different ID)
      const newEmbedding = parent.embedding.map(
        (v) => v + (Math.random() - 0.5) * 0.2
      );

      // New port shares same action (will compete with parent via agency)
      const newPort = this.createPort(parent.action, newEmbedding, parent.aperture);

      // Train the new port on the triggering trajectory
      const stateEmb = this.enc.embedSituation(situation);
      this.causalNet.observe(stateEmb, newEmbedding, trajectory);
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
   * Get predictions for an action in a situation.
   * Uses agency-based port selection.
   */
  predict(action: string, sit: Situation<S, C>, k?: number): Prediction[] {
    const port = this.selectPort(action, sit);
    if (!port) {
      // No applicable port - return empty predictions
      return [];
    }
    return port.predict(sit, k);
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
      portsByAction,
      causalNet: this.causalNet.snapshot(),
      commitmentNet: this.commitmentNet.snapshot(),
      history: this.history.snapshot(),
    };
  }

  /**
   * Clean up all resources.
   */
  dispose(): void {
    this.causalNet.dispose();
    this.commitmentNet.dispose();
    this.allPorts.clear();
    this.portsByAction.clear();
  }
}

