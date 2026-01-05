/**
 * LatentLogger: Tracks port embeddings and commitment behavior over time
 *
 * Since cone volumes (commitment distances) are state-dependent, we evaluate
 * the actor at a fixed set of representative states at each checkpoint.
 */

import type { Actor } from "../actor";
import type { Vec } from "../types";
import type { FileSink } from "bun";
import { join } from "node:path";

/**
 * Port state snapshot at a checkpoint
 */
export interface PortSnapshot {
  portIdx: number;
  embedding: Vec;
  commitments: Array<{
    stateIdx: number;
    concentration: number;  // Concentration parameter at this state
    selected: boolean;      // Was this port selected for this state?
  }>;
}

/**
 * Checkpoint snapshot of all ports
 */
export interface LatentCheckpoint {
  sample: number;
  ports: PortSnapshot[];
}

/**
 * LatentLogger configuration
 */
export interface LatentLoggerConfig {
  /** Number of representative states to track (default: 10) */
  numStates?: number;
}

/**
 * LatentLogger: Tracks latent space evolution
 */
export class LatentLogger<S> {
  private runDir: string;
  private representativeStates: S[];
  private embedState: (s: S) => Vec;
  private latentLog: FileSink;

  constructor(
    runDir: string,
    representativeStates: S[],
    embedState: (s: S) => Vec
  ) {
    this.runDir = runDir;
    this.representativeStates = representativeStates;
    this.embedState = embedState;
    this.latentLog = Bun.file(join(runDir, "latent.jsonl")).writer();
  }

  /**
   * Create a new LatentLogger instance
   */
  static create<S>(
    runDir: string,
    generateState: () => S,
    embedState: (s: S) => Vec,
    config?: LatentLoggerConfig
  ): LatentLogger<S> {
    const numStates = config?.numStates ?? 10;
    const representativeStates = Array.from({ length: numStates }, () =>
      generateState()
    );

    return new LatentLogger(runDir, representativeStates, embedState);
  }

  /**
   * Log current latent space state at a checkpoint
   */
  logCheckpoint(sample: number, actor: Actor<S>): void {
    const checkpoint: LatentCheckpoint = {
      sample,
      ports: [],
    };

    // Get port commitments at each representative state
    const commitmentsByPort = new Map<number, PortSnapshot["commitments"]>();

    for (let stateIdx = 0; stateIdx < this.representativeStates.length; stateIdx++) {
      const state = this.representativeStates[stateIdx]!;
      const commitments = actor.getPortCommitments(state);

      for (const commit of commitments) {
        if (!commitmentsByPort.has(commit.portIdx)) {
          commitmentsByPort.set(commit.portIdx, []);
        }

        commitmentsByPort.get(commit.portIdx)!.push({
          stateIdx,
          concentration: commit.concentration,
          selected: commit.selected,
        });
      }
    }

    // Build port snapshots
    for (const [portIdx, commitments] of commitmentsByPort) {
      const portEmbeddings = actor.exportPorts();
      checkpoint.ports.push({
        portIdx,
        embedding: portEmbeddings[portIdx]!,
        commitments,
      });
    }

    this.latentLog.write(JSON.stringify(checkpoint) + "\n");
    this.latentLog.flush();
  }

  /**
   * Get representative states (for saving to metadata)
   */
  getRepresentativeStates(): Array<{ stateIdx: number; embedding: Vec }> {
    return this.representativeStates.map((state, idx) => ({
      stateIdx: idx,
      embedding: this.embedState(state),
    }));
  }

  /**
   * Close the logger
   */
  close(): void {
    this.latentLog.end();
  }
}
