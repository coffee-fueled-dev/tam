/**
 * StatsLogger: Structured analytics logging for TAM v2 experiments
 *
 * Creates directory structure:
 *   runs/{run-id}/
 *     metadata.json       - Run configuration and timestamp
 *     checkpoints.jsonl   - Global metrics time series
 *     ports/
 *       port-0.jsonl      - Per-port metrics time series
 *       port-1.jsonl
 *       ...
 */

import type { FileSink } from "bun";
import { mkdir } from "node:fs/promises";

/**
 * Global checkpoint metrics
 */
export interface CheckpointMetrics {
  sample: number;
  portCount: number;
  // Test metrics
  testAgency: number;
  testError: number;
  testBindingRate: number;
  // Training metrics
  trainAgency: number;
  trainError: number;
  trainBindingRate: number;
  // Calibration metrics
  testCalibrationGap?: number;  // |testAgency - testBindingRate|
  trainCalibrationGap?: number; // |trainAgency - trainBindingRate|
  // Cone geometry metrics
  avgConeRadius?: number; // Average cone radius across all dimensions and ports
}

/**
 * Per-port metrics
 */
export interface PortMetrics {
  portIdx: number;
  sample: number;
  samples: number;
  bindings: number;
  bindingRate: number;
  avgAgency: number;
  agencyStdDev: number;
  avgError: number;
  errorStdDev: number;
  [key: string]: number; // Allow additional custom metrics
}

/**
 * Run metadata
 */
export interface RunMetadata {
  runId: string;
  timestamp: string;
  experiment: string;
  config: Record<string, unknown>;
}

/**
 * StatsLogger configuration
 */
export interface StatsLoggerConfig {
  baseDir?: string;  // Base directory for runs (default: "runs")
  runId?: string;    // Custom run ID (default: auto-generated)
}

/**
 * StatsLogger: Manages structured logging for TAM v2 experiments
 */
export class StatsLogger {
  private runId: string;
  private runDir: string;
  private portsDir: string;
  private checkpointLog: FileSink;
  private portWriters: Map<number, FileSink>;

  private constructor(
    runId: string,
    runDir: string,
    portsDir: string,
    checkpointLog: FileSink
  ) {
    this.runId = runId;
    this.runDir = runDir;
    this.portsDir = portsDir;
    this.checkpointLog = checkpointLog;
    this.portWriters = new Map();
  }

  /**
   * Create a new StatsLogger instance
   */
  static async create(
    metadata: Omit<RunMetadata, "runId" | "timestamp">,
    config?: StatsLoggerConfig
  ): Promise<StatsLogger> {
    const baseDir = config?.baseDir ?? "runs";
    const runId = config?.runId ?? `${metadata.experiment.toLowerCase().replace(/\s+/g, "-")}-${Date.now()}`;
    const runDir = `${baseDir}/${runId}`;
    const portsDir = `${runDir}/ports`;

    // Create directory structure
    await mkdir(runDir, { recursive: true });
    await mkdir(portsDir, { recursive: true });

    // Write metadata
    const fullMetadata: RunMetadata = {
      runId,
      timestamp: new Date().toISOString(),
      ...metadata,
    };

    await Bun.write(
      `${runDir}/metadata.json`,
      JSON.stringify(fullMetadata, null, 2)
    );

    // Create checkpoint log writer
    const checkpointLog = Bun.file(`${runDir}/checkpoints.jsonl`).writer();

    return new StatsLogger(runId, runDir, portsDir, checkpointLog);
  }

  /**
   * Log global checkpoint metrics
   */
  logCheckpoint(metrics: CheckpointMetrics): void {
    this.checkpointLog.write(JSON.stringify(metrics) + "\n");
    this.checkpointLog.flush();
  }

  /**
   * Log per-port metrics
   * Creates port file lazily on first write
   */
  logPort(metrics: PortMetrics): void {
    const { portIdx, ...data } = metrics;

    // Create port writer if it doesn't exist
    if (!this.portWriters.has(portIdx)) {
      const portFile = Bun.file(`${this.portsDir}/port-${portIdx}.jsonl`);
      this.portWriters.set(portIdx, portFile.writer());
    }

    const portWriter = this.portWriters.get(portIdx)!;
    portWriter.write(JSON.stringify(data) + "\n");
    portWriter.flush();
  }

  /**
   * Log multiple port metrics at once
   */
  logPorts(portsMetrics: PortMetrics[]): void {
    for (const portMetrics of portsMetrics) {
      this.logPort(portMetrics);
    }
  }

  /**
   * Get run directory path
   */
  getRunDir(): string {
    return this.runDir;
  }

  /**
   * Get run ID
   */
  getRunId(): string {
    return this.runId;
  }

  /**
   * Close all file writers
   */
  close(): void {
    this.checkpointLog.end();
    this.portWriters.forEach((writer) => writer.end());
    this.portWriters.clear();
  }
}
