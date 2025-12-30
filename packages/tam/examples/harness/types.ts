/**
 * Experiment Harness Types
 *
 * Provides standardized interfaces for domains, metrics, and experiment configuration.
 * Goal: Make experiments concise, comparable, and easy to analyze.
 */

import type { GeometricPortBank, Situation, Transition } from "../../src";

// ============================================================================
// Domain Interface
// ============================================================================

/**
 * Standardized domain specification.
 * All experiments use this interface for consistency.
 */
export interface Domain<S> {
  /** Domain name for reporting */
  name: string;

  /** Generate random initial state */
  randomState(): S;

  /** Simulate k steps forward (default k=1) */
  simulate(state: S, steps?: number): S;

  /** Embed state as vector */
  embed(state: S): number[];

  /** Dimension of embedding */
  embeddingDim: number;

  /** Optional: Extract raw features (for learnable encoders) */
  extractRaw?(state: S): number[];

  /** Optional: Dimension of raw features */
  rawDim?: number;
}

// ============================================================================
// Metrics & Tracking
// ============================================================================

/**
 * Core metrics tracked during training and evaluation.
 */
export interface Metrics {
  /** Prediction error: ||predicted - actual|| */
  predictionError: number;

  /** Agency: 1 - normalized cone volume */
  agency: number;

  /** Coverage: % of trajectories inside cones */
  coverageRate: number;

  /** Calibration: Correlation between agency and accuracy */
  agencyErrorCorrelation: number;

  /** Number of specialist ports */
  portCount: number;

  /** Sample count */
  samples: number;
}

/**
 * Metrics aggregated over multiple evaluations.
 */
export interface AggregateMetrics extends Metrics {
  /** Standard deviation of prediction error */
  errorStd: number;

  /** Calibration buckets: agency range → metrics */
  calibrationCurve: Array<{
    agencyRange: string;
    avgError: number;
    bindingRate: number;
    count: number;
  }>;
}

/**
 * Training checkpoint with metrics.
 */
export interface Checkpoint {
  /** Episode/sample count */
  episode: number;

  /** Training metrics (from bank's internal tracking) */
  train: Partial<Metrics>;

  /** Test metrics (from held-out evaluation) */
  test: Partial<AggregateMetrics>;

  /** Timestamp */
  timestamp: number;
}

// ============================================================================
// Experiment Configuration
// ============================================================================

/**
 * Configuration for an experiment run.
 */
export interface ExperimentConfig<S> {
  /** Domain to use */
  domain: Domain<S>;

  /** Number of training episodes */
  episodes: number;

  /** Checkpoint frequency (episodes) */
  checkpointEvery?: number;

  /** Test set size */
  testSize?: number;

  /** Multi-horizon settings (if enabled) */
  horizons?: number[];

  /** Encoder configuration */
  encoder?: {
    type: "static" | "learnable";
    config?: any;
  };

  /** Port bank configuration */
  bankConfig?: any;

  /** Experiment-specific options */
  options?: Record<string, any>;
}

/**
 * Result of an experiment run.
 */
export interface ExperimentResult<S> {
  /** Experiment name */
  name: string;

  /** Configuration used */
  config: ExperimentConfig<S>;

  /** All checkpoints */
  checkpoints: Checkpoint[];

  /** Final trained bank (disposed after analysis) */
  bank?: GeometricPortBank<S, unknown>;

  /** Analysis summary */
  summary: {
    grokked: boolean;
    grokkingEpisode?: number;
    wellCalibrated: boolean;
    finalMetrics: Partial<AggregateMetrics>;
  };
}

// ============================================================================
// Training & Evaluation Utilities
// ============================================================================

/**
 * Callback for custom training logic (e.g., encoder training).
 */
export type TrainingCallback<S> = (
  transition: Transition<S, unknown>,
  bank: GeometricPortBank<S, unknown>,
  episode: number
) => Promise<void> | void;

/**
 * Callback for custom evaluation logic.
 */
export type EvaluationCallback<S> = (
  bank: GeometricPortBank<S, unknown>,
  testStates: S[]
) => Promise<Partial<Metrics>> | Partial<Metrics>;
