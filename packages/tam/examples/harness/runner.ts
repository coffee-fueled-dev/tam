/**
 * Experiment Runner
 *
 * Core harness for running TAM experiments with standardized:
 * - Training loops
 * - Metric collection
 * - Checkpointing
 * - Evaluation
 * - Analysis
 */

import {
  GeometricPortBank,
  createEncoderBridge,
  type Situation,
  type Transition,
  type EncoderBridge,
} from "../../src";
import { norm, sub } from "../../src/vec";
import type {
  Domain,
  ExperimentConfig,
  ExperimentResult,
  Checkpoint,
  Metrics,
  AggregateMetrics,
  TrainingCallback,
  EvaluationCallback,
} from "./types";

// ============================================================================
// Core Training Loop
// ============================================================================

/**
 * Run a complete experiment: train, evaluate, analyze.
 */
export async function runExperiment<S>(
  name: string,
  config: ExperimentConfig<S>,
  trainingCallback?: TrainingCallback<S>,
  evaluationCallback?: EvaluationCallback<S>
): Promise<ExperimentResult<S>> {
  const {
    domain,
    episodes,
    checkpointEvery = 100,
    testSize = 100,
    encoder,
    bankConfig,
  } = config;

  console.log(`\n${"═".repeat(63)}`);
  console.log(`  Experiment: ${name}`);
  console.log(`${"═".repeat(63)}\n`);
  console.log(`Domain: ${domain.name}`);
  console.log(`Episodes: ${episodes}`);
  console.log(`Test size: ${testSize}`);
  console.log(`Checkpoints: every ${checkpointEvery} episodes\n`);

  // Create encoder bridge
  const bridge = createEncoderBridge<S>({
    extractRaw: domain.extractRaw || domain.embed,
    ...(encoder?.type === "learnable" && encoder.config
      ? { learnableEncoder: encoder.config }
      : { staticEmbedder: domain.embed }),
  });

  // Create bank
  const bank = new GeometricPortBank(
    bridge.encoders,
    bankConfig || { embeddingDim: domain.embeddingDim }
  );

  // Generate fixed test set
  const testStates = Array.from({ length: testSize }, () => domain.randomState());

  // Training loop with checkpointing
  const checkpoints: Checkpoint[] = [];
  let lastCheckpoint = 0;

  for (let ep = 1; ep <= episodes; ep++) {
    // Sample and observe transition
    const before = domain.randomState();
    const after = domain.simulate(before);

    const transition: Transition<S, unknown> = {
      before: { state: before, context: null },
      after: { state: after, context: null },
      action: "step",
    };

    await bank.observe(transition);

    // Custom training logic (e.g., encoder training)
    if (trainingCallback) {
      await trainingCallback(transition, bank, ep);
    }

    // Periodic flush
    if (ep % 50 === 0) {
      bank.flush();
    }

    // Checkpoint
    if (ep % checkpointEvery === 0 || ep === episodes) {
      bank.flush();

      console.log(`\n[${ep}/${episodes}] Checkpoint...`);

      // Collect training metrics
      const trainMetrics = collectTrainingMetrics(bank);

      // Evaluate on test set
      const testMetrics = await evaluateBank(
        bank,
        domain,
        testStates,
        evaluationCallback
      );

      const checkpoint: Checkpoint = {
        episode: ep,
        train: trainMetrics,
        test: testMetrics,
        timestamp: Date.now(),
      };

      checkpoints.push(checkpoint);
      lastCheckpoint = ep;

      // Report progress
      console.log(`  Test Error: ${testMetrics.predictionError?.toFixed(4) ?? "N/A"}`);
      console.log(`  Agency: ${((testMetrics.agency ?? 0) * 100).toFixed(1)}%`);
      console.log(`  Coverage: ${((testMetrics.coverageRate ?? 0) * 100).toFixed(1)}%`);
      console.log(`  Ports: ${testMetrics.portCount ?? trainMetrics.portCount ?? 0}`);
    }
  }

  // Analyze results
  const summary = analyzeResults(checkpoints);

  console.log(`\n${"═".repeat(63)}`);
  console.log(`  Analysis`);
  console.log(`${"═".repeat(63)}\n`);
  console.log(`Grokked: ${summary.grokked ? "✓ Yes" : "✗ No"}${summary.grokkingEpisode ? ` (episode ${summary.grokkingEpisode})` : ""}`);
  console.log(`Well-Calibrated: ${summary.wellCalibrated ? "✓ Yes" : "⚠ Needs work"}`);
  console.log(`Final Error: ${summary.finalMetrics.predictionError?.toFixed(4) ?? "N/A"}`);
  console.log(`Final Agency: ${((summary.finalMetrics.agency ?? 0) * 100).toFixed(1)}%`);
  console.log(`Final Coverage: ${((summary.finalMetrics.coverageRate ?? 0) * 100).toFixed(1)}%`);

  return {
    name,
    config,
    checkpoints,
    bank, // Caller should dispose!
    summary,
  };
}

// ============================================================================
// Metric Collection
// ============================================================================

/**
 * Collect training metrics from bank's internal tracking.
 */
function collectTrainingMetrics<S>(
  bank: GeometricPortBank<S, unknown>
): Partial<Metrics> {
  // Get calibration from bank
  const calibration = bank.getCalibrationDiagnostics();
  const portIds = Object.keys(calibration);

  if (portIds.length === 0) {
    return { portCount: 0 };
  }

  // Aggregate across ports
  let totalCoverage = 0;
  let totalCorrelation = 0;

  for (const portId of portIds) {
    const cal = calibration[portId]!;
    totalCoverage += cal.coverageRate;
    totalCorrelation += cal.agencyBindingCorrelation;
  }

  return {
    coverageRate: totalCoverage / portIds.length,
    agencyErrorCorrelation: totalCorrelation / portIds.length,
    portCount: portIds.length,
  };
}

/**
 * Evaluate bank on test set.
 */
async function evaluateBank<S>(
  bank: GeometricPortBank<S, unknown>,
  domain: Domain<S>,
  testStates: S[],
  customEval?: EvaluationCallback<S>
): Promise<Partial<AggregateMetrics>> {
  // Custom evaluation if provided
  if (customEval) {
    const custom = await customEval(bank, testStates);
    return { ...collectTrainingMetrics(bank), ...custom };
  }

  // Standard evaluation: prediction error and agency
  let totalError = 0;
  let totalAgency = 0;
  let validSamples = 0;

  const agencyErrorPairs: Array<{ agency: number; error: number }> = [];

  for (const state of testStates) {
    const sit: Situation<S, unknown> = { state, context: null };
    const predictions = bank.predictFromState("step", sit, 1);
    const pred = predictions[0];
    if (!pred) continue;

    // Compute prediction error
    const actualNext = domain.simulate(state);
    const actualEmb = domain.embed(state);
    const nextEmb = domain.embed(actualNext);
    const actualTrajectory = sub(nextEmb, actualEmb);
    const error = norm(sub(pred.delta, actualTrajectory));

    totalError += error;
    totalAgency += pred.agency;
    validSamples++;

    agencyErrorPairs.push({ agency: pred.agency, error });
  }

  if (validSamples === 0) {
    return collectTrainingMetrics(bank);
  }

  const meanError = totalError / validSamples;
  const meanAgency = totalAgency / validSamples;

  // Compute correlation
  let covariance = 0;
  let varianceError = 0;
  let varianceAgency = 0;

  for (const pair of agencyErrorPairs) {
    const errDiff = pair.error - meanError;
    const agDiff = pair.agency - meanAgency;
    covariance += errDiff * agDiff;
    varianceError += errDiff * errDiff;
    varianceAgency += agDiff * agDiff;
  }

  const correlation =
    varianceError > 0 && varianceAgency > 0
      ? covariance / Math.sqrt(varianceError * varianceAgency)
      : 0;

  // Compute calibration curve
  const buckets = [
    { range: "[0.0-0.2]", min: 0.0, max: 0.2, errors: [] as number[] },
    { range: "[0.2-0.4]", min: 0.2, max: 0.4, errors: [] as number[] },
    { range: "[0.4-0.6]", min: 0.4, max: 0.6, errors: [] as number[] },
    { range: "[0.6-0.8]", min: 0.6, max: 0.8, errors: [] as number[] },
    { range: "[0.8-1.0]", min: 0.8, max: 1.0, errors: [] as number[] },
  ];

  for (const pair of agencyErrorPairs) {
    for (const bucket of buckets) {
      if (pair.agency >= bucket.min && pair.agency < bucket.max) {
        bucket.errors.push(pair.error);
        break;
      }
    }
  }

  const calibrationCurve = buckets
    .filter((b) => b.errors.length > 0)
    .map((b) => ({
      agencyRange: b.range,
      avgError: b.errors.reduce((s, e) => s + e, 0) / b.errors.length,
      bindingRate: 0, // Would need actual cone checks
      count: b.errors.length,
    }));

  // Get training metrics too
  const trainMetrics = collectTrainingMetrics(bank);

  return {
    predictionError: meanError,
    errorStd: Math.sqrt(varianceError / validSamples),
    agency: meanAgency,
    coverageRate: trainMetrics.coverageRate,
    agencyErrorCorrelation: correlation,
    portCount: trainMetrics.portCount,
    samples: validSamples,
    calibrationCurve,
  };
}

// ============================================================================
// Analysis
// ============================================================================

/**
 * Analyze experiment results to detect grokking, calibration quality, etc.
 */
function analyzeResults(checkpoints: Checkpoint[]): {
  grokked: boolean;
  grokkingEpisode?: number;
  wellCalibrated: boolean;
  finalMetrics: Partial<AggregateMetrics>;
} {
  if (checkpoints.length === 0) {
    return {
      grokked: false,
      wellCalibrated: false,
      finalMetrics: {},
    };
  }

  // Detect grokking: sudden error drop (>30%)
  let grokked = false;
  let grokkingEpisode: number | undefined;

  for (let i = 1; i < checkpoints.length; i++) {
    const prev = checkpoints[i - 1]!.test.predictionError;
    const curr = checkpoints[i]!.test.predictionError;

    if (prev && curr) {
      const drop = (prev - curr) / prev;
      if (drop > 0.3) {
        grokked = true;
        grokkingEpisode = checkpoints[i]!.episode;
        break;
      }
    }
  }

  // Check final calibration
  const final = checkpoints[checkpoints.length - 1]!.test;
  const wellCalibrated =
    (final.predictionError ?? 1) < 0.05 &&
    (final.agency ?? 0) > 0.7 &&
    (final.coverageRate ?? 0) > 0.85;

  return {
    grokked,
    grokkingEpisode,
    wellCalibrated,
    finalMetrics: final as AggregateMetrics,
  };
}

// ============================================================================
// Comparison Utilities
// ============================================================================

/**
 * Compare multiple experiment results side-by-side.
 */
export function compareExperiments<S>(
  results: ExperimentResult<S>[]
): void {
  console.log(`\n${"═".repeat(63)}`);
  console.log(`  Comparison`);
  console.log(`${"═".repeat(63)}\n`);

  console.log("| Experiment | Final Error | Agency | Coverage | Grokked |");
  console.log("|------------|-------------|--------|----------|---------|");

  for (const result of results) {
    const m = result.summary.finalMetrics;
    console.log(
      `| ${result.name.padEnd(10)} | ${(m.predictionError ?? 0).toFixed(4).padStart(11)} | ${((m.agency ?? 0) * 100).toFixed(1).padStart(5)}% | ${((m.coverageRate ?? 0) * 100).toFixed(1).padStart(7)}% | ${result.summary.grokked ? "✓".padStart(7) : "✗".padStart(7)} |`
    );
  }

  console.log();
}

// ============================================================================
// Export Utilities
// ============================================================================

/**
 * Save experiment result to JSON file.
 *
 * Serializes the experiment result (excluding the non-serializable bank)
 * to a JSON file. Useful for archiving results and later analysis.
 *
 * @param result - The experiment result to save
 * @param filepath - Path to save the JSON file (e.g., "results/01-basic.json")
 *
 * @example
 * ```typescript
 * const result = await runExperiment("Basic Training", config);
 * await saveResultsToJson(result, "examples/results/01-basic.json");
 * ```
 */
export async function saveResultsToJson<S>(
  result: ExperimentResult<S>,
  filepath: string
): Promise<void> {
  // Create serializable result (exclude bank)
  const serializable = {
    name: result.name,
    config: {
      domain: {
        name: result.config.domain.name,
        embeddingDim: result.config.domain.embeddingDim,
        rawDim: result.config.domain.rawDim,
      },
      episodes: result.config.episodes,
      checkpointEvery: result.config.checkpointEvery,
      testSize: result.config.testSize,
      horizons: result.config.horizons,
      encoder: result.config.encoder,
      bankConfig: result.config.bankConfig,
      options: result.config.options,
    },
    checkpoints: result.checkpoints,
    summary: result.summary,
    timestamp: new Date().toISOString(),
  };

  // Write to file
  const json = JSON.stringify(serializable, null, 2);
  await Bun.write(filepath, json);

  console.log(`✓ Results saved to ${filepath}`);
}
