/**
 * Complete Calibration Analysis: Combining Both Perspectives
 *
 * Two types of calibration:
 * 1. TRAINING-TIME (Bank's built-in tracking):
 *    - Records (agency, binding_success) during observe()
 *    - Measures: Does reality honor the actor's commitments?
 *    - Epistemic integrity: Do cones contain actual trajectories?
 *
 * 2. TEST-TIME (Manual tracking):
 *    - Records (agency, prediction_error) during predict()
 *    - Measures: Does confidence correlate with accuracy?
 *    - Predictive calibration: Are high-agency predictions more accurate?
 *
 * Both are important! They answer different questions:
 * - Binding: "Is trajectory inside the cone?" (geometric containment)
 * - Accuracy: "How close is prediction to reality?" (error magnitude)
 *
 * Well-calibrated model should have:
 * - High agency → Low error (confident = accurate)
 * - High agency → High binding rate (commitments honored)
 * - Low error + High binding → High agency (accurate = confident)
 */

import {
  GeometricPortBank,
  type Situation,
  type Transition,
  type GeometricPortConfigInput,
  createEncoderBridge,
} from "../../src";
import { norm, sub } from "../../src/vec";

// ============================================================================
// Domain: 1D Damped Spring
// ============================================================================

interface SpringState {
  x: number;
  v: number;
}

const k = 1.0;
const b = 0.1;
const dt = 0.1;

function simulate(state: SpringState): SpringState {
  const ax = -k * state.x - b * state.v;
  const newV = state.v + ax * dt;
  const newX = state.x + newV * dt;
  return { x: newX, v: newV };
}

function randomState(xRange: [number, number], vRange: [number, number]): SpringState {
  const x = xRange[0] + Math.random() * (xRange[1] - xRange[0]);
  const v = vRange[0] + Math.random() * (vRange[1] - vRange[0]);
  return { x, v };
}

function embedState(state: SpringState): number[] {
  return [state.x, state.v];
}

// ============================================================================
// Training with Both Calibration Analyses
// ============================================================================

interface CalibrationResult {
  episodes: number;

  // Test-time calibration (manual tracking)
  testError: number;
  testAgency: number;
  testAgencyErrorCorrelation: number;
  testCalibrationBuckets: Array<{
    agencyRange: string;
    avgError: number;
    count: number;
  }>;

  // Training-time calibration (bank's tracking)
  trainCoverage: number;
  trainAgencyBindingCorrelation: number;
  trainCalibrationBuckets: Array<{
    agencyRange: string;
    bindingRate: number;
    count: number;
  }>;
}

async function trainAndEvaluate(
  episodes: number,
  xRange: [number, number],
  vRange: [number, number]
): Promise<CalibrationResult> {
  const bridge = createEncoderBridge<SpringState>({
    extractRaw: embedState,
  });

  const config: GeometricPortConfigInput = {
    embeddingDim: 2,
  };

  const bank = new GeometricPortBank(bridge.encoders, config);

  // Training (with automatic calibration tracking)
  for (let i = 0; i < episodes; i++) {
    const before = randomState(xRange, vRange);
    const after = simulate(before);

    const transition: Transition<SpringState, unknown> = {
      before: { state: before, context: null },
      after: { state: after, context: null },
      action: "step",
    };

    await bank.observe(transition);

    if ((i + 1) % 50 === 0) {
      bank.flush();
    }
  }

  bank.flush();

  // ============================================================================
  // TEST-TIME CALIBRATION: (agency, error) pairs
  // ============================================================================

  const testStates = Array.from({ length: 200 }, () => randomState(xRange, vRange));
  const testPairs: Array<{ agency: number; error: number }> = [];

  for (const state of testStates) {
    const sit: Situation<SpringState, unknown> = { state, context: null };
    const predictions = bank.predict("step", sit, 1);
    const pred = predictions[0];
    if (!pred) continue;

    // Compute actual trajectory and error
    const actualNext = simulate(state);
    const actualTrajectory = [actualNext.x - state.x, actualNext.v - state.v];
    const error = norm(sub(pred.delta, actualTrajectory));

    testPairs.push({ agency: pred.agency, error });
  }

  // Aggregate test metrics
  const testError = testPairs.reduce((sum, p) => sum + p.error, 0) / testPairs.length;
  const testAgency = testPairs.reduce((sum, p) => sum + p.agency, 0) / testPairs.length;

  // Compute correlation between agency and error
  const meanError = testError;
  const meanAgency = testAgency;
  let covariance = 0;
  let varianceError = 0;
  let varianceAgency = 0;

  for (const pair of testPairs) {
    const errDiff = pair.error - meanError;
    const agDiff = pair.agency - meanAgency;
    covariance += errDiff * agDiff;
    varianceError += errDiff * errDiff;
    varianceAgency += agDiff * agDiff;
  }

  const testAgencyErrorCorrelation = varianceError > 0 && varianceAgency > 0
    ? covariance / Math.sqrt(varianceError * varianceAgency)
    : 0;

  // Bucket by agency for test-time calibration curve
  const testBuckets = [
    { range: "[0.0-0.2]", min: 0.0, max: 0.2, errors: [] as number[] },
    { range: "[0.2-0.4]", min: 0.2, max: 0.4, errors: [] as number[] },
    { range: "[0.4-0.6]", min: 0.4, max: 0.6, errors: [] as number[] },
    { range: "[0.6-0.8]", min: 0.6, max: 0.8, errors: [] as number[] },
    { range: "[0.8-1.0]", min: 0.8, max: 1.0, errors: [] as number[] },
  ];

  for (const pair of testPairs) {
    for (const bucket of testBuckets) {
      if (pair.agency >= bucket.min && pair.agency < bucket.max) {
        bucket.errors.push(pair.error);
        break;
      }
    }
  }

  const testCalibrationBuckets = testBuckets
    .filter(b => b.errors.length > 0)
    .map(b => ({
      agencyRange: b.range,
      avgError: b.errors.reduce((s, e) => s + e, 0) / b.errors.length,
      count: b.errors.length,
    }));

  // ============================================================================
  // TRAINING-TIME CALIBRATION: From bank's history
  // ============================================================================

  const trainCalibration = bank.getCalibrationDiagnostics();
  const portIds = Object.keys(trainCalibration);

  // Aggregate across all ports
  let totalCoverage = 0;
  let totalCorrelation = 0;
  let portCount = 0;

  const aggregatedTrainBuckets = new Map<string, { successes: number; total: number }>();

  for (const portId of portIds) {
    const cal = trainCalibration[portId]!;
    totalCoverage += cal.coverageRate;
    totalCorrelation += cal.agencyBindingCorrelation;
    portCount++;

    // Aggregate calibration buckets
    for (const bucket of cal.calibrationBuckets) {
      const existing = aggregatedTrainBuckets.get(bucket.agencyRange);
      const successes = Math.round(bucket.bindingRate * bucket.count);
      if (existing) {
        existing.successes += successes;
        existing.total += bucket.count;
      } else {
        aggregatedTrainBuckets.set(bucket.agencyRange, { successes, total: bucket.count });
      }
    }
  }

  const trainCoverage = portCount > 0 ? totalCoverage / portCount : 0;
  const trainAgencyBindingCorrelation = portCount > 0 ? totalCorrelation / portCount : 0;

  const trainCalibrationBuckets = Array.from(aggregatedTrainBuckets.entries()).map(
    ([range, data]) => ({
      agencyRange: range,
      bindingRate: data.total > 0 ? data.successes / data.total : 0,
      count: data.total,
    })
  );

  bank.dispose();

  return {
    episodes,
    testError,
    testAgency,
    testAgencyErrorCorrelation,
    testCalibrationBuckets,
    trainCoverage,
    trainAgencyBindingCorrelation,
    trainCalibrationBuckets,
  };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Complete Calibration Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Two perspectives on calibration:\n");
  console.log("1. TEST-TIME: Does confidence predict accuracy?");
  console.log("   - High agency → Low prediction error?");
  console.log("   - Measures: Predictive calibration\n");

  console.log("2. TRAINING-TIME: Does reality honor commitments?");
  console.log("   - High agency → High binding rate?");
  console.log("   - Measures: Epistemic integrity\n");

  const checkpoints = [100, 200, 500, 1000];
  const results: CalibrationResult[] = [];

  for (const ep of checkpoints) {
    console.log(`\n${"─".repeat(61)}`);
    console.log(`Training with ${ep} episodes...`);
    const result = await trainAndEvaluate(ep, [-1, 1], [-1, 1]);
    results.push(result);

    console.log(`\nTest-Time Calibration (Accuracy):`);
    console.log(`  Mean Error: ${result.testError.toFixed(4)}`);
    console.log(`  Mean Agency: ${(result.testAgency * 100).toFixed(1)}%`);
    console.log(`  Agency-Error Correlation: ${result.testAgencyErrorCorrelation.toFixed(3)}`);
    console.log(`  Calibration Curve:`);
    for (const bucket of result.testCalibrationBuckets) {
      console.log(`    ${bucket.agencyRange}: error=${bucket.avgError.toFixed(4)} (n=${bucket.count})`);
    }

    console.log(`\nTraining-Time Calibration (Binding):`);
    console.log(`  Coverage Rate: ${(result.trainCoverage * 100).toFixed(1)}%`);
    console.log(`  Agency-Binding Correlation: ${result.trainAgencyBindingCorrelation.toFixed(3)}`);
    if (result.trainCalibrationBuckets.length > 0) {
      console.log(`  Calibration Curve:`);
      for (const bucket of result.trainCalibrationBuckets) {
        console.log(`    ${bucket.agencyRange}: binding=${(bucket.bindingRate * 100).toFixed(1)}% (n=${bucket.count})`);
      }
    } else {
      console.log(`  (No calibration data - too few samples per bucket)`);
    }
  }

  // ============================================================================
  // Analysis
  // ============================================================================

  console.log("\n" + "═".repeat(61));
  console.log("  Analysis: Grokking & Calibration");
  console.log("═".repeat(61) + "\n");

  // Check for grokking
  for (let i = 1; i < results.length; i++) {
    const prev = results[i - 1]!;
    const curr = results[i]!;
    const errorDrop = prev.testError - curr.testError;
    const errorDropPct = (errorDrop / prev.testError) * 100;

    if (errorDropPct > 30) {
      console.log(`✨ GROKKING: ${prev.episodes} → ${curr.episodes} episodes`);
      console.log(`   Error: ${prev.testError.toFixed(4)} → ${curr.testError.toFixed(4)} (${errorDropPct.toFixed(0)}% drop)`);
      console.log(`   Agency: ${(prev.testAgency * 100).toFixed(1)}% → ${(curr.testAgency * 100).toFixed(1)}%`);
      console.log(`   Coverage: ${(prev.trainCoverage * 100).toFixed(1)}% → ${(curr.trainCoverage * 100).toFixed(1)}%\n`);
    }
  }

  // Final state analysis
  const final = results[results.length - 1]!;
  console.log("Final State:");
  console.log(`  Prediction Error: ${final.testError.toFixed(4)}`);
  console.log(`  Test Agency: ${(final.testAgency * 100).toFixed(1)}%`);
  console.log(`  Training Coverage: ${(final.trainCoverage * 100).toFixed(1)}%`);
  console.log(`  Test Correlation (agency-error): ${final.testAgencyErrorCorrelation.toFixed(3)}`);
  console.log(`  Train Correlation (agency-binding): ${final.trainAgencyBindingCorrelation.toFixed(3)}\n`);

  // Check calibration quality
  const wellCalibratedAccuracy = final.testError < 0.05;
  const wellCalibratedAgency = final.testAgency > 0.7;
  const wellCalibratedCoverage = final.trainCoverage > 0.85;
  const goodTestCorr = final.testAgencyErrorCorrelation < -0.3; // Negative: high agency = low error
  const goodTrainCorr = final.trainAgencyBindingCorrelation > 0.3; // Positive: high agency = high binding

  if (wellCalibratedAccuracy && wellCalibratedAgency && wellCalibratedCoverage) {
    console.log("✓ WELL-CALIBRATED");
    console.log("  → Accurate predictions (low error)");
    console.log("  → Confident commitments (high agency)");
    console.log("  → Epistemic integrity (high coverage)");
  } else if (wellCalibratedAccuracy && !wellCalibratedAgency) {
    console.log("⚠ UNDERCONFIDENT");
    console.log("  → Accurate but lacks confidence");
    console.log("  → Could commit to narrower cones");
  } else if (!wellCalibratedAccuracy && wellCalibratedAgency) {
    console.log("✗ OVERCONFIDENT");
    console.log("  → Confident but inaccurate");
    console.log("  → Dangerous: Confidently wrong!");
  } else {
    console.log("⚠ NEEDS MORE TRAINING");
  }

  console.log("\n" + "═".repeat(61));
  console.log("  Key Insights");
  console.log("═".repeat(61) + "\n");

  console.log("Agency ≠ Accuracy (but they should correlate!):\n");

  console.log("• AGENCY: How confident is the actor?");
  console.log("  Measured by cone volume");
  console.log("  High agency = narrow cones = strong commitments\n");

  console.log("• ACCURACY: How correct are predictions?");
  console.log("  Measured by prediction error");
  console.log("  Low error = good world model\n");

  console.log("• BINDING: Does reality honor commitments?");
  console.log("  Measured by coverage rate");
  console.log("  High coverage = trajectories fall inside cones\n");

  console.log("Well-calibrated model:");
  console.log("  1. High agency correlates with low error");
  console.log("     (Confident predictions are accurate)");
  console.log("  2. High agency correlates with high binding");
  console.log("     (Narrow cones still contain trajectories)");
  console.log("  3. Together: Confidence matches reality");
  console.log("     → Epistemic integrity\n");

  console.log("This experiment shows:");
  if (goodTestCorr) {
    console.log("  ✓ Test calibration: Agency predicts accuracy");
  } else {
    console.log("  ✗ Test calibration: Agency doesn't predict accuracy well");
  }

  if (goodTrainCorr) {
    console.log("  ✓ Train calibration: Agency predicts binding");
  } else {
    console.log("  ⚠ Train calibration: Weak agency-binding correlation");
  }

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
