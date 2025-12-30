/**
 * Calibration Analysis: Agency vs Accuracy vs Binding
 *
 * Uses the bank's built-in calibration tracking to properly measure:
 * 1. Prediction Error: ||predicted - actual||
 * 2. Agency: Cone volume (confidence)
 * 3. Coverage Rate: % of trajectories inside cones (true binding rate)
 * 4. Calibration: Does high agency correlate with low error AND high coverage?
 *
 * Key insight: A well-calibrated model should have:
 * - High agency → Low error (confidently correct)
 * - High agency → High coverage (commitments honored)
 * - Low error + High coverage → High agency (epistemic integrity)
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
// Domain: 1D Damped Spring (same as predictive-accuracy.ts)
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
// Training with Calibration Tracking
// ============================================================================

async function trainAndEvaluate(
  episodes: number,
  xRange: [number, number],
  vRange: [number, number]
): Promise<{
  bank: GeometricPortBank<SpringState, unknown>;
  testError: number;
  calibration: Record<string, ReturnType<typeof bank.getCalibrationDiagnostics>>;
}> {
  const bridge = createEncoderBridge<SpringState>({
    extractRaw: embedState,
  });

  const config: GeometricPortConfigInput = {
    embeddingDim: 2,
  };

  const bank = new GeometricPortBank(bridge.encoders, config);

  // Training
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

  // Test on held-out data
  const testStates = Array.from({ length: 100 }, () => randomState(xRange, vRange));
  let totalError = 0;

  for (const state of testStates) {
    const sit: Situation<SpringState, unknown> = { state, context: null };
    const predictions = bank.predict("step", sit, 1);
    const pred = predictions[0];
    if (!pred) continue;

    const actualNext = simulate(state);
    const actualTrajectory = [actualNext.x - state.x, actualNext.v - state.v];
    const predError = norm(sub(pred.delta, actualTrajectory));
    totalError += predError;
  }

  const meanError = totalError / testStates.length;

  // Get calibration from bank (uses actual cone checks!)
  const calibration = bank.getCalibrationDiagnostics();

  return { bank, testError: meanError, calibration };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Calibration Analysis: Agency × Accuracy × Binding");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Question: Does the model grok? Is it well-calibrated?\n");
  console.log("Metrics:");
  console.log("  • Prediction Error: How accurate are predictions?");
  console.log("  • Coverage Rate: % of actual trajectories inside predicted cones");
  console.log("  • Agency: How confident is the actor? (1 - cone_volume)");
  console.log("  • Calibration: High agency → Low error + High coverage?\n");

  // Training curve with proper calibration tracking
  const checkpoints = [100, 200, 500, 1000];
  const results: Array<{
    episodes: number;
    error: number;
    avgCoverage: number;
    avgAgency: number;
    avgCorrelation: number;
  }> = [];

  for (const ep of checkpoints) {
    console.log(`\n─────────────────────────────────────────────────────────`);
    console.log(`Training with ${ep} episodes...`);

    const { bank, testError, calibration } = await trainAndEvaluate(ep, [-1, 1], [-1, 1]);

    // Aggregate calibration across all ports
    const portIds = Object.keys(calibration);
    let totalCoverage = 0;
    let totalCorrelation = 0;
    let totalAgency = 0;
    let portCount = 0;

    console.log(`\nPer-Port Calibration (${portIds.length} ports):`);

    for (const portId of portIds) {
      const cal = calibration[portId]!;

      // Get average agency from calibration buckets
      let avgAgency = 0;
      let totalSamples = 0;
      for (const bucket of cal.calibrationBuckets) {
        // Parse agency range "[0.8-0.9]" to get midpoint
        const match = bucket.agencyRange.match(/\[([\d.]+)-([\d.]+)\]/);
        if (match) {
          const mid = (parseFloat(match[1]!) + parseFloat(match[2]!)) / 2;
          avgAgency += mid * bucket.count;
          totalSamples += bucket.count;
        }
      }
      avgAgency = totalSamples > 0 ? avgAgency / totalSamples : 0;

      totalCoverage += cal.coverageRate;
      totalCorrelation += cal.agencyBindingCorrelation;
      totalAgency += avgAgency;
      portCount++;

      console.log(`  Port ${portId.slice(0, 8)}...:`);
      console.log(`    Coverage: ${(cal.coverageRate * 100).toFixed(1)}%`);
      console.log(`    Avg Agency: ${(avgAgency * 100).toFixed(1)}%`);
      console.log(`    Correlation: ${cal.agencyBindingCorrelation.toFixed(3)}`);
    }

    const avgCoverage = totalCoverage / portCount;
    const avgCorrelation = totalCorrelation / portCount;
    const avgAgency = totalAgency / portCount;

    console.log(`\nAggregate Metrics:`);
    console.log(`  Test Error: ${testError.toFixed(4)}`);
    console.log(`  Avg Coverage Rate: ${(avgCoverage * 100).toFixed(1)}%`);
    console.log(`  Avg Agency: ${(avgAgency * 100).toFixed(1)}%`);
    console.log(`  Avg Calibration Correlation: ${avgCorrelation.toFixed(3)}`);

    results.push({
      episodes: ep,
      error: testError,
      avgCoverage,
      avgAgency,
      avgCorrelation,
    });

    bank.dispose();
  }

  // ============================================================================
  // Analysis
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis: Training Dynamics");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Check for grokking (sudden error drop)
  for (let i = 1; i < results.length; i++) {
    const prev = results[i - 1]!;
    const curr = results[i]!;
    const errorDrop = prev.error - curr.error;
    const errorDropPct = (errorDrop / prev.error) * 100;

    if (errorDropPct > 30) {
      console.log(`✨ GROKKING detected between ${prev.episodes} → ${curr.episodes} episodes:`);
      console.log(`   Test error: ${prev.error.toFixed(4)} → ${curr.error.toFixed(4)} (${errorDropPct.toFixed(0)}% improvement)`);
      console.log(`   Coverage: ${(prev.avgCoverage * 100).toFixed(1)}% → ${(curr.avgCoverage * 100).toFixed(1)}%`);
      console.log(`   Agency: ${(prev.avgAgency * 100).toFixed(1)}% → ${(curr.avgAgency * 100).toFixed(1)}%`);
      console.log();
    }
  }

  // Check final state calibration
  const final = results[results.length - 1]!;
  console.log("Final State Calibration:");

  const wellCalibratedError = final.error < 0.05;
  const wellCalibratedCoverage = final.avgCoverage > 0.9;
  const wellCalibratedAgency = final.avgAgency > 0.8;
  const wellCalibratedCorr = final.avgCorrelation > 0.5;

  console.log(`  Prediction Error: ${final.error.toFixed(4)} ${wellCalibratedError ? "✓ Low" : "✗ High"}`);
  console.log(`  Coverage Rate: ${(final.avgCoverage * 100).toFixed(1)}% ${wellCalibratedCoverage ? "✓ High" : "✗ Low"}`);
  console.log(`  Agency: ${(final.avgAgency * 100).toFixed(1)}% ${wellCalibratedAgency ? "✓ High" : "✗ Low"}`);
  console.log(`  Calibration Correlation: ${final.avgCorrelation.toFixed(3)} ${wellCalibratedCorr ? "✓ Good" : "⚠ Weak"}`);

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Insights");
  console.log("═══════════════════════════════════════════════════════════\n");

  if (wellCalibratedError && wellCalibratedCoverage && wellCalibratedAgency) {
    console.log("✓ WELL-CALIBRATED: High accuracy + High coverage + High agency");
    console.log("  → Actor makes accurate predictions");
    console.log("  → Actor commits to narrow cones (high confidence)");
    console.log("  → Reality honors those commitments (epistemic integrity)");
  } else if (wellCalibratedError && !wellCalibratedAgency) {
    console.log("⚠ UNDERCONFIDENT: Accurate but lacks agency");
    console.log("  → Predictions are correct");
    console.log("  → But cones are too wide (not confident enough)");
    console.log("  → Conservative, safe, but uninformative");
  } else if (!wellCalibratedError && wellCalibratedAgency) {
    console.log("✗ OVERCONFIDENT: High agency but inaccurate");
    console.log("  → Actor commits to narrow cones (confident)");
    console.log("  → But predictions are wrong (dangerous!)");
    console.log("  → Confidently incorrect = epistemic failure");
  } else if (!wellCalibratedCoverage) {
    console.log("✗ MISCALIBRATED: Poor coverage rate");
    console.log("  → Actual trajectories fall outside predicted cones");
    console.log("  → Actor violates its own commitments");
    console.log("  → Binding failures indicate epistemic integrity failure");
  } else {
    console.log("⚠ NEEDS MORE TRAINING");
    console.log("  → Model hasn't converged yet");
    console.log("  → Continue training to see if it groks");
  }

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Three-Way Relationship");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("1. ACCURACY (Prediction Error)");
  console.log("   Measures: How correct are the predictions?");
  console.log("   Low error = Good world model\n");

  console.log("2. AGENCY (Commitment Strength)");
  console.log("   Measures: How confident is the actor?");
  console.log("   High agency = Narrow cones = Strong commitments\n");

  console.log("3. BINDING (Epistemic Integrity)");
  console.log("   Measures: Does reality honor the commitments?");
  console.log("   High coverage = Trajectories fall inside cones\n");

  console.log("Well-Calibrated Model:");
  console.log("  • Makes accurate predictions (low error)");
  console.log("  • Commits confidently (high agency / narrow cones)");
  console.log("  • Reality validates commitments (high coverage)");
  console.log("  → This is epistemic integrity: Claim matches reality\n");

  console.log("Miscalibrated Model:");
  console.log("  • High agency + low coverage = Overconfident");
  console.log("    (Makes strong claims that reality violates)");
  console.log("  • Low agency + low error = Underconfident");
  console.log("    (Correct but timid, cones too wide)");
  console.log("  • High error + high agency = Dangerous");
  console.log("    (Confidently wrong, epistemic failure)");

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
