/**
 * Predictive Accuracy & Generalization Experiment
 *
 * Tests the critical distinction: Agency ≠ Accuracy
 * - Agency: How confident is the actor? (cone volume)
 * - Accuracy: How correct is the prediction? (prediction error)
 * - Generalization: Can it predict outside training distribution?
 * - Grokking: Does it transition from memorization to understanding?
 *
 * Test domain: 1D spring dynamics (simple, interpretable)
 * - Physics: x' = v, v' = -kx - bv (damped harmonic oscillator)
 * - Training: States with x ∈ [-1, 1], v ∈ [-1, 1]
 * - Test (in-distribution): Same range
 * - Test (out-of-distribution): x ∈ [1, 2], v ∈ [-1, 1]
 *
 * Metrics:
 * 1. Prediction Error: RMSE between predicted and actual trajectory
 * 2. Agency: Cone volume (existing metric)
 * 3. Binding Rate: Fraction of trajectories inside cones
 * 4. OOD Performance: Error on out-of-distribution states
 * 5. Calibration: Does high agency correlate with low error?
 *
 * Expected results:
 * - Good model: Low error, high agency, well-calibrated, generalizes OOD
 * - Overfit model: Low training error, high test error, poor OOD
 * - Underfit model: High error everywhere, low agency
 * - Miscalibrated model: High agency but high error (confidently wrong)
 */

import {
  GeometricPortBank,
  type Situation,
  type Transition,
  type GeometricPortConfigInput,
  createEncoderBridge,
  type EncoderBridge,
} from "../../src";
import { norm, sub } from "../../src/vec";

// ============================================================================
// Domain: 1D Damped Spring
// ============================================================================

interface SpringState {
  x: number; // Position
  v: number; // Velocity
}

const k = 1.0; // Spring constant
const b = 0.1; // Damping coefficient
const dt = 0.1; // Time step

/**
 * Simulate one step of damped spring dynamics.
 */
function simulate(state: SpringState): SpringState {
  const ax = -k * state.x - b * state.v;
  const newV = state.v + ax * dt;
  const newX = state.x + newV * dt;
  return { x: newX, v: newV };
}

/**
 * Random state in given range.
 */
function randomState(xRange: [number, number], vRange: [number, number]): SpringState {
  const x = xRange[0] + Math.random() * (xRange[1] - xRange[0]);
  const v = vRange[0] + Math.random() * (vRange[1] - vRange[0]);
  return { x, v };
}

/**
 * Embed state as [x, v] vector.
 */
function embedState(state: SpringState): number[] {
  return [state.x, state.v];
}

// ============================================================================
// Evaluation Metrics
// ============================================================================

interface EvaluationResult {
  // Prediction metrics
  meanPredictionError: number; // RMSE of trajectory predictions
  maxPredictionError: number; // Worst-case prediction error

  // Confidence metrics
  meanAgency: number; // Average agency (existing metric)

  // Calibration
  bindingRate: number; // Fraction of predictions inside cones
  errorVsAgency: Array<{ agency: number; error: number }>; // For correlation analysis

  // Summary
  wellCalibrated: boolean; // High agency correlates with low error
}

/**
 * Evaluate model on test set.
 */
function evaluate(
  bank: GeometricPortBank<SpringState, unknown>,
  testStates: SpringState[],
  label: string
): EvaluationResult {
  let totalError = 0;
  let totalAgency = 0;
  let boundCount = 0;
  let maxError = 0;
  const errorVsAgency: Array<{ agency: number; error: number }> = [];

  for (const state of testStates) {
    const sit: Situation<SpringState, unknown> = {
      state,
      context: null,
    };

    // Get prediction
    const predictions = bank.predict("step", sit, 1);
    const pred = predictions[0];
    if (!pred) continue;

    // Compute actual trajectory
    const actualNext = simulate(state);
    const actualTrajectory = [
      actualNext.x - state.x,
      actualNext.v - state.v,
    ];

    // Prediction error
    const predError = norm(sub(pred.delta, actualTrajectory));
    totalError += predError;
    maxError = Math.max(maxError, predError);

    // Agency
    totalAgency += pred.agency;

    // Check binding (is actual trajectory in cone?)
    // Note: This is approximate - we don't have direct cone access here
    // Use prediction error as proxy: if error < expected cone radius, it's bound
    const expectedConeRadius = 0.1; // Rough estimate
    if (predError < expectedConeRadius) {
      boundCount++;
    }

    // Store for calibration analysis
    errorVsAgency.push({ agency: pred.agency, error: predError });
  }

  const n = testStates.length;
  const meanError = totalError / n;
  const meanAgency = totalAgency / n;
  const bindingRate = boundCount / n;

  // Calibration: High agency should mean low error
  // Compute correlation coefficient
  const errors = errorVsAgency.map(d => d.error);
  const agencies = errorVsAgency.map(d => d.agency);
  const meanErr = errors.reduce((s, e) => s + e, 0) / n;
  const meanAg = agencies.reduce((s, a) => s + a, 0) / n;

  let correlation = 0;
  if (n > 1) {
    const covErrAg = errorVsAgency.reduce(
      (s, d) => s + (d.error - meanErr) * (d.agency - meanAg),
      0
    ) / n;
    const stdErr = Math.sqrt(
      errors.reduce((s, e) => s + (e - meanErr) ** 2, 0) / n
    );
    const stdAg = Math.sqrt(
      agencies.reduce((s, a) => s + (a - meanAg) ** 2, 0) / n
    );
    correlation = covErrAg / (stdErr * stdAg + 1e-8);
  }

  const wellCalibrated = correlation < -0.3; // Negative correlation: high agency = low error

  console.log(`\n${label}:`);
  console.log(`  Mean prediction error: ${meanError.toFixed(4)}`);
  console.log(`  Max prediction error:  ${maxError.toFixed(4)}`);
  console.log(`  Mean agency:           ${(meanAgency * 100).toFixed(1)}%`);
  console.log(`  Binding rate:          ${(bindingRate * 100).toFixed(1)}%`);
  console.log(`  Error-Agency correlation: ${correlation.toFixed(3)}`);
  console.log(`  Well calibrated:       ${wellCalibrated ? "✓ Yes" : "✗ No"}`);

  return {
    meanPredictionError: meanError,
    maxPredictionError: maxError,
    meanAgency,
    bindingRate,
    errorVsAgency,
    wellCalibrated,
  };
}

// ============================================================================
// Training
// ============================================================================

async function trainModel(
  episodes: number,
  xRange: [number, number],
  vRange: [number, number],
  verbose: boolean = false
): Promise<GeometricPortBank<SpringState, unknown>> {
  const bridge = createEncoderBridge<SpringState>({
    extractRaw: embedState,
  });

  const config: GeometricPortConfigInput = {
    embeddingDim: 2,
  };

  const bank = new GeometricPortBank(bridge.encoders, config);

  for (let i = 0; i < episodes; i++) {
    const before = randomState(xRange, vRange);
    const after = simulate(before);

    const transition: Transition<SpringState, unknown> = {
      before: { state: before, context: null },
      after: { state: after, context: null },
      action: "step",
    };

    await bank.observe(transition);

    if (verbose && (i + 1) % 100 === 0) {
      console.log(`  [${i + 1}/${episodes}] Training...`);
    }

    if ((i + 1) % 50 === 0) {
      bank.flush();
    }
  }

  bank.flush();
  return bank;
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Predictive Accuracy & Generalization Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Domain: 1D Damped Spring");
  console.log("  Dynamics: x' = v, v' = -kx - bv");
  console.log(`  Parameters: k=${k}, b=${b}, dt=${dt}`);
  console.log("  Training range: x ∈ [-1, 1], v ∈ [-1, 1]");

  // ============================================================================
  // Experiment 1: In-Distribution Performance
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Experiment 1: In-Distribution Performance");
  console.log("─────────────────────────────────────────────────────────");

  console.log("\nTraining model (300 episodes)...");
  const bank = await trainModel(300, [-1, 1], [-1, 1], true);

  // Generate test sets
  const testInDist = Array.from({ length: 100 }, () => randomState([-1, 1], [-1, 1]));
  const testOOD = Array.from({ length: 100 }, () => randomState([1, 2], [-1, 1]));

  // Evaluate
  const resultInDist = evaluate(bank, testInDist, "In-Distribution Test");
  const resultOOD = evaluate(bank, testOOD, "Out-of-Distribution Test (x ∈ [1,2])");

  // ============================================================================
  // Experiment 2: Training Curve (Grokking?)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Experiment 2: Training Curve Analysis");
  console.log("─────────────────────────────────────────────────────────");

  const checkpoints = [50, 100, 200, 300, 500, 1000];
  const trainingCurve: Array<{
    episodes: number;
    trainError: number;
    testError: number;
    agency: number;
  }> = [];

  const testSetFixed = Array.from({ length: 50 }, () => randomState([-1, 1], [-1, 1]));

  for (const ep of checkpoints) {
    console.log(`\nTraining model with ${ep} episodes...`);
    const bankCheckpoint = await trainModel(ep, [-1, 1], [-1, 1], false);

    const result = evaluate(bankCheckpoint, testSetFixed, `Checkpoint: ${ep} episodes`);
    trainingCurve.push({
      episodes: ep,
      trainError: result.meanPredictionError, // Approximate (using test as proxy)
      testError: result.meanPredictionError,
      agency: result.meanAgency,
    });

    bankCheckpoint.dispose();
  }

  // ============================================================================
  // Experiment 3: Long-Horizon Rollout
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Experiment 3: Long-Horizon Rollout");
  console.log("─────────────────────────────────────────────────────────");

  const initialState = randomState([-0.5, 0.5], [-0.5, 0.5]);
  const horizons = [1, 5, 10, 20];

  console.log("\nComparing predicted vs actual trajectories:");
  for (const h of horizons) {
    // Ground truth
    let actualState = initialState;
    for (let i = 0; i < h; i++) {
      actualState = simulate(actualState);
    }

    // Predicted (iterative)
    let predState = initialState;
    for (let i = 0; i < h; i++) {
      const sit: Situation<SpringState, unknown> = {
        state: predState,
        context: null,
      };
      const predictions = bank.predict("step", sit, 1);
      const delta = predictions[0]?.delta;
      if (!delta) break;

      predState = {
        x: predState.x + delta[0]!,
        v: predState.v + delta[1]!,
      };
    }

    const error = Math.sqrt(
      (actualState.x - predState.x) ** 2 + (actualState.v - predState.v) ** 2
    );
    console.log(`  ${h}-step horizon: error = ${error.toFixed(4)}`);
  }

  // ============================================================================
  // Analysis & Interpretation
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Check generalization
  const generalizationGap = resultOOD.meanPredictionError / resultInDist.meanPredictionError;
  console.log("Generalization:");
  if (generalizationGap < 1.5) {
    console.log("  ✓ Model generalizes well to OOD states");
    console.log(`    OOD error only ${(generalizationGap * 100).toFixed(0)}% of in-dist error`);
  } else if (generalizationGap < 3.0) {
    console.log("  ⚠ Model shows some generalization degradation");
    console.log(`    OOD error is ${generalizationGap.toFixed(1)}x in-dist error`);
  } else {
    console.log("  ✗ Model fails to generalize to OOD");
    console.log(`    OOD error is ${generalizationGap.toFixed(1)}x worse than in-dist`);
  }

  // Check calibration
  console.log("\nCalibration:");
  if (resultInDist.wellCalibrated) {
    console.log("  ✓ Model is well-calibrated");
    console.log("    High agency correlates with low error (trustworthy confidence)");
  } else {
    console.log("  ⚠ Model may be miscalibrated");
    console.log("    Agency doesn't reliably predict accuracy");
  }

  // Check for grokking
  console.log("\nTraining Dynamics:");
  const errorDecreasing = trainingCurve.every((curr, i) => {
    if (i === 0) return true;
    return curr.testError <= trainingCurve[i - 1]!.testError * 1.1; // Allow 10% noise
  });

  if (errorDecreasing) {
    console.log("  ✓ Smooth learning curve (continuous improvement)");
  } else {
    console.log("  ⚠ Non-monotonic learning (possible grokking or instability)");
  }

  // Check accuracy vs agency
  console.log("\nAccuracy vs Agency:");
  if (resultInDist.meanPredictionError < 0.1 && resultInDist.meanAgency > 0.7) {
    console.log("  ✓ Model achieves both high accuracy AND high agency");
    console.log("    (Confidently correct - ideal outcome)");
  } else if (resultInDist.meanPredictionError < 0.1) {
    console.log("  ⚠ Model is accurate but lacks confidence");
    console.log("    (Correctly uncertain - conservative)");
  } else if (resultInDist.meanAgency > 0.7) {
    console.log("  ✗ Model is confident but inaccurate");
    console.log("    (Confidently wrong - dangerous miscalibration!)");
  } else {
    console.log("  ✗ Model lacks both accuracy and confidence");
    console.log("    (Needs more training or better architecture)");
  }

  // Key insight
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Insight: Agency ≠ Accuracy");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  Agency measures: How confident is the actor?");
  console.log("  Accuracy measures: How correct are the predictions?");
  console.log("");
  console.log("  Both are necessary:");
  console.log("    - High accuracy + low agency = Underconfident (conservative)");
  console.log("    - Low accuracy + high agency = Overconfident (dangerous)");
  console.log("    - High accuracy + high agency = Well-calibrated (ideal)");
  console.log("");
  console.log("  Binding rate measures epistemic integrity:");
  console.log("    - Actor commits to cones (agency)");
  console.log("    - Reality falls inside cones (binding)");
  console.log("    - Violation = epistemic failure against own commitments");

  // Cleanup
  bank.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
