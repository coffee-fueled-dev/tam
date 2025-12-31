/**
 * Experiment 05: Discontinuous Composition with Online Adaptation
 *
 * Demonstrates TAM's composition API for bridging discontinuous dynamics.
 * Primitives bootstrap knowledge, online learning adapts to novel regions.
 *
 * Domain: Piecewise Potential Wells with Repulsive Gap
 * - Left region (x < -0.5):  Attractive well at x=-1 (oscillates inward)
 * - Right region (x > 0.5):  Attractive well at x=+1 (oscillates inward)
 * - Gap region [-0.5, 0.5]: REPULSIVE (pushes away from origin!)
 *
 * Key Challenge:
 * - Primitives learn attractive dynamics (restoring force)
 * - Gap has OPPOSITE dynamics (repulsive force)
 * - Simple interpolation/extrapolation FAILS (wrong sign!)
 *
 * Solution:
 * - Train primitive specialists on left/right wells
 * - Discover functor between primitives (coordinate transformation)
 * - Use ComposedPort.observe() to adapt online in gap region
 *
 * Expected Outcome:
 * - Functor discovery succeeds (both wells are oscillators)
 * - Initial gap predictions poor (wrong dynamics)
 * - Online learning adapts to repulsive gap
 * - Composed model learns faster than baseline (bootstrap advantage)
 */

import { TAM } from "../src/composition";
import type { Vec } from "../src/vec";
import { GeometricPortBank } from "../src/geometric/bank";
import { sub } from "../src/vec";

// ============================================================================
// Piecewise Potential with Repulsive Gap
// ============================================================================

const dt = 0.1;

type State = { x: number; v: number };

/**
 * Piecewise dynamics: Attractive wells + Repulsive gap.
 */
function piecewisePotential(state: State): State {
  const { x, v } = state;
  let ax: number;

  if (x < -0.5) {
    // Left well: Attractive force toward x=-1
    const k = 2.0;
    const b = 0.2;
    const equilibrium = -1.0;
    ax = -k * (x - equilibrium) - b * v;
  } else if (x > 0.5) {
    // Right well: Attractive force toward x=+1
    const k = 2.0;
    const b = 0.2;
    const equilibrium = 1.0;
    ax = -k * (x - equilibrium) - b * v;
  } else {
    // Gap: REPULSIVE force (pushes away from origin!)
    const k = 1.5;
    const b = 0.3;
    ax = k * x - b * v; // Note: +k (not -k) = repulsion
  }

  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

/**
 * Left well dynamics (for primitive training).
 */
function leftWellDynamics(state: State): State {
  const { x, v } = state;
  const k = 2.0;
  const b = 0.2;
  const equilibrium = -1.0;
  const ax = -k * (x - equilibrium) - b * v;
  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

/**
 * Right well dynamics (for primitive training).
 */
function rightWellDynamics(state: State): State {
  const { x, v } = state;
  const k = 2.0;
  const b = 0.2;
  const equilibrium = 1.0;
  const ax = -k * (x - equilibrium) - b * v;
  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

function randomState(xMin: number, xMax: number): State {
  return {
    x: xMin + Math.random() * (xMax - xMin),
    v: (Math.random() - 0.5) * 2,
  };
}

const embedder = (s: State): Vec => [s.x, s.v];

// ============================================================================
// Evaluation
// ============================================================================

interface RegionMetrics {
  error: number;
  agency: number;
  coverage: number;
}

function evaluateRegion(
  composed: any,
  states: State[],
  xMin: number,
  xMax: number,
  tolerance: number = 0.15
): RegionMetrics {
  const regionStates = states.filter((s) => s.x >= xMin && s.x < xMax);
  if (regionStates.length === 0) {
    return { error: 1.0, agency: 0, coverage: 0 };
  }

  let totalError = 0;
  let totalAgency = 0;
  let boundCount = 0;

  for (const state of regionStates) {
    const truth = piecewisePotential(state);
    const truthDelta = [truth.x - state.x, truth.v - state.v];

    const pred = composed.predict(state);
    const predDelta = pred.delta;

    const error = Math.sqrt(
      (predDelta[0]! - truthDelta[0]!) ** 2 +
        (predDelta[1]! - truthDelta[1]!) ** 2
    );

    // Agency from binding rate (0-1)
    const agency = pred.composedBindingRate;

    totalError += error;
    totalAgency += agency;
    if (error < tolerance) boundCount++;
  }

  return {
    error: totalError / regionStates.length,
    agency: totalAgency / regionStates.length,
    coverage: boundCount / regionStates.length,
  };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Experiment 05: Discontinuous Composition");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Domain: Piecewise Potential Wells");
  console.log("  Left:  x ∈ [-2, -0.5], ATTRACTIVE (k=-2.0, equilibrium=-1)");
  console.log("  Right: x ∈ [0.5, 2],   ATTRACTIVE (k=-2.0, equilibrium=+1)");
  console.log("  Gap:   x ∈ [-0.5, 0.5], REPULSIVE (k=+1.5, origin)\n");

  console.log("Challenge:");
  console.log("  Primitives learn attractive dynamics (wrong for gap!)");
  console.log("  Composition must adapt online to repulsive forces.\n");

  // Test states
  const testStates: State[] = [];
  for (let i = 0; i < 100; i++) {
    testStates.push(randomState(-2, 2));
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 1: Train Primitive Specialists
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 1: Train Primitive Specialists");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const tam = new TAM({
    maxEpochs: 100,
    successThreshold: 0.7,
    learningRate: 0.01,
  });

  console.log("Training 'left' specialist (300 samples)...");
  await tam.learn(
    "left",
    {
      randomState: () => randomState(-2, -0.5),
      simulate: leftWellDynamics,
      embedder,
      embeddingDim: 2,
    },
    {
      epochs: 30,
      samplesPerEpoch: 10,
      flushFrequency: 10,
    }
  );
  console.log("✓ Left specialist trained\n");

  console.log("Training 'right' specialist (300 samples)...");
  await tam.learn(
    "right",
    {
      randomState: () => randomState(0.5, 2),
      simulate: rightWellDynamics,
      embedder,
      embeddingDim: 2,
    },
    {
      epochs: 30,
      samplesPerEpoch: 10,
      flushFrequency: 10,
    }
  );
  console.log("✓ Right specialist trained\n");

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 2: Discover Functor
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 2: Functor Discovery");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  console.log("Discovering functor: left → right...");
  const path = await tam.findPath("left", "right");

  if (!path) {
    console.log("✗ Functor discovery FAILED");
    console.log("  Primitives may be too different or need more training.");
    return;
  }

  console.log(`✓ Functor discovered: ${path.describe()}`);
  console.log(`  Binding rate: ${(path.totalBindingRate * 100).toFixed(1)}%\n`);

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 3: Initial Evaluation (Before Online Learning)
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 3: Initial Evaluation (Before Online Learning)");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const composed = tam.compose(path);

  const leftInitial = evaluateRegion(composed, testStates, -2, -0.5);
  const gapInitial = evaluateRegion(composed, testStates, -0.5, 0.5);
  const rightInitial = evaluateRegion(composed, testStates, 0.5, 2);

  console.log("Initial Performance (Composed Port, No Online Learning):");
  console.log(
    `  Left:  Error=${leftInitial.error.toFixed(4)}, Agency=${(leftInitial.agency * 100).toFixed(1)}%, Coverage=${(leftInitial.coverage * 100).toFixed(1)}%`
  );
  console.log(
    `  Gap:   Error=${gapInitial.error.toFixed(4)}, Agency=${(gapInitial.agency * 100).toFixed(1)}%, Coverage=${(gapInitial.coverage * 100).toFixed(1)}%`
  );
  console.log(
    `  Right: Error=${rightInitial.error.toFixed(4)}, Agency=${(rightInitial.agency * 100).toFixed(1)}%, Coverage=${(rightInitial.coverage * 100).toFixed(1)}%\n`
  );

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 4a: Baseline (Train from Scratch on Gap)
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 4a: Baseline - Train from Scratch on Gap");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  console.log("Training baseline model (no primitives, no composition)...");

  const gapSamples = 500;
  const checkpointEvery = 100;

  // Create baseline bank directly (no TAM wrapper needed)
  const baselineEncoders = {
    embedSituation: (sit: { state: State; context: {} }) => [sit.state.x, sit.state.v],
    delta: (before: { state: State; context: {} }, after: { state: State; context: {} }) =>
      sub([after.state.x, after.state.v], [before.state.x, before.state.v]),
  };

  const baselineBank = new GeometricPortBank<State, {}>(baselineEncoders);

  // Train directly on gap region (no bootstrap)
  console.log("Training baseline on gap samples...");
  console.log("Sample | Baseline Gap Cov | Gap Error");
  console.log("-------|------------------|----------");

  interface BaselineCheckpoint {
    sample: number;
    gap: RegionMetrics;
  }

  const baselineCheckpoints: BaselineCheckpoint[] = [];

  // Train in batches so we can checkpoint
  const baselineBatchSize = 100;
  for (let batch = 0; batch < gapSamples / baselineBatchSize; batch++) {
    // Train batch
    for (let i = 0; i < baselineBatchSize; i++) {
      const before = randomState(-0.5, 0.5);
      const after = piecewisePotential(before);
      await baselineBank.observe({
        before: { state: before, context: {} },
        after: { state: after, context: {} },
        action: "default",
      });
    }
    baselineBank.flush();

    const sample = (batch + 1) * baselineBatchSize;

    // Evaluate baseline
    const baselineGap = evaluateRegion(
      { predict: (s: State) => {
        const pred = baselineBank.predictFromState("default", { state: s, context: {} }, 1)[0];
        if (!pred) return { delta: [0, 0], composedBindingRate: 0 };
        return {
          delta: pred.delta,
          composedBindingRate: pred.agency || 0
        };
      }},
      testStates,
      -0.5,
      0.5
    );

    baselineCheckpoints.push({ sample, gap: baselineGap });

    console.log(
      `${sample.toString().padStart(6)} | ${(baselineGap.coverage * 100).toFixed(1).padStart(15)}% | ${baselineGap.error.toFixed(4)}`
    );
  }

  console.log();

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 4b: Composed Model (Bootstrap + Online Learning)
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 4b: Composed Model - Bootstrap + Online Learning");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  console.log("Training composed model on gap samples...");
  console.log("Sample | Composed Gap Cov | Gap Error");
  console.log("-------|------------------|----------");

  interface Checkpoint {
    sample: number;
    gap: RegionMetrics;
    left: RegionMetrics;
    right: RegionMetrics;
  }

  const checkpoints: Checkpoint[] = [
    {
      sample: 0,
      gap: gapInitial,
      left: leftInitial,
      right: rightInitial,
    },
  ];

  for (let i = 0; i < gapSamples; i++) {
    // Sample from full range (but mostly gap)
    const before = randomState(-0.5, 0.5);
    const after = piecewisePotential(before);

    await composed.observe({ before, after });

    if ((i + 1) % checkpointEvery === 0) {
      const left = evaluateRegion(composed, testStates, -2, -0.5);
      const gap = evaluateRegion(composed, testStates, -0.5, 0.5);
      const right = evaluateRegion(composed, testStates, 0.5, 2);

      checkpoints.push({ sample: i + 1, gap, left, right });

      console.log(
        `${(i + 1).toString().padStart(6)} | ${(gap.coverage * 100).toFixed(1).padStart(15)}% | ${gap.error.toFixed(4)}`
      );
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Final Results: Baseline vs Composed
  // ──────────────────────────────────────────────────────────────────────────

  console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Final Results: Baseline vs Composed");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const finalComposed = checkpoints[checkpoints.length - 1]!;
  const finalBaseline = baselineCheckpoints[baselineCheckpoints.length - 1]!;

  console.log("Baseline (From Scratch):");
  console.log(`  Initial: 0.0% coverage`);
  console.log(`  Final:   ${(finalBaseline.gap.coverage * 100).toFixed(1)}% coverage`);
  console.log(`  Error:   ${finalBaseline.gap.error.toFixed(4)}\n`);

  console.log("Composed (Bootstrap + Online):");
  console.log(`  Initial: ${(gapInitial.coverage * 100).toFixed(1)}% coverage (from primitives)`);
  console.log(`  Final:   ${(finalComposed.gap.coverage * 100).toFixed(1)}% coverage`);
  console.log(`  Error:   ${finalComposed.gap.error.toFixed(4)}\n`);

  console.log("Learning Trajectories (Gap Coverage):");
  console.log("Sample | Baseline | Composed | Advantage");
  console.log("-------|----------|----------|----------");
  console.log(`     0 |     0.0% | ${(gapInitial.coverage * 100).toFixed(1).padStart(7)}% | +${(gapInitial.coverage * 100).toFixed(1)}%`);

  for (let i = 0; i < baselineCheckpoints.length; i++) {
    const baseline = baselineCheckpoints[i]!;
    const composed = checkpoints[i + 1]!; // +1 because composed has initial checkpoint
    const advantage = composed.gap.coverage - baseline.gap.coverage;
    const sign = advantage >= 0 ? "+" : "";

    console.log(
      `${baseline.sample.toString().padStart(6)} | ` +
      `${(baseline.gap.coverage * 100).toFixed(1).padStart(7)}% | ` +
      `${(composed.gap.coverage * 100).toFixed(1).padStart(7)}% | ` +
      `${sign}${(advantage * 100).toFixed(1)}%`
    );
  }

  console.log();

  // Compute acceleration metric
  const compositionAdvantage = finalComposed.gap.coverage - finalBaseline.gap.coverage;
  const bootstrapAdvantage = gapInitial.coverage; // Starting advantage from primitives

  if (compositionAdvantage > 0.1) {
    console.log(`✓ Composition provides ${(compositionAdvantage * 100).toFixed(1)}% coverage advantage`);
    console.log(`✓ Bootstrap gave ${(bootstrapAdvantage * 100).toFixed(1)}% initial coverage`);
    console.log("✓ Discontinuous composition accelerates learning!");
  } else if (compositionAdvantage > -0.1) {
    console.log(`~ Composition and baseline converge to similar performance`);
    console.log(`~ Bootstrap helped initially (+${(bootstrapAdvantage * 100).toFixed(1)}%) but gap closed`);
    console.log("~ Composition effect: marginal");
  } else {
    console.log(`✗ Baseline outperforms composition by ${(Math.abs(compositionAdvantage) * 100).toFixed(1)}%`);
    console.log("✗ Primitives may have introduced negative transfer");
  }

  // Export results
  await Bun.write(
    "examples/results/05-composition.json",
    JSON.stringify(
      {
        name: "Discontinuous Composition vs Baseline",
        config: {
          domain: "Piecewise Potential (Attractive Wells + Repulsive Gap)",
          primitives: ["left", "right"],
          functorBindingRate: path.totalBindingRate,
          gapSamples,
        },
        baseline: {
          checkpoints: baselineCheckpoints,
          final: {
            coverage: finalBaseline.gap.coverage,
            error: finalBaseline.gap.error,
          },
        },
        composed: {
          checkpoints,
          initial: {
            coverage: gapInitial.coverage,
            error: gapInitial.error,
          },
          final: {
            coverage: finalComposed.gap.coverage,
            error: finalComposed.gap.error,
          },
        },
        comparison: {
          bootstrapAdvantage,
          finalAdvantage: compositionAdvantage,
          compositionHelps: compositionAdvantage > 0.1,
        },
        timestamp: new Date().toISOString(),
      },
      null,
      2
    )
  );

  console.log("\n✓ Results saved to examples/results/05-composition.json");

  tam.dispose();
  baselineBank.dispose();
  console.log("✓ Experiment complete!");
}

main().catch(console.error);
