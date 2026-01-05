/**
 * Experiment 05b: Emergent Composition via Port Import
 *
 * Demonstrates natural selection approach to composition:
 * - Primitives are imported as CANDIDATE specialists (not forced)
 * - Agency-based selection and proliferation handle adaptation
 * - Bad primitives get sidelined, new specialists emerge for novel dynamics
 *
 * Contrast with 05-composition.ts (imperative composition):
 * - 05: Forces all samples through functor → negative transfer
 * - 05b: Imports ports as candidates → natural selection
 *
 * Domain: Same piecewise potential (attractive wells + repulsive gap)
 *
 * Expected Outcome:
 * - Bootstrap advantage from imported primitives (55% initial coverage)
 * - Proliferation creates specialists for gap dynamics
 * - Imported primitives naturally sidelined when they fail
 * - Should beat baseline (bootstrap) WITHOUT negative transfer
 */

import { GeometricPortBank } from "../src/geometric/bank";
import type { Encoders, Situation } from "../src/types";
import { sub } from "../src/vec";

// ============================================================================
// Piecewise Potential with Repulsive Gap (Same as 05)
// ============================================================================

const dt = 0.1;

type State = { x: number; v: number };

function piecewisePotential(state: State): State {
  const { x, v } = state;
  let ax: number;

  if (x < -0.5) {
    const k = 2.0;
    const b = 0.2;
    const equilibrium = -1.0;
    ax = -k * (x - equilibrium) - b * v;
  } else if (x > 0.5) {
    const k = 2.0;
    const b = 0.2;
    const equilibrium = 1.0;
    ax = -k * (x - equilibrium) - b * v;
  } else {
    // Gap: REPULSIVE
    const k = 1.5;
    const b = 0.3;
    ax = k * x - b * v;
  }

  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

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

const encoders: Encoders<State, {}> = {
  embedSituation: (sit: Situation<State, {}>) => [sit.state.x, sit.state.v],
  delta: (before, after) => sub([after.state.x, after.state.v], [before.state.x, before.state.v]),
};

// ============================================================================
// Evaluation
// ============================================================================

interface RegionMetrics {
  error: number;
  agency: number;
  coverage: number;
}

function evaluateRegion(
  bank: GeometricPortBank<State, {}>,
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

    const pred = bank.predictFromState("default", { state, context: {} }, 1)[0];
    if (!pred) continue;

    const predDelta = pred.delta;

    const error = Math.sqrt(
      (predDelta[0]! - truthDelta[0]!) ** 2 +
        (predDelta[1]! - truthDelta[1]!) ** 2
    );

    const agency = pred.agency;

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
  console.log("  Experiment 05b: Emergent Composition");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Approach: Import primitive ports as CANDIDATES");
  console.log("  → Natural selection via agency-based port selection");
  console.log("  → Proliferation creates specialists when primitives fail");
  console.log("  → No forced composition through functors\n");

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

  console.log("Training left well specialist...");
  const leftBank = new GeometricPortBank<State, {}>(encoders);
  for (let i = 0; i < 300; i++) {
    const before = randomState(-2, -0.5);
    const after = leftWellDynamics(before);
    await leftBank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      action: "default",
    });
  }
  leftBank.flush();
  console.log(`✓ Left specialist trained (${leftBank.getPortCount()} ports)\n`);

  console.log("Training right well specialist...");
  const rightBank = new GeometricPortBank<State, {}>(encoders);
  for (let i = 0; i < 300; i++) {
    const before = randomState(0.5, 2);
    const after = rightWellDynamics(before);
    await rightBank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      action: "default",
    });
  }
  rightBank.flush();
  console.log(`✓ Right specialist trained (${rightBank.getPortCount()} ports)\n`);

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 2: Create Composed Bank with Imported Primitives
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 2: Import Primitive Ports as Candidates");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const emergentBank = new GeometricPortBank<State, {}>(encoders, {
    enableProliferation: true, // Key: allow new specialists to emerge
    minAlignmentThreshold: 0.0,
  });

  // Import ports from primitives
  const leftPorts = leftBank.exportPorts();
  const rightPorts = rightBank.exportPorts();

  for (const portData of leftPorts) {
    emergentBank.importPort({ action: portData.action, embedding: portData.embedding });
  }
  for (const portData of rightPorts) {
    emergentBank.importPort({ action: portData.action, embedding: portData.embedding });
  }

  console.log(`✓ Imported ${leftPorts.length + rightPorts.length} primitive ports as candidates\n`);

  // Evaluate initial performance (before any gap training)
  const initialLeft = evaluateRegion(emergentBank, testStates, -2, -0.5);
  const initialGap = evaluateRegion(emergentBank, testStates, -0.5, 0.5);
  const initialRight = evaluateRegion(emergentBank, testStates, 0.5, 2);

  console.log("Initial Performance (Imported Primitives, No Gap Training):");
  console.log(
    `  Left:  Coverage=${(initialLeft.coverage * 100).toFixed(1)}%, Agency=${(initialLeft.agency * 100).toFixed(1)}%`
  );
  console.log(
    `  Gap:   Coverage=${(initialGap.coverage * 100).toFixed(1)}%, Agency=${(initialGap.agency * 100).toFixed(1)}%`
  );
  console.log(
    `  Right: Coverage=${(initialRight.coverage * 100).toFixed(1)}%, Agency=${(initialRight.agency * 100).toFixed(1)}%\n`
  );

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 3a: Baseline (From Scratch)
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 3a: Baseline - Train from Scratch");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const baselineBank = new GeometricPortBank<State, {}>(encoders, {
    enableProliferation: true,
  });

  console.log("Training baseline on gap samples...");
  console.log("Sample | Baseline Gap Cov | Ports");
  console.log("-------|------------------|------");

  interface Checkpoint {
    sample: number;
    gap: RegionMetrics;
    portCount: number;
  }

  const baselineCheckpoints: Checkpoint[] = [];
  const gapSamples = 500;
  const checkpointEvery = 100;

  for (let i = 0; i < gapSamples; i++) {
    const before = randomState(-0.5, 0.5);
    const after = piecewisePotential(before);
    await baselineBank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      action: "default",
    });

    if ((i + 1) % checkpointEvery === 0) {
      baselineBank.flush();
      const gap = evaluateRegion(baselineBank, testStates, -0.5, 0.5);
      baselineCheckpoints.push({
        sample: i + 1,
        gap,
        portCount: baselineBank.getPortCount(),
      });

      console.log(
        `${(i + 1).toString().padStart(6)} | ${(gap.coverage * 100).toFixed(1).padStart(15)}% | ${baselineBank.getPortCount()}`
      );
    }
  }

  console.log();

  // ──────────────────────────────────────────────────────────────────────────
  // Phase 3b: Emergent Composition (Import + Online Learning)
  // ──────────────────────────────────────────────────────────────────────────

  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Phase 3b: Emergent - Train with Imported Primitives");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  console.log("Training emergent model on gap samples...");
  console.log("Sample | Emergent Gap Cov | Ports");
  console.log("-------|------------------|------");

  const emergentCheckpoints: Checkpoint[] = [
    {
      sample: 0,
      gap: initialGap,
      portCount: emergentBank.getPortCount(),
    },
  ];

  for (let i = 0; i < gapSamples; i++) {
    const before = randomState(-0.5, 0.5);
    const after = piecewisePotential(before);
    await emergentBank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      action: "default",
    });

    if ((i + 1) % checkpointEvery === 0) {
      emergentBank.flush();
      const gap = evaluateRegion(emergentBank, testStates, -0.5, 0.5);
      emergentCheckpoints.push({
        sample: i + 1,
        gap,
        portCount: emergentBank.getPortCount(),
      });

      console.log(
        `${(i + 1).toString().padStart(6)} | ${(gap.coverage * 100).toFixed(1).padStart(15)}% | ${emergentBank.getPortCount()}`
      );
    }
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Final Results: Baseline vs Emergent
  // ──────────────────────────────────────────────────────────────────────────

  console.log("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
  console.log("Final Results: Baseline vs Emergent Composition");
  console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

  const finalBaseline = baselineCheckpoints[baselineCheckpoints.length - 1]!;
  const finalEmergent = emergentCheckpoints[emergentCheckpoints.length - 1]!;

  console.log("Baseline (From Scratch):");
  console.log(`  Initial: 0.0% coverage, 0 ports`);
  console.log(`  Final:   ${(finalBaseline.gap.coverage * 100).toFixed(1)}% coverage, ${finalBaseline.portCount} ports`);
  console.log(`  Error:   ${finalBaseline.gap.error.toFixed(4)}\n`);

  console.log("Emergent (Imported Primitives + Natural Selection):");
  console.log(`  Initial: ${(initialGap.coverage * 100).toFixed(1)}% coverage, ${emergentCheckpoints[0]!.portCount} ports (imported)`);
  console.log(`  Final:   ${(finalEmergent.gap.coverage * 100).toFixed(1)}% coverage, ${finalEmergent.portCount} ports`);
  console.log(`  Error:   ${finalEmergent.gap.error.toFixed(4)}\n`);

  console.log("Learning Trajectories (Gap Coverage):");
  console.log("Sample | Baseline | Emergent | Advantage");
  console.log("-------|----------|----------|----------");
  console.log(
    `     0 |     0.0% | ${(initialGap.coverage * 100).toFixed(1).padStart(7)}% | +${(initialGap.coverage * 100).toFixed(1)}%`
  );

  for (let i = 0; i < baselineCheckpoints.length; i++) {
    const baseline = baselineCheckpoints[i]!;
    const emergent = emergentCheckpoints[i + 1]!;
    const advantage = emergent.gap.coverage - baseline.gap.coverage;
    const sign = advantage >= 0 ? "+" : "";

    console.log(
      `${baseline.sample.toString().padStart(6)} | ` +
        `${(baseline.gap.coverage * 100).toFixed(1).padStart(7)}% | ` +
        `${(emergent.gap.coverage * 100).toFixed(1).padStart(7)}% | ` +
        `${sign}${(advantage * 100).toFixed(1)}%`
    );
  }

  console.log();

  // Analysis
  const bootstrapAdvantage = initialGap.coverage;
  const finalAdvantage = finalEmergent.gap.coverage - finalBaseline.gap.coverage;

  if (finalAdvantage > 0.05) {
    console.log(`✓ Emergent composition provides ${(finalAdvantage * 100).toFixed(1)}% advantage`);
    console.log(`✓ Bootstrap gave ${(bootstrapAdvantage * 100).toFixed(1)}% initial coverage`);
    console.log("✓ Natural selection avoided negative transfer!");
  } else if (finalAdvantage > -0.05) {
    console.log(`~ Emergent and baseline converge (diff: ${(finalAdvantage * 100).toFixed(1)}%)`);
    console.log(`~ Bootstrap helped initially (+${(bootstrapAdvantage * 100).toFixed(1)}%)`);
    console.log("~ Natural selection prevented harm but no net benefit");
  } else {
    console.log(`✗ Baseline outperforms emergent by ${(Math.abs(finalAdvantage) * 100).toFixed(1)}%`);
    console.log("✗ Imported primitives still caused negative transfer");
  }

  // Port evolution analysis
  const importedPorts = leftPorts.length + rightPorts.length;
  const newPorts = finalEmergent.portCount - importedPorts;

  console.log(`\nPort Evolution:`);
  console.log(`  Started with: ${importedPorts} imported ports`);
  console.log(`  Proliferated: ${newPorts > 0 ? newPorts : 0} new specialists`);
  console.log(`  Final count:  ${finalEmergent.portCount} ports`);

  if (newPorts > 0) {
    console.log(`  → System discovered need for ${newPorts} new specialists`);
  }

  // Export results
  await Bun.write(
    "examples/results/05b-emergent-composition.json",
    JSON.stringify(
      {
        name: "Emergent Composition vs Baseline",
        approach: "Import ports as candidates, natural selection via agency",
        config: {
          domain: "Piecewise Potential (Attractive Wells + Repulsive Gap)",
          gapSamples,
          importedPorts,
          proliferationEnabled: true,
        },
        baseline: {
          checkpoints: baselineCheckpoints,
          final: {
            coverage: finalBaseline.gap.coverage,
            error: finalBaseline.gap.error,
            ports: finalBaseline.portCount,
          },
        },
        emergent: {
          checkpoints: emergentCheckpoints,
          initial: {
            coverage: initialGap.coverage,
            ports: importedPorts,
          },
          final: {
            coverage: finalEmergent.gap.coverage,
            error: finalEmergent.gap.error,
            ports: finalEmergent.portCount,
            newPorts,
          },
        },
        comparison: {
          bootstrapAdvantage,
          finalAdvantage,
          emergentWins: finalAdvantage > 0.05,
        },
        timestamp: new Date().toISOString(),
      },
      null,
      2
    )
  );

  console.log("\n✓ Results saved to examples/results/05b-emergent-composition.json");

  leftBank.dispose();
  rightBank.dispose();
  baselineBank.dispose();
  emergentBank.dispose();
  console.log("✓ Experiment complete!");
}

main().catch(console.error);
