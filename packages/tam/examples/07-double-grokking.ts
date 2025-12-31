/**
 * Experiment 07: Double Grokking
 *
 * Observes the "double grokking" phenomenon:
 * 1. First grokking: Model suddenly learns dynamics (error drops)
 * 2. Second grokking: Model learns epistemics (agency/calibration improves)
 *
 * Key insight: The model first learns WHAT will happen, then learns
 * HOW CONFIDENT it should be about its predictions.
 *
 * Tracks fine-grained metrics to observe both grokking moments:
 * - Prediction error (dynamics learning)
 * - Agency (epistemic learning)
 * - Coverage/binding rate (commitment accuracy)
 * - Port count (specialization dynamics)
 */

import { GeometricPortBank } from "../src/geometric/bank";
import type { Encoders, Situation } from "../src/types";
import type { Vec } from "../src/vec";
import { sub } from "../src/vec";

// Simple 1D damped spring
const k = 1.0;
const b = 0.1;
const dt = 0.1;

function springStep(x: number, v: number): { x: number; v: number } {
  const ax = -k * x - b * v;
  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

function randomState(): { x: number; v: number } {
  return {
    x: (Math.random() - 0.5) * 2, // [-1, 1]
    v: (Math.random() - 0.5) * 2, // [-1, 1]
  };
}

const encoders: Encoders<{ x: number; v: number }, {}> = {
  embedSituation: (sit: Situation<{ x: number; v: number }, {}>) => [sit.state.x, sit.state.v],
  delta: (before, after) => {
    const beforeEmb = [before.state.x, before.state.v];
    const afterEmb = [after.state.x, after.state.v];
    return sub(afterEmb, beforeEmb);
  },
};

interface Checkpoint {
  sample: number;

  // Dynamics learning (first grokking)
  errorInDist: number;
  errorOutDist: number;

  // Epistemic learning (second grokking)
  agencyInDist: number;
  agencyOutDist: number;
  coverageRate: number;

  // System dynamics
  portCount: number;
  avgConeVolume: number;

  // Calibration quality
  calibrationGap: number; // |agency - actual_binding|
}

interface DistributionSplit {
  inDist: Array<{ x: number; v: number }>;
  outDist: Array<{ x: number; v: number }>;
}

function generateTestSets(): DistributionSplit {
  const inDist: Array<{ x: number; v: number }> = [];
  const outDist: Array<{ x: number; v: number }> = [];

  // In-distribution: training region [-1, 1] x [-1, 1]
  for (let i = 0; i < 100; i++) {
    inDist.push(randomState());
  }

  // Out-of-distribution: extrapolation region
  for (let i = 0; i < 50; i++) {
    const sign = Math.random() < 0.5 ? -1 : 1;
    outDist.push({
      x: sign * (1.5 + Math.random() * 0.5), // [1.5, 2.0] or [-2.0, -1.5]
      v: sign * (1.5 + Math.random() * 0.5),
    });
  }

  return { inDist, outDist };
}

function evaluateError(
  bank: GeometricPortBank<{ x: number; v: number }, {}>,
  testStates: Array<{ x: number; v: number }>
): { error: number; agency: number; bindingRate: number; avgConeVolume: number } {
  let totalError = 0;
  let totalAgency = 0;
  let boundCount = 0;
  let totalConeVolume = 0;
  let count = 0;

  for (const state of testStates) {
    const truth = springStep(state.x, state.v);
    const truthDelta = [truth.x - state.x, truth.v - state.v];

    const pred = bank.predictFromState("default", { state, context: {} }, 1)[0];
    if (!pred) continue;

    const err = Math.sqrt(
      Math.pow(pred.delta[0]! - truthDelta[0]!, 2) +
      Math.pow(pred.delta[1]! - truthDelta[1]!, 2)
    );

    // Check if prediction actually bound (error < threshold)
    const bound = err < 0.1;

    totalError += err;
    totalAgency += pred.agency;
    if (bound) boundCount++;

    // Estimate cone volume from agency (agency ≈ 1 - normalized_cone_volume)
    totalConeVolume += (1 - pred.agency);

    count++;
  }

  if (count === 0) return { error: 1.0, agency: 0, bindingRate: 0, avgConeVolume: 1.0 };

  return {
    error: totalError / count,
    agency: totalAgency / count,
    bindingRate: boundCount / count,
    avgConeVolume: totalConeVolume / count,
  };
}

function computeCalibrationGap(agency: number, bindingRate: number): number {
  // How well does agency predict actual binding success?
  // Perfect calibration: agency = bindingRate
  return Math.abs(agency - bindingRate);
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Double Grokking Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Tracking the two phases of learning:");
  console.log("  1. First grokking:  Learning WHAT happens (error drops)");
  console.log("  2. Second grokking: Learning CONFIDENCE (agency calibrates)\n");
  console.log("Domain: 1D Damped Spring (k=1.0, b=0.1)");
  console.log("Training: 2000 samples with checkpoints every 50 samples\n");

  // Create bank
  const bank = new GeometricPortBank<{ x: number; v: number }, {}>(encoders, {
    embeddingDim: 2,
    initialRadius: 0.3,
    learningRate: 0.01,
  });

  // Generate test sets
  const testSets = generateTestSets();
  console.log(`Test sets: ${testSets.inDist.length} in-dist, ${testSets.outDist.length} out-dist\n`);

  // Training loop with fine-grained checkpoints
  const checkpoints: Checkpoint[] = [];
  const totalSamples = 2000;
  const checkpointEvery = 50;

  console.log("Sample | Error (In/Out) | Agency (In/Out) | Coverage | Ports | Cal.Gap");
  console.log("-------|----------------|-----------------|----------|-------|--------");

  for (let sample = 0; sample < totalSamples; sample++) {
    // Training step
    const before = randomState();
    const after = springStep(before.x, before.v);

    bank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      reward: 1.0,
    });

    // Checkpoint evaluation
    if ((sample + 1) % checkpointEvery === 0) {
      bank.flush();

      const evalInDist = evaluateError(bank, testSets.inDist);
      const evalOutDist = evaluateError(bank, testSets.outDist);

      const checkpoint: Checkpoint = {
        sample: sample + 1,
        errorInDist: evalInDist.error,
        errorOutDist: evalOutDist.error,
        agencyInDist: evalInDist.agency,
        agencyOutDist: evalOutDist.agency,
        coverageRate: evalInDist.bindingRate,
        portCount: bank.getPortIds().length,
        avgConeVolume: evalInDist.avgConeVolume,
        calibrationGap: computeCalibrationGap(evalInDist.agency, evalInDist.bindingRate),
      };

      checkpoints.push(checkpoint);

      console.log(
        `${checkpoint.sample.toString().padStart(6)} | ` +
        `${checkpoint.errorInDist.toFixed(4)}/${checkpoint.errorOutDist.toFixed(4)} | ` +
        `${(checkpoint.agencyInDist * 100).toFixed(1).padStart(4)}%/${(checkpoint.agencyOutDist * 100).toFixed(1).padStart(4)}% | ` +
        `${(checkpoint.coverageRate * 100).toFixed(1).padStart(6)}% | ` +
        `${checkpoint.portCount.toString().padStart(5)} | ` +
        `${checkpoint.calibrationGap.toFixed(3)}`
      );
    }
  }

  // Final flush
  bank.flush();

  // Analyze grokking moments
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Grokking Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  // First grokking: sudden error drop
  let firstGrokkingSample = null;
  let maxErrorDrop = 0;

  for (let i = 1; i < checkpoints.length; i++) {
    const errorDrop = checkpoints[i - 1]!.errorInDist - checkpoints[i]!.errorInDist;
    if (errorDrop > maxErrorDrop) {
      maxErrorDrop = errorDrop;
      firstGrokkingSample = checkpoints[i]!.sample;
    }
  }

  // Second grokking: sudden agency improvement
  let secondGrokkingSample = null;
  let maxAgencyJump = 0;

  for (let i = 1; i < checkpoints.length; i++) {
    const agencyJump = checkpoints[i]!.agencyInDist - checkpoints[i - 1]!.agencyInDist;
    if (agencyJump > maxAgencyJump && checkpoints[i]!.sample > (firstGrokkingSample || 0)) {
      maxAgencyJump = agencyJump;
      secondGrokkingSample = checkpoints[i]!.sample;
    }
  }

  // Find calibration convergence: when calibration gap becomes consistently small
  let calibrationConvergenceSample = null;
  const calibrationThreshold = 0.1;
  let consecutiveGood = 0;

  for (let i = 0; i < checkpoints.length; i++) {
    if (checkpoints[i]!.calibrationGap < calibrationThreshold) {
      consecutiveGood++;
      if (consecutiveGood >= 3 && !calibrationConvergenceSample) {
        calibrationConvergenceSample = checkpoints[i]!.sample;
      }
    } else {
      consecutiveGood = 0;
    }
  }

  if (firstGrokkingSample) {
    console.log(`✓ First Grokking (Dynamics): Sample ~${firstGrokkingSample}`);
    console.log(`  Error dropped by ${maxErrorDrop.toFixed(4)}`);
    console.log(`  Interpretation: Model learned the spring dynamics\n`);
  } else {
    console.log(`⚠ First grokking not clearly observed\n`);
  }

  if (secondGrokkingSample) {
    console.log(`✓ Second Grokking (Epistemics): Sample ~${secondGrokkingSample}`);
    console.log(`  Agency jumped by ${(maxAgencyJump * 100).toFixed(1)}%`);
    console.log(`  Interpretation: Model learned when to be confident\n`);
  } else {
    console.log(`⚠ Second grokking not clearly observed\n`);
  }

  if (calibrationConvergenceSample) {
    console.log(`✓ Calibration Convergence: Sample ~${calibrationConvergenceSample}`);
    console.log(`  Calibration gap fell below ${calibrationThreshold}`);
    console.log(`  Interpretation: Model's confidence matches actual performance\n`);
  } else {
    console.log(`⚠ Calibration not yet converged\n`);
  }

  // Phase analysis
  const initial = checkpoints[0]!;
  const final = checkpoints[checkpoints.length - 1]!;

  console.log("Phase Summary:");
  console.log(`  Initial → Final:`);
  console.log(`    Error (in-dist):     ${initial.errorInDist.toFixed(4)} → ${final.errorInDist.toFixed(4)} (${((1 - final.errorInDist / initial.errorInDist) * 100).toFixed(1)}% improvement)`);
  console.log(`    Agency (in-dist):    ${(initial.agencyInDist * 100).toFixed(1)}% → ${(final.agencyInDist * 100).toFixed(1)}%`);
  console.log(`    Coverage:            ${(initial.coverageRate * 100).toFixed(1)}% → ${(final.coverageRate * 100).toFixed(1)}%`);
  console.log(`    Calibration Gap:     ${initial.calibrationGap.toFixed(3)} → ${final.calibrationGap.toFixed(3)}`);
  console.log(`    Ports:               ${initial.portCount} → ${final.portCount}`);

  // Export detailed results
  const exportData = {
    name: "Double Grokking Experiment",
    config: {
      domain: "1D Damped Spring (k=1.0, b=0.1)",
      samples: totalSamples,
      checkpointEvery,
      testSets: {
        inDist: testSets.inDist.length,
        outDist: testSets.outDist.length,
      },
    },
    grokking: {
      firstGrokking: firstGrokkingSample ? {
        sample: firstGrokkingSample,
        errorDrop: maxErrorDrop,
        interpretation: "Learned dynamics",
      } : null,
      secondGrokking: secondGrokkingSample ? {
        sample: secondGrokkingSample,
        agencyJump: maxAgencyJump,
        interpretation: "Learned epistemic calibration",
      } : null,
      calibrationConvergence: calibrationConvergenceSample ? {
        sample: calibrationConvergenceSample,
        threshold: calibrationThreshold,
        interpretation: "Confidence matches performance",
      } : null,
    },
    checkpoints,
    summary: {
      initial: {
        error: initial.errorInDist,
        agency: initial.agencyInDist,
        coverage: initial.coverageRate,
        calibrationGap: initial.calibrationGap,
      },
      final: {
        error: final.errorInDist,
        agency: final.agencyInDist,
        coverage: final.coverageRate,
        calibrationGap: final.calibrationGap,
      },
      improvement: {
        errorReduction: ((1 - final.errorInDist / initial.errorInDist) * 100),
        agencyIncrease: ((final.agencyInDist - initial.agencyInDist) * 100),
        calibrationImprovement: ((initial.calibrationGap - final.calibrationGap) / initial.calibrationGap * 100),
      },
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write(
    "examples/results/07-double-grokking.json",
    JSON.stringify(exportData, null, 2)
  );

  console.log("\n✓ Results saved to examples/results/07-double-grokking.json");
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Insight");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  Double grokking shows two distinct learning phases:");
  console.log("    1. Dynamics learning:  Model figures out WHAT happens");
  console.log("    2. Epistemic learning: Model figures out HOW SURE to be");
  console.log("\n  This separation suggests the model is not just fitting,");
  console.log("  but developing genuine understanding of both the world");
  console.log("  (ontology) and its own knowledge (epistemology).\n");

  // Cleanup
  bank.dispose();

  console.log("✓ Experiment complete!");
}

main().catch(console.error);
