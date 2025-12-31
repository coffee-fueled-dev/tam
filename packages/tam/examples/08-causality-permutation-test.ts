/**
 * Experiment 08: Causality Permutation Test
 *
 * Critical validation: Does TAM's agency jump depend on causal structure?
 *
 * Setup:
 * - Condition A (Causal): Train with correct before→after pairs
 * - Condition B (Permuted): Train with shuffled after states (breaks causality)
 *
 * Hypothesis:
 * If TAM learns causal structure (not just distribution):
 * - Causal condition: Agency should jump (second grokking)
 * - Permuted condition: Agency should NOT jump (no causal structure to learn)
 *
 * This test distinguishes genuine causal learning from statistical fitting.
 */

import { GeometricPortBank } from "../src/geometric/bank";
import type { Encoders, Situation } from "../src/types";
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
    x: (Math.random() - 0.5) * 2,
    v: (Math.random() - 0.5) * 2,
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
  error: number;
  agency: number;
  coverage: number;
  portCount: number;
}

function evaluateBank(
  bank: GeometricPortBank<{ x: number; v: number }, {}>,
  testStates: Array<{ x: number; v: number }>,
  useCausalTruth: boolean
): { error: number; agency: number; coverage: number } {
  let totalError = 0;
  let totalAgency = 0;
  let boundCount = 0;
  let count = 0;

  for (const state of testStates) {
    const truth = useCausalTruth ? springStep(state.x, state.v) : randomState();
    const truthDelta = [truth.x - state.x, truth.v - state.v];

    const pred = bank.predictFromState("default", { state, context: {} }, 1)[0];
    if (!pred) continue;

    const err = Math.sqrt(
      Math.pow(pred.delta[0]! - truthDelta[0]!, 2) +
      Math.pow(pred.delta[1]! - truthDelta[1]!, 2)
    );

    const bound = err < 0.1;

    totalError += err;
    totalAgency += pred.agency;
    if (bound) boundCount++;
    count++;
  }

  if (count === 0) return { error: 1.0, agency: 0, coverage: 0 };

  return {
    error: totalError / count,
    agency: totalAgency / count,
    coverage: boundCount / count,
  };
}

async function trainWithCondition(
  conditionName: string,
  permuted: boolean,
  totalSamples: number,
  checkpointEvery: number
): Promise<{ checkpoints: Checkpoint[]; bank: GeometricPortBank<any, any> }> {
  const bank = new GeometricPortBank<{ x: number; v: number }, {}>(encoders, {
    embeddingDim: 2,
    initialRadius: 0.3,
    learningRate: 0.01,
  });

  const checkpoints: Checkpoint[] = [];

  // Generate test set (always use causal truth for evaluation)
  const testStates: Array<{ x: number; v: number }> = [];
  for (let i = 0; i < 100; i++) {
    testStates.push(randomState());
  }

  // Pre-generate a pool of random states for permutation
  const permutationPool: Array<{ x: number; v: number }> = [];
  if (permuted) {
    for (let i = 0; i < totalSamples; i++) {
      permutationPool.push(randomState());
    }
  }

  console.log(`\nTraining ${conditionName}:`);
  console.log("Sample | Error   | Agency  | Coverage | Note");
  console.log("-------|---------|---------|----------|-----");

  for (let sample = 0; sample < totalSamples; sample++) {
    const before = randomState();
    const after = permuted
      ? permutationPool[sample]! // Random unrelated state
      : springStep(before.x, before.v); // Causal successor

    bank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      reward: 1.0,
    });

    // Checkpoint
    if ((sample + 1) % checkpointEvery === 0) {
      bank.flush();

      const eval_result = evaluateBank(bank, testStates, true); // Always use causal truth for eval

      const checkpoint: Checkpoint = {
        sample: sample + 1,
        error: eval_result.error,
        agency: eval_result.agency,
        coverage: eval_result.coverage,
        portCount: bank.getPortIds().length,
      };

      checkpoints.push(checkpoint);

      // Detect notable events
      let note = "";
      if (checkpoint.coverage > 0.5 && (checkpoints.length === 1 || checkpoints[checkpoints.length - 2]!.coverage <= 0.5)) {
        note = "Coverage > 50%";
      }
      if (checkpoint.agency > 0.8 && (checkpoints.length === 1 || checkpoints[checkpoints.length - 2]!.agency <= 0.8)) {
        note = "Agency > 80%";
      }
      if (checkpoint.coverage >= 1.0 && (checkpoints.length === 1 || checkpoints[checkpoints.length - 2]!.coverage < 1.0)) {
        note = "Full coverage";
      }

      console.log(
        `${checkpoint.sample.toString().padStart(6)} | ` +
        `${checkpoint.error.toFixed(4)} | ` +
        `${(checkpoint.agency * 100).toFixed(1).padStart(6)}% | ` +
        `${(checkpoint.coverage * 100).toFixed(1).padStart(7)}% | ` +
        `${note}`
      );
    }
  }

  bank.flush();

  return { checkpoints, bank };
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Causality Permutation Test");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Critical validation: Does agency jump require causal structure?\n");
  console.log("Setup:");
  console.log("  Condition A (Causal):   Train with correct before→after pairs");
  console.log("  Condition B (Permuted): Train with random after states\n");
  console.log("Hypothesis:");
  console.log("  If TAM learns causality, agency jumps only in causal condition.\n");

  const totalSamples = 2000;
  const checkpointEvery = 100;

  // Run both conditions
  const causalResult = await trainWithCondition("Causal (Control)", false, totalSamples, checkpointEvery);
  const permutedResult = await trainWithCondition("Permuted (Broken Causality)", true, totalSamples, checkpointEvery);

  // Analysis
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis: Causal vs Permuted");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Find agency jumps
  function findLargestAgencyJump(checkpoints: Checkpoint[]): { sample: number; jump: number } | null {
    let maxJump = 0;
    let maxSample = null;

    for (let i = 1; i < checkpoints.length; i++) {
      const jump = checkpoints[i]!.agency - checkpoints[i - 1]!.agency;
      if (jump > maxJump) {
        maxJump = jump;
        maxSample = checkpoints[i]!.sample;
      }
    }

    return maxSample ? { sample: maxSample, jump: maxJump } : null;
  }

  const causalJump = findLargestAgencyJump(causalResult.checkpoints);
  const permutedJump = findLargestAgencyJump(permutedResult.checkpoints);

  console.log("Agency Jump Detection:\n");

  if (causalJump) {
    console.log(`Causal Condition:`);
    console.log(`  Largest agency jump: ${(causalJump.jump * 100).toFixed(1)}% at sample ${causalJump.sample}`);
    console.log(`  Final agency: ${(causalResult.checkpoints[causalResult.checkpoints.length - 1]!.agency * 100).toFixed(1)}%`);
  } else {
    console.log(`Causal Condition: No significant agency jump detected`);
  }

  console.log();

  if (permutedJump) {
    console.log(`Permuted Condition:`);
    console.log(`  Largest agency jump: ${(permutedJump.jump * 100).toFixed(1)}% at sample ${permutedJump.sample}`);
    console.log(`  Final agency: ${(permutedResult.checkpoints[permutedResult.checkpoints.length - 1]!.agency * 100).toFixed(1)}%`);
  } else {
    console.log(`Permuted Condition: No significant agency jump detected`);
  }

  // Compare final states
  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Final Performance Comparison:");
  console.log("─────────────────────────────────────────────────────────\n");

  const causalFinal = causalResult.checkpoints[causalResult.checkpoints.length - 1]!;
  const permutedFinal = permutedResult.checkpoints[permutedResult.checkpoints.length - 1]!;

  console.log("                | Causal      | Permuted    | Difference");
  console.log("----------------|-------------|-------------|------------");
  console.log(
    `Error           | ${causalFinal.error.toFixed(4).padStart(11)} | ` +
    `${permutedFinal.error.toFixed(4).padStart(11)} | ` +
    `${((permutedFinal.error - causalFinal.error) / causalFinal.error * 100).toFixed(1)}%`
  );
  console.log(
    `Agency          | ${(causalFinal.agency * 100).toFixed(1).padStart(10)}% | ` +
    `${(permutedFinal.agency * 100).toFixed(1).padStart(10)}% | ` +
    `${((permutedFinal.agency - causalFinal.agency) * 100).toFixed(1)}%`
  );
  console.log(
    `Coverage        | ${(causalFinal.coverage * 100).toFixed(1).padStart(10)}% | ` +
    `${(permutedFinal.coverage * 100).toFixed(1).padStart(10)}% | ` +
    `${((permutedFinal.coverage - causalFinal.coverage) * 100).toFixed(1)}%`
  );
  console.log(
    `Ports           | ${causalFinal.portCount.toString().padStart(11)} | ` +
    `${permutedFinal.portCount.toString().padStart(11)} | ` +
    `${permutedFinal.portCount - causalFinal.portCount}`
  );

  // Verdict
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Verdict");
  console.log("═══════════════════════════════════════════════════════════\n");

  const agencyGap = causalFinal.agency - permutedFinal.agency;
  const jumpRatio = causalJump && permutedJump ? causalJump.jump / permutedJump.jump : 0;

  if (agencyGap > 0.2 && causalJump && causalJump.jump > 0.03) {
    console.log("✓ CAUSALITY CONFIRMED");
    console.log("\n  Evidence:");
    console.log(`  1. Causal condition achieves ${(causalFinal.agency * 100).toFixed(1)}% agency`);
    console.log(`  2. Permuted condition only reaches ${(permutedFinal.agency * 100).toFixed(1)}% agency`);
    console.log(`  3. Gap: ${(agencyGap * 100).toFixed(1)}% (causal learns epistemics)`);
    console.log(`  4. Causal shows clear second grokking (jump: ${(causalJump.jump * 100).toFixed(1)}%)`);

    if (permutedJump && permutedJump.jump < causalJump.jump * 0.5) {
      console.log(`  5. Permuted lacks second grokking (jump only ${(permutedJump.jump * 100).toFixed(1)}%)`);
    }

    console.log("\n  Interpretation:");
    console.log("  TAM's agency jump is NOT just distribution fitting.");
    console.log("  The model genuinely learns causal structure and develops");
    console.log("  epistemic awareness based on predictable dynamics.\n");
  } else if (agencyGap > 0.1) {
    console.log("≈ PARTIAL SENSITIVITY TO CAUSALITY");
    console.log("\n  Causal condition shows higher agency, but difference is modest.");
    console.log("  May need longer training or different hyperparameters.\n");
  } else {
    console.log("⚠ CAUSALITY NOT DISTINGUISHED");
    console.log("\n  Both conditions achieve similar agency.");
    console.log("  This suggests the model may be learning distributional");
    console.log("  patterns rather than true causal structure.\n");
  }

  // Export
  const exportData = {
    name: "Causality Permutation Test",
    config: {
      domain: "1D Damped Spring",
      samples: totalSamples,
      checkpointEvery,
    },
    results: {
      causal: {
        checkpoints: causalResult.checkpoints,
        agencyJump: causalJump,
        final: causalFinal,
      },
      permuted: {
        checkpoints: permutedResult.checkpoints,
        agencyJump: permutedJump,
        final: permutedFinal,
      },
    },
    analysis: {
      agencyGap,
      jumpRatio: causalJump && permutedJump ? jumpRatio : null,
      verdict: agencyGap > 0.2 ? "Causality confirmed" : agencyGap > 0.1 ? "Partial sensitivity" : "Not distinguished",
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write(
    "examples/results/08-causality-permutation-test.json",
    JSON.stringify(exportData, null, 2)
  );

  console.log("✓ Results saved to examples/results/08-causality-permutation-test.json");

  // Cleanup
  causalResult.bank.dispose();
  permutedResult.bank.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
