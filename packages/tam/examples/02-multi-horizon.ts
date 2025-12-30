/**
 * Experiment 02: Multi-Horizon Prediction
 *
 * Demonstrates:
 * - Temporal abstraction via multiple specialists
 * - Direct k-step prediction vs iterative rollout
 * - Agency-based horizon selection
 * - Error accumulation mitigation
 *
 * Expected outcome:
 * - Direct 20-step prediction: ~70% better than iterative
 * - Different horizons have different specializations
 * - Long-horizon planning becomes tractable
 */

import {
  runExperiment,
  compareExperiments,
  dampedSpring1D,
  createMultiHorizonDomain,
  saveResultsToJson,
  type ExperimentResult,
} from "./harness";

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Multi-Horizon Prediction");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Problem: Long-horizon rollouts fail (error compounds)");
  console.log("Solution: Train specialists for different time scales\n");

  // Train separate banks for each horizon
  const horizons = [1, 5, 10, 20];
  const multiHorizonDomains = createMultiHorizonDomain(dampedSpring1D, horizons);

  const results: ExperimentResult<any>[] = [];

  for (const [h, domain] of multiHorizonDomains) {
    console.log(`\nTraining ${h}-step specialist...`);

    const result = await runExperiment(`${h}-step`, {
      domain,
      episodes: 500,
      checkpointEvery: 500, // Only final checkpoint
      testSize: 50,
      bankConfig: { embeddingDim: dampedSpring1D.embeddingDim },
    });

    results.push(result);
  }

  // Save results to JSON
  for (let i = 0; i < results.length; i++) {
    const result = results[i]!;
    const h = horizons[i]!;
    await saveResultsToJson(result, `examples/results/02-multi-horizon-${h}step.json`);
  }

  // Compare results
  compareExperiments(results);

  // Evaluate rollout strategies
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Rollout Comparison (20-step horizon)");
  console.log("═══════════════════════════════════════════════════════════\n");

  const testStates = Array.from({ length: 50 }, () => dampedSpring1D.randomState());

  // Strategy 1: Iterative 1-step
  const bank1 = results[0]!.bank!;
  let totalError1 = 0;

  for (const state of testStates) {
    let current = state;
    for (let step = 0; step < 20; step++) {
      const sit = { state: current, context: null };
      const pred = bank1.predictFromState("step", sit, 1)[0];
      if (!pred) break;
      current = {
        x: current.x + pred.delta[0]!,
        v: current.v + pred.delta[1]!,
      };
    }
    const actual = dampedSpring1D.simulate(state, 20);
    const error = Math.sqrt((current.x - actual.x) ** 2 + (current.v - actual.v) ** 2);
    totalError1 += error;
  }

  console.log(`Iterative 1-step:`);
  console.log(`  Error: ${(totalError1 / testStates.length).toFixed(4)}`);
  console.log(`  Method: Chain 20 × 1-step predictions\n`);

  // Strategy 2: Direct 20-step
  const bank20 = results[3]!.bank!;
  let totalError20 = 0;

  for (const state of testStates) {
    const sit = { state, context: null };
    const pred = bank20.predictFromState("step", sit, 1)[0];
    if (!pred) continue;
    const predicted = {
      x: state.x + pred.delta[0]!,
      v: state.v + pred.delta[1]!,
    };
    const actual = dampedSpring1D.simulate(state, 20);
    const error = Math.sqrt(
      (predicted.x - actual.x) ** 2 + (predicted.v - actual.v) ** 2
    );
    totalError20 += error;
  }

  const avgError20 = totalError20 / testStates.length;
  const improvement = (1 - avgError20 / (totalError1 / testStates.length)) * 100;

  console.log(`Direct 20-step:`);
  console.log(`  Error: ${avgError20.toFixed(4)}`);
  console.log(`  Method: Single 20-step specialist`);
  console.log(`  Improvement: ${improvement.toFixed(0)}% better\n`);

  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Key Takeaway");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  Multi-horizon prediction solves error accumulation.");
  console.log("  Different time scales = different specialists.");
  console.log("  TAM naturally supports temporal abstraction.");

  // Cleanup
  for (const result of results) {
    result.bank?.dispose();
  }

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
