/**
 * Experiment 01: Basic Single-Domain Training
 *
 * Demonstrates:
 * - Simple training loop
 * - Metric tracking
 * - Grokking detection
 * - Basic calibration
 *
 * Expected outcome:
 * - Model learns spring dynamics
 * - Shows grokking transition around 200-500 episodes
 * - Achieves <0.05 prediction error
 * - Agency correlates with accuracy
 */

import { runExperiment, dampedSpring1D, saveResultsToJson } from "./harness";

async function main() {
  const result = await runExperiment("Basic Training", {
    domain: dampedSpring1D,
    episodes: 1000,
    checkpointEvery: 100,
    testSize: 100,
  });

  // Save results to JSON
  await saveResultsToJson(result, "examples/results/01-basic-training.json");

  // Cleanup
  result.bank?.dispose();

  console.log("\n✓ Experiment complete!");
  console.log("\nKey Takeaway:");
  console.log("  TAM can learn simple dynamics from scratch.");
  console.log("  Grokking shows sudden generalization (not just memorization).");
  console.log("  Agency and accuracy correlate when properly trained.");
}

main().catch(console.error);
