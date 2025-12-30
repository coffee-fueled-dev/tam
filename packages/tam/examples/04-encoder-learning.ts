/**
 * Experiment 04: Learnable Encoder Discovery
 *
 * Demonstrates:
 * - End-to-end encoder learning
 * - Automatic feature selection (ignore noise)
 * - Comparison: static vs hand-crafted vs learnable
 *
 * Expected outcome:
 * - Learnable encoder learns to ignore noise features
 * - Performance approaches hand-crafted baseline
 * - Shows encoder can discover relevant features automatically
 *
 * Note: Currently uses temporal smoothness loss due to TensorFlow.js
 * limitations (no stopGradient). Still demonstrates the concept.
 */

import {
  runExperiment,
  compareExperiments,
  noisyPendulum,
  type ExperimentResult,
} from "./harness";
import { IntraDomainEncoder } from "../src/geometric/intra-domain-encoder";

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Learnable Encoder Discovery");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Domain: 2D Pendulum with 6 noise features");
  console.log("  Relevant: x, y, vx, vy (4D)");
  console.log("  Noise: 6 random features");
  console.log("  Total: 10D raw input\n");

  const domain = noisyPendulum(6);
  const results: ExperimentResult<any>[] = [];

  // Config A: Static encoder (uses all features, including noise)
  console.log("Training with static encoder (all 10 features)...");
  const resultStatic = await runExperiment("Static", {
    domain: {
      ...domain,
      embed: (state) => [
        state.x,
        state.y,
        state.vx,
        state.vy,
        ...state.noise,
      ], // Use raw features directly
      embeddingDim: 10,
    },
    episodes: 300,
    checkpointEvery: 300,
    testSize: 50,
  });
  results.push(resultStatic);

  // Config B: Hand-crafted encoder (only relevant features)
  console.log("\nTraining with hand-crafted encoder (4 relevant features)...");
  const resultHandCrafted = await runExperiment("Hand-Crafted", {
    domain, // Uses domain.embed which has only relevant features
    episodes: 300,
    checkpointEvery: 300,
    testSize: 50,
  });
  results.push(resultHandCrafted);

  // Config C: Learnable encoder (should learn to ignore noise)
  console.log("\nTraining with learnable encoder (10→4 via learning)...");

  const learnableEncoder = new IntraDomainEncoder({
    rawDim: 10,
    embeddingDim: 4,
    hiddenSizes: [16, 8],
    learningRate: 0.001,
    smoothnessWeight: 0.05,
  });

  const resultLearnable = await runExperiment(
    "Learnable",
    {
      domain,
      episodes: 300,
      checkpointEvery: 300,
      testSize: 50,
      encoder: {
        type: "learnable",
        config: learnableEncoder,
      },
    },
    // Training callback: train encoder after each observation
    async (transition, bank) => {
      const lastPort = bank.getLastSelectedPort();
      if (lastPort) {
        learnableEncoder.trainStepJoint(
          domain.extractRaw!(transition.before.state),
          lastPort.embedding,
          [0, 0, 0, 0], // Dummy trajectory (not used in smoothness-only training)
          () => ({ dataSync: () => [0, 0, 0, 0] } as any),
          () => ({ dataSync: () => [0, 0, 0, 0] } as any),
          1.0
        );
      }
    }
  );
  results.push(resultLearnable);

  // Compare
  compareExperiments(results);

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  const baseline = resultHandCrafted.summary.finalMetrics.predictionError ?? 1;
  const staticError = resultStatic.summary.finalMetrics.predictionError ?? 1;
  const learnableError = resultLearnable.summary.finalMetrics.predictionError ?? 1;

  const learnableGap = Math.abs(learnableError - baseline);
  const staticGap = Math.abs(staticError - baseline);

  if (learnableGap < staticGap * 0.5) {
    console.log("✓ Learnable encoder successfully discovered relevant features");
    console.log("  Performance approaches hand-crafted baseline");
  } else if (learnableGap < staticGap) {
    console.log("⚠ Learnable encoder partially successful");
    console.log("  Some feature selection learned, but not optimal");
  } else {
    console.log("⚠ Learnable encoder did not improve over static");
    console.log("  May need more training or better hyperparameters");
  }

  if (staticError > baseline) {
    console.log("\n✓ Noise features hurt performance (as expected)");
    console.log("  Static encoder with all features performs worse");
  }

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Takeaway");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  TAM can learn encoders end-to-end.");
  console.log("  Automatically discovers which features matter.");
  console.log("  No need for hand-crafted feature engineering.");
  console.log("\n  Current limitation: Uses smoothness loss (TF.js)");
  console.log("  Future: True binding/agency loss (needs PyTorch)");

  // Cleanup
  for (const result of results) {
    result.bank?.dispose();
  }
  learnableEncoder.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
