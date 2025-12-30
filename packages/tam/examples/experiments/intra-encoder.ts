/**
 * Intra-Domain Encoder Learning Experiment
 *
 * Validates end-to-end encoder learning with temporal smoothness objective.
 * Compares three approaches:
 * 1. Static encoder (uses all features, including noise)
 * 2. Hand-crafted encoder (uses only relevant features)
 * 3. Learnable encoder (learns via temporal smoothness)
 *
 * Test domain: 2D pendulum with noise features
 * - Relevant: position (x, y), velocity (vx, vy)
 * - Noise: 6 random features that don't affect dynamics
 * - Dynamics: Simple physics (gravity + damping)
 *
 * NOTE: Due to TensorFlow.js lacking tf.stopGradient(), the learnable encoder
 * uses temporal smoothness loss rather than direct binding/agency loss.
 * This maintains loose coupling with shared networks (alternating optimization).
 *
 * Expected result:
 * - Learnable encoder successfully trains without errors
 * - Performance may be lower than hand-crafted due to indirect learning signal
 * - Demonstrates end-to-end encoder optimization is feasible
 */

import {
  GeometricPortBank,
  type Encoders,
  type Situation,
  type Transition,
  type GeometricPortConfigInput,
  IntraDomainEncoder,
  createEncoderBridge,
  type EncoderBridge,
} from "../../src";
import { sub } from "../../src/vec";

// ============================================================================
// Domain: 2D Pendulum with Noise
// ============================================================================

interface PendulumState {
  x: number; // Position X
  y: number; // Position Y
  vx: number; // Velocity X
  vy: number; // Velocity Y
  noise: number[]; // 6 noise features (irrelevant)
}

/**
 * Simulate one step: gravity + damping + noise in observations
 */
function simulate(state: PendulumState): PendulumState {
  const dt = 0.1;
  const gravity = 0.5;
  const damping = 0.95;

  // Physics (deterministic)
  const ax = 0;
  const ay = gravity;

  const newVx = (state.vx + ax * dt) * damping;
  const newVy = (state.vy + ay * dt) * damping;

  const newX = state.x + newVx * dt;
  const newY = state.y + newVy * dt;

  // Generate new noise (different each step, not predictive)
  const newNoise = Array.from({ length: 6 }, () => Math.random() - 0.5);

  return {
    x: newX,
    y: newY,
    vx: newVx,
    vy: newVy,
    noise: newNoise,
  };
}

/**
 * Random initial state
 */
function randomState(): PendulumState {
  return {
    x: (Math.random() - 0.5) * 2,
    y: (Math.random() - 0.5) * 2,
    vx: (Math.random() - 0.5) * 2,
    vy: (Math.random() - 0.5) * 2,
    noise: Array.from({ length: 6 }, () => Math.random() - 0.5),
  };
}

/**
 * Extract raw features (all features, including noise)
 */
function extractRaw(state: PendulumState): number[] {
  return [state.x, state.y, state.vx, state.vy, ...state.noise];
}

/**
 * Hand-crafted embedder (optimal - only relevant features)
 */
function handCraftedEmbed(state: PendulumState): number[] {
  return [state.x, state.y, state.vx, state.vy];
}

// ============================================================================
// Training
// ============================================================================

async function trainWithEncoder(
  label: string,
  bridge: EncoderBridge<PendulumState>,
  episodes: number = 300
): Promise<{
  bank: GeometricPortBank<PendulumState, unknown>;
  avgAgency: number;
  encoderStats?: ReturnType<typeof bridge.getEncoderStats>;
}> {
  console.log(`\nTraining ${label}...`);

  const config: GeometricPortConfigInput = {
    embeddingDim: 4, // Small embedding (4 relevant features)
  };

  const bank = new GeometricPortBank(bridge.encoders, config);

  for (let i = 0; i < episodes; i++) {
    const before = randomState();
    const after = simulate(before);

    const transition: Transition<PendulumState, unknown> = {
      before: { state: before, context: null },
      after: { state: after, context: null },
      action: "step",
    };

    await bank.observe(transition);

    // Train encoder if learnable
    if (bridge.isLearnable()) {
      const lastPort = bank.getLastSelectedPort();
      if (lastPort) {
        const losses = bridge.trainEncoder(
          transition,
          bank.getCausalNet(),
          bank.getCommitmentNet(),
          lastPort.embedding,
          bank.getReferenceVolume()
        );

        // Log occasional training stats
        if (losses && (i + 1) % 100 === 0) {
          console.log(
            `  [${i + 1}/${episodes}] Encoder loss: binding=${losses.bindingLoss.toFixed(3)}, agency=${losses.agencyLoss.toFixed(3)}`
          );
        }
      }
    }

    // Periodic flush
    if ((i + 1) % 50 === 0) {
      bank.flush();
    }
  }

  bank.flush();

  // Evaluate agency on test set
  const testSamples = 20;
  let totalAgency = 0;

  for (let i = 0; i < testSamples; i++) {
    const testState = randomState();
    const sit: Situation<PendulumState, unknown> = {
      state: testState,
      context: null,
    };

    const predictions = bank.predict("step", sit, 1);
    totalAgency += predictions[0]?.agency ?? 0;
  }

  const avgAgency = totalAgency / testSamples;
  console.log(`  Average agency: ${(avgAgency * 100).toFixed(1)}%`);

  return {
    bank,
    avgAgency,
    encoderStats: bridge.getEncoderStats(),
  };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Intra-Domain Encoder Learning Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Domain: 2D Pendulum with Noise");
  console.log("  Relevant features: x, y, vx, vy (4D)");
  console.log("  Noise features: 6 random values (not predictive)");
  console.log("  Total raw dimension: 10D");

  // ============================================================================
  // Config A: Static Encoder (All Features Including Noise)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration A: Static Encoder (All 10 Features)");
  console.log("─────────────────────────────────────────────────────────");

  const bridgeStatic = createEncoderBridge<PendulumState>({
    extractRaw, // Uses all features as-is
  });

  const resultStatic = await trainWithEncoder(
    "static encoder",
    bridgeStatic,
    300
  );

  // ============================================================================
  // Config B: Hand-Crafted Encoder (Only Relevant Features)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration B: Hand-Crafted Encoder (4 Relevant Features)");
  console.log("─────────────────────────────────────────────────────────");

  const bridgeHandCrafted = createEncoderBridge<PendulumState>({
    extractRaw,
    staticEmbedder: handCraftedEmbed, // Only relevant features
  });

  const resultHandCrafted = await trainWithEncoder(
    "hand-crafted encoder",
    bridgeHandCrafted,
    300
  );

  // ============================================================================
  // Config C: Learnable Encoder (Should Learn to Ignore Noise)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration C: Learnable Encoder (End-to-End)");
  console.log("─────────────────────────────────────────────────────────");

  const learnableEncoder = new IntraDomainEncoder({
    rawDim: 10, // All features
    embeddingDim: 4, // Compress to relevant dimension
    hiddenSizes: [16, 8],
    learningRate: 0.001,
    agencyWeight: 0.1,
    smoothnessWeight: 0.05,
  });

  const bridgeLearnable = createEncoderBridge<PendulumState>({
    extractRaw,
    learnableEncoder,
  });

  const resultLearnable = await trainWithEncoder(
    "learnable encoder",
    bridgeLearnable,
    300
  );

  // ============================================================================
  // Comparison
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Comparison");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("| Encoder Type     | Avg Agency | Delta from Baseline |");
  console.log("|------------------|------------|---------------------|");

  const baseline = resultHandCrafted.avgAgency;

  console.log(
    `| Static (10D)     | ${(resultStatic.avgAgency * 100).toFixed(1).padStart(9)}% | ${((resultStatic.avgAgency - baseline) * 100).toFixed(1).padStart(18)}% |`
  );
  console.log(
    `| Hand-Crafted (4D) | ${(resultHandCrafted.avgAgency * 100).toFixed(1).padStart(9)}% | ${((resultHandCrafted.avgAgency - baseline) * 100).toFixed(1).padStart(18)}% |`
  );
  console.log(
    `| Learnable (10→4D) | ${(resultLearnable.avgAgency * 100).toFixed(1).padStart(9)}% | ${((resultLearnable.avgAgency - baseline) * 100).toFixed(1).padStart(18)}% |`
  );

  // Show encoder training statistics
  if (resultLearnable.encoderStats) {
    console.log("\n─────────────────────────────────────────────────────────");
    console.log("Learnable Encoder Training Statistics:");
    console.log("─────────────────────────────────────────────────────────");
    console.log(`  Total training steps: ${resultLearnable.encoderStats.trainSteps}`);
    console.log(
      `  Average binding loss: ${resultLearnable.encoderStats.avgBindingLoss.toFixed(4)}`
    );
    console.log(
      `  Average agency loss:  ${resultLearnable.encoderStats.avgAgencyLoss.toFixed(4)}`
    );
  }

  // ============================================================================
  // Interpretation
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Interpretation");
  console.log("═══════════════════════════════════════════════════════════\n");

  const learnableGap = Math.abs(
    resultLearnable.avgAgency - resultHandCrafted.avgAgency
  );
  const staticGap = Math.abs(
    resultStatic.avgAgency - resultHandCrafted.avgAgency
  );

  if (learnableGap < staticGap * 0.5) {
    console.log("  ✓ Learnable encoder successfully discovered relevant features");
    console.log("    Performance approaches hand-crafted baseline");
  } else if (learnableGap < staticGap) {
    console.log("  ⚠ Learnable encoder partially successful");
    console.log("    Some feature selection learned, but not optimal");
  } else {
    console.log("  ✗ Learnable encoder did not improve over static");
    console.log("    May need more training or better hyperparameters");
  }

  if (resultStatic.avgAgency < resultHandCrafted.avgAgency) {
    console.log("\n  ✓ Noise features hurt performance (as expected)");
    console.log("    Static encoder with all features performs worse");
  }

  // Cleanup
  resultStatic.bank.dispose();
  resultHandCrafted.bank.dispose();
  resultLearnable.bank.dispose();
  bridgeStatic.dispose();
  bridgeHandCrafted.dispose();
  bridgeLearnable.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
