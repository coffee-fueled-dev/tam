/**
 * Functor Inversion Encoder Learning Experiment
 *
 * Validates encoder learning via functor quality optimization.
 * This experiment demonstrates:
 * 1. Multi-port training with functor discovery
 * 2. Encoder training to improve functor systematicity
 * 3. Comparison of static vs learnable encoders
 *
 * Test domain: Simple 2D dynamics with multiple "modes"
 * - Mode A: Leftward drift (x -= 0.2)
 * - Mode B: Rightward drift (x += 0.2)
 * - Mode C: Upward drift (y += 0.2)
 * - Mode D: Downward drift (y -= 0.2)
 *
 * Each mode should get its own specialist port. Functors should relate:
 * - F(A→B): Reverses x-direction
 * - F(A→C): Changes from x-drift to y-drift
 * - etc.
 *
 * NOTE: Current implementation uses temporal smoothness as a proxy for functor quality
 * due to TensorFlow.js lacking tf.stopGradient(). True functor inversion would require:
 * - State-port association tracking
 * - Backprop through functor quality metrics
 * - Selective gradient blocking
 *
 * This experiment validates the architecture is sound and ready for when we can
 * implement true functor inversion (e.g., via PyTorch backend).
 */

import {
  GeometricPortBank,
  type Situation,
  type Transition,
  type GeometricPortConfigInput,
  IntraDomainEncoder,
  createEncoderBridge,
  type EncoderBridge,
} from "../../src";

// ============================================================================
// Domain: 2D Point with Multiple Drift Modes
// ============================================================================

type DriftMode = "leftward" | "rightward" | "upward" | "downward";

interface DriftState {
  x: number;
  y: number;
  mode: DriftMode;
  noise: number[]; // 4 noise features
}

/**
 * Simulate one step based on current mode
 */
function simulate(state: DriftState): DriftState {
  const drift = 0.2;
  const newState = { ...state };

  switch (state.mode) {
    case "leftward":
      newState.x = state.x - drift;
      break;
    case "rightward":
      newState.x = state.x + drift;
      break;
    case "upward":
      newState.y = state.y + drift;
      break;
    case "downward":
      newState.y = state.y - drift;
      break;
  }

  // Keep in bounds [-2, 2]
  newState.x = Math.max(-2, Math.min(2, newState.x));
  newState.y = Math.max(-2, Math.min(2, newState.y));

  // New noise each step
  newState.noise = Array.from({ length: 4 }, () => Math.random() - 0.5);

  return newState;
}

/**
 * Random initial state with specified mode
 */
function randomState(mode: DriftMode): DriftState {
  return {
    x: (Math.random() - 0.5) * 2,
    y: (Math.random() - 0.5) * 2,
    mode,
    noise: Array.from({ length: 4 }, () => Math.random() - 0.5),
  };
}

/**
 * Extract raw features (including mode as one-hot and noise)
 */
function extractRaw(state: DriftState): number[] {
  // Position + mode (one-hot) + noise
  const modeOneHot = {
    leftward: [1, 0, 0, 0],
    rightward: [0, 1, 0, 0],
    upward: [0, 0, 1, 0],
    downward: [0, 0, 0, 1],
  }[state.mode];

  return [state.x, state.y, ...modeOneHot, ...state.noise];
}

/**
 * Hand-crafted embedder (optimal - position + mode, no noise)
 */
function handCraftedEmbed(state: DriftState): number[] {
  const modeOneHot = {
    leftward: [1, 0, 0, 0],
    rightward: [0, 1, 0, 0],
    upward: [0, 0, 1, 0],
    downward: [0, 0, 0, 1],
  }[state.mode];

  return [state.x, state.y, ...modeOneHot];
}

// ============================================================================
// Training
// ============================================================================

async function trainWithEncoder(
  label: string,
  bridge: EncoderBridge<DriftState>,
  episodes: number = 500
): Promise<{
  bank: GeometricPortBank<DriftState, unknown>;
  avgAgency: number;
  portCount: number;
  functorStats: { totalProliferations: number; functorBased: number; randomBased: number };
  encoderStats?: ReturnType<typeof bridge.getEncoderStats>;
}> {
  console.log(`\nTraining ${label}...`);

  const config: GeometricPortConfigInput = {
    embeddingDim: 6, // 2 position + 4 mode features
    specialistThreshold: 0.7, // Lower threshold to encourage specialization
    enablePortFunctors: true, // Enable functor-based proliferation
    portFunctorTolerance: 0.3, // Error tolerance for functor discovery
    portFunctorMaxEpochs: 50, // Training epochs for functor
  };

  const bank = new GeometricPortBank(bridge.encoders, config);

  const modes: DriftMode[] = ["leftward", "rightward", "upward", "downward"];

  for (let i = 0; i < episodes; i++) {
    // Sample random mode
    const mode = modes[Math.floor(Math.random() * modes.length)]!;
    const before = randomState(mode);
    const after = simulate(before);

    const transition: Transition<DriftState, unknown> = {
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
    if ((i + 1) % 100 === 0) {
      bank.flush();
    }
  }

  bank.flush();

  // Count discovered ports
  const portCount = bank.getAllPorts().length;
  console.log(`  Discovered ${portCount} specialist ports`);

  // Get functor statistics
  const functorStats = bank.getPortFunctorStats();
  console.log(`  Functor-based proliferations: ${functorStats.functorBased}/${functorStats.totalProliferations}`);

  // Evaluate agency on test set (one sample per mode)
  let totalAgency = 0;
  const testSamplesPerMode = 5;

  for (const mode of modes) {
    for (let i = 0; i < testSamplesPerMode; i++) {
      const testState = randomState(mode);
      const sit: Situation<DriftState, unknown> = {
        state: testState,
        context: null,
      };

      const predictions = bank.predict("step", sit, 1);
      totalAgency += predictions[0]?.agency ?? 0;
    }
  }

  const avgAgency = totalAgency / (modes.length * testSamplesPerMode);
  console.log(`  Average agency: ${(avgAgency * 100).toFixed(1)}%`);

  return {
    bank,
    avgAgency,
    portCount,
    functorStats,
    encoderStats: bridge.getEncoderStats(),
  };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Functor Inversion Encoder Learning Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Domain: 2D Point with Drift Modes");
  console.log("  Relevant features: x, y, mode (one-hot 4D) = 6D");
  console.log("  Noise features: 4 random values (not predictive)");
  console.log("  Total raw dimension: 10D");
  console.log("  Expected: 4 specialist ports (one per mode)");

  // ============================================================================
  // Config A: Static Encoder (All Features Including Noise)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration A: Static Encoder (All 10 Features)");
  console.log("─────────────────────────────────────────────────────────");

  const bridgeStatic = createEncoderBridge<DriftState>({
    extractRaw,
  });

  const resultStatic = await trainWithEncoder(
    "static encoder",
    bridgeStatic,
    500
  );

  // ============================================================================
  // Config B: Hand-Crafted Encoder (Only Relevant Features)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration B: Hand-Crafted Encoder (6 Relevant Features)");
  console.log("─────────────────────────────────────────────────────────");

  const bridgeHandCrafted = createEncoderBridge<DriftState>({
    extractRaw,
    staticEmbedder: handCraftedEmbed,
  });

  const resultHandCrafted = await trainWithEncoder(
    "hand-crafted encoder",
    bridgeHandCrafted,
    500
  );

  // ============================================================================
  // Config C: Learnable Encoder (Functor Inversion)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration C: Learnable Encoder (Functor Inversion)");
  console.log("─────────────────────────────────────────────────────────");

  const learnableEncoder = new IntraDomainEncoder({
    rawDim: 10, // All features
    embeddingDim: 6, // Compress to relevant dimension
    hiddenSizes: [16, 12],
    learningRate: 0.001,
    agencyWeight: 0.1,
    smoothnessWeight: 0.05,
  });

  const bridgeLearnable = createEncoderBridge<DriftState>({
    extractRaw,
    learnableEncoder,
  });

  const resultLearnable = await trainWithEncoder(
    "learnable encoder",
    bridgeLearnable,
    500
  );

  // ============================================================================
  // Comparison
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Comparison");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("| Encoder Type      | Avg Agency | Ports | Functor-Based Prolif |");
  console.log("|-------------------|------------|-------|----------------------|");

  console.log(
    `| Static (10D)      | ${(resultStatic.avgAgency * 100).toFixed(1).padStart(9)}% | ${String(resultStatic.portCount).padStart(5)} | ${String(resultStatic.functorStats.functorBased).padStart(4)}/${String(resultStatic.functorStats.totalProliferations).padStart(4).padEnd(13)} |`
  );
  console.log(
    `| Hand-Crafted (6D) | ${(resultHandCrafted.avgAgency * 100).toFixed(1).padStart(9)}% | ${String(resultHandCrafted.portCount).padStart(5)} | ${String(resultHandCrafted.functorStats.functorBased).padStart(4)}/${String(resultHandCrafted.functorStats.totalProliferations).padStart(4).padEnd(13)} |`
  );
  console.log(
    `| Learnable (10→6D) | ${(resultLearnable.avgAgency * 100).toFixed(1).padStart(9)}% | ${String(resultLearnable.portCount).padStart(5)} | ${String(resultLearnable.functorStats.functorBased).padStart(4)}/${String(resultLearnable.functorStats.totalProliferations).padStart(4).padEnd(13)} |`
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

  // Check port specialization
  const expectedPorts = 4;
  if (resultHandCrafted.portCount >= expectedPorts) {
    console.log("  ✓ Baseline discovered specialist ports per mode");
  } else {
    console.log("  ⚠ Baseline did not fully specialize (may need more episodes)");
  }

  // Check functor discovery
  if (resultHandCrafted.functorStats.functorBased > 0) {
    console.log(`  ✓ Baseline used ${resultHandCrafted.functorStats.functorBased} functor-based proliferations`);
  } else {
    console.log("  ⚠ Baseline did not discover functors (may need more episodes or enable functors)");
  }

  // Compare learnable vs hand-crafted
  const learnableGap = Math.abs(
    resultLearnable.avgAgency - resultHandCrafted.avgAgency
  );
  const staticGap = Math.abs(
    resultStatic.avgAgency - resultHandCrafted.avgAgency
  );

  console.log("\n  Encoder Quality:");
  if (learnableGap < staticGap * 0.5) {
    console.log("    ✓ Learnable encoder successfully learned relevant features");
    console.log("      Performance approaches hand-crafted baseline");
  } else if (learnableGap < staticGap) {
    console.log("    ⚠ Learnable encoder partially successful");
    console.log("      Some feature selection learned, but not optimal");
  } else {
    console.log("    ✗ Learnable encoder did not improve over static");
    console.log("      May need more training or better hyperparameters");
  }

  // Compare port counts
  if (Math.abs(resultLearnable.portCount - resultHandCrafted.portCount) <= 1) {
    console.log("    ✓ Learnable encoder enabled similar specialization");
  } else if (resultLearnable.portCount < resultHandCrafted.portCount) {
    console.log("    ⚠ Learnable encoder resulted in fewer specialists");
    console.log("      May indicate suboptimal embedding space");
  } else {
    console.log("    ⚠ Learnable encoder resulted in more specialists");
    console.log("      May indicate over-fragmentation");
  }

  // ============================================================================
  // Current Limitations
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Current Limitations");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("  This experiment validates the architecture for functor inversion");
  console.log("  encoder learning, but uses temporal smoothness as a proxy loss.");
  console.log("");
  console.log("  True functor inversion would require:");
  console.log("    1. State-port association tracking");
  console.log("    2. Functor quality metrics");
  console.log("    3. Backprop through functor application");
  console.log("    4. tf.stopGradient() for alternating optimization");
  console.log("");
  console.log("  TensorFlow.js currently lacks #4, so we use smoothness loss.");
  console.log("  When porting to PyTorch, full functor inversion can be implemented.");

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
