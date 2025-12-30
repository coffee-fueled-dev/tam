/**
 * Port Functor Discovery Experiment
 *
 * Validates that port functor discovery works for structured proliferation.
 * Compares two configurations:
 * 1. Proliferation with random perturbation (baseline)
 * 2. Proliferation with functor discovery (hybrid approach)
 *
 * Test domain: 1D space with position-dependent dynamics
 * - Left side (x < 0): objects drift left (Δx ≈ -0.5)
 * - Right side (x > 0): objects drift right (Δx ≈ +0.5)
 *
 * This creates structured bimodality that functors should discover.
 */

import {
  GeometricPortBank,
  type Encoders,
  type Situation,
  type Transition,
  type GeometricPortConfigInput,
} from "../../src";

// ============================================================================
// Domain: Position-Dependent Drift
// ============================================================================

interface DriftState {
  x: number; // Position in 1D space
}

/**
 * Simulate one step: objects drift in the direction of their position.
 * - Left side (x < 0): drift left by ~0.5
 * - Right side (x > 0): drift right by ~0.5
 * - Near center: small drift
 */
function simulate(state: DriftState): DriftState {
  const direction = Math.sign(state.x); // -1, 0, or +1
  const magnitude = Math.abs(state.x) > 0.1 ? 0.5 : 0.1; // Stronger drift away from center
  const drift = direction * magnitude + (Math.random() - 0.5) * 0.1; // Small noise
  return { x: state.x + drift };
}

/**
 * Generate random state in [-2, 2]
 */
function randomState(): DriftState {
  return { x: (Math.random() - 0.5) * 4 };
}

/**
 * Encoders: simple 1D embedding
 */
const encoders: Encoders<DriftState, unknown> = {
  embedSituation: (sit) => [sit.state.x],
  delta: (before, after) => [after.state.x - before.state.x],
};

// ============================================================================
// Training
// ============================================================================

async function trainBank(
  config: GeometricPortConfigInput,
  episodes: number = 500
): Promise<GeometricPortBank<DriftState, unknown>> {
  const bank = new GeometricPortBank(encoders, config);

  for (let i = 0; i < episodes; i++) {
    const before = randomState();
    const after = simulate(before);

    const transition: Transition<DriftState, unknown> = {
      before: { state: before, context: null },
      after: { state: after, context: null },
      action: "drift",
    };

    await bank.observe(transition);

    // Periodic flush
    if (i % 50 === 0) {
      bank.flush();
    }
  }

  bank.flush();
  return bank;
}

// ============================================================================
// Evaluation
// ============================================================================

/**
 * Evaluate bank on test situations across the space.
 */
function evaluate(bank: GeometricPortBank<DriftState, unknown>): {
  avgAgency: number;
  leftAgency: number;
  rightAgency: number;
  portCount: number;
} {
  // Test points: left, center, right
  const testPoints = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];

  let totalAgency = 0;
  let leftAgency = 0;
  let rightAgency = 0;
  let leftCount = 0;
  let rightCount = 0;

  for (const x of testPoints) {
    const sit: Situation<DriftState, unknown> = {
      state: { x },
      context: null,
    };

    const predictions = bank.predict("drift", sit, 1);
    const agency = predictions[0]?.agency ?? 0;

    totalAgency += agency;

    if (x < 0) {
      leftAgency += agency;
      leftCount++;
    } else if (x > 0) {
      rightAgency += agency;
      rightCount++;
    }
  }

  return {
    avgAgency: totalAgency / testPoints.length,
    leftAgency: leftAgency / leftCount,
    rightAgency: rightAgency / rightCount,
    portCount: bank.getPortCountForAction("drift"),
  };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Port Functor Discovery Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Domain: Position-dependent drift");
  console.log("  Left side (x < 0): drift left by ~0.5");
  console.log("  Right side (x > 0): drift right by ~0.5");
  console.log("  Creates structured bimodality\n");

  // ============================================================================
  // Config A: Proliferation with Random Perturbation (Baseline)
  // ============================================================================

  console.log("─────────────────────────────────────────────────────────");
  console.log("Configuration A: Random Perturbation (Baseline)");
  console.log("─────────────────────────────────────────────────────────\n");

  const configA: GeometricPortConfigInput = {
    embeddingDim: 4,
    enableProliferation: true,
    proliferationAgencyThreshold: 0.6, // Higher threshold to trigger proliferation
    proliferationMinSamples: 30, // Lower sample requirement
    proliferationCooldown: 80,
    enablePortFunctors: false, // Random perturbation only
  };

  console.log("Training with random perturbation...");
  const bankA = await trainBank(configA, 500);

  const resultsA = evaluate(bankA);
  const statsA = bankA.getPortFunctorStats();

  console.log("\nResults:");
  console.log(`  Average agency:     ${(resultsA.avgAgency * 100).toFixed(1)}%`);
  console.log(`  Left-side agency:   ${(resultsA.leftAgency * 100).toFixed(1)}%`);
  console.log(`  Right-side agency:  ${(resultsA.rightAgency * 100).toFixed(1)}%`);
  console.log(`  Port count:         ${resultsA.portCount}`);
  console.log(`\nProliferation stats:`);
  console.log(`  Total:              ${statsA.totalProliferations}`);
  console.log(`  Functor-based:      ${statsA.functorBased}`);
  console.log(`  Random-based:       ${statsA.randomBased}`);

  // ============================================================================
  // Config B: Proliferation with Functor Discovery (Hybrid)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Configuration B: Functor Discovery (Hybrid)");
  console.log("─────────────────────────────────────────────────────────\n");

  const configB: GeometricPortConfigInput = {
    embeddingDim: 4,
    enableProliferation: true,
    proliferationAgencyThreshold: 0.6, // Higher threshold to trigger proliferation
    proliferationMinSamples: 30, // Lower sample requirement
    proliferationCooldown: 80,
    enablePortFunctors: true, // Enable functor discovery
    portFunctorTolerance: 0.4,
    portFunctorMaxEpochs: 50,
  };

  console.log("Training with functor discovery...");
  const bankB = await trainBank(configB, 500);

  const resultsB = evaluate(bankB);
  const statsB = bankB.getPortFunctorStats();

  console.log("\nResults:");
  console.log(`  Average agency:     ${(resultsB.avgAgency * 100).toFixed(1)}%`);
  console.log(`  Left-side agency:   ${(resultsB.leftAgency * 100).toFixed(1)}%`);
  console.log(`  Right-side agency:  ${(resultsB.rightAgency * 100).toFixed(1)}%`);
  console.log(`  Port count:         ${resultsB.portCount}`);
  console.log(`\nProliferation stats:`);
  console.log(`  Total:              ${statsB.totalProliferations}`);
  console.log(`  Functor-based:      ${statsB.functorBased}`);
  console.log(`  Random-based:       ${statsB.randomBased}`);

  // ============================================================================
  // Comparison
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Comparison");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("| Metric                  | Random (A) | Functor (B) | Delta    |");
  console.log("|-------------------------|------------|-------------|----------|");

  const deltaAgency = resultsB.avgAgency - resultsA.avgAgency;
  const deltaLeft = resultsB.leftAgency - resultsA.leftAgency;
  const deltaRight = resultsB.rightAgency - resultsA.rightAgency;
  const deltaPorts = resultsB.portCount - resultsA.portCount;

  console.log(
    `| Average Agency          | ${(resultsA.avgAgency * 100).toFixed(1).padStart(9)}% | ${(resultsB.avgAgency * 100).toFixed(1).padStart(10)}% | ${(deltaAgency > 0 ? "+" : "") + (deltaAgency * 100).toFixed(1).padStart(7)}% |`
  );
  console.log(
    `| Left-Side Agency        | ${(resultsA.leftAgency * 100).toFixed(1).padStart(9)}% | ${(resultsB.leftAgency * 100).toFixed(1).padStart(10)}% | ${(deltaLeft > 0 ? "+" : "") + (deltaLeft * 100).toFixed(1).padStart(7)}% |`
  );
  console.log(
    `| Right-Side Agency       | ${(resultsA.rightAgency * 100).toFixed(1).padStart(9)}% | ${(resultsB.rightAgency * 100).toFixed(1).padStart(10)}% | ${(deltaRight > 0 ? "+" : "") + (deltaRight * 100).toFixed(1).padStart(7)}% |`
  );
  console.log(
    `| Port Count              | ${resultsA.portCount.toString().padStart(10)} | ${resultsB.portCount.toString().padStart(11)} | ${(deltaPorts > 0 ? "+" : "") + deltaPorts.toString().padStart(8)} |`
  );

  console.log("\nFunctor Usage:");
  console.log(
    `  Config A (random):  ${statsA.totalProliferations} proliferations, ${statsA.functorBased} functor-based`
  );
  console.log(
    `  Config B (hybrid):  ${statsB.totalProliferations} proliferations, ${statsB.functorBased} functor-based`
  );

  // ============================================================================
  // Interpretation
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Interpretation");
  console.log("═══════════════════════════════════════════════════════════\n");

  if (resultsB.avgAgency > resultsA.avgAgency) {
    console.log("  ✓ Functor-based proliferation improves agency");
    console.log("    Ports specialize better when transforms are learned");
  } else {
    console.log("  ⚠ Functor-based proliferation shows similar agency");
    console.log("    Both approaches handle this domain equally well");
  }

  if (statsB.functorBased > 0) {
    const functorRate = (statsB.functorBased / statsB.totalProliferations) * 100;
    console.log(
      `\n  ✓ Functors discovered in ${functorRate.toFixed(0)}% of proliferations`
    );
    console.log("    System detected structured variance");
  } else {
    console.log("\n  ⚠ No functors discovered");
    console.log("    Variance may be too noisy or domain too simple");
  }

  if (resultsB.portCount < resultsA.portCount) {
    console.log("\n  ✓ Functor-based approach more efficient");
    console.log("    Fewer ports needed with systematic transforms");
  }

  // Cleanup
  bankA.dispose();
  bankB.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
