/**
 * Experiment 05d: Composition for Online Adaptation
 *
 * THE CORRECT TEST OF COMPOSITION!
 *
 * Key insight: TAM is naturally online. Composition isn't about frozen reuse,
 * it's about ACCELERATING ADAPTATION to new domains by bootstrapping from
 * related actors.
 *
 * Question: Does composition speed up learning in a new region compared to
 * training from scratch?
 *
 * Setup:
 * 1. Train left & right actors on their regions (stiff/soft springs)
 * 2. Create composed actor that can leverage both via functors
 * 3. CONTINUE TRAINING in the gap region (online adaptation)
 * 4. Compare learning curves:
 *    - Composed (bootstrapped from primitives)
 *    - From scratch (no prior knowledge)
 *
 * Expected: Composed actor should learn faster in gap due to transfer
 * from related domains through functors.
 */

import { TAM, type DomainSpec } from "../src";

// Discontinuous spring dynamics
function piecewiseSpringDynamics(x: number, v: number, dt: number): { x: number; v: number } {
  let k: number, b: number;

  if (x < -0.3) {
    k = 2.0;  // Stiff
    b = 0.3;
  } else if (x > 0.3) {
    k = 0.5;  // Soft
    b = 0.1;
  } else {
    // Gap: Blend
    const blend = (x + 0.3) / 0.6;
    k = 2.0 + blend * (0.5 - 2.0);
    b = 0.3 + blend * (0.1 - 0.3);
  }

  const ax = -k * x - b * v;
  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

function sampleState(xMin: number, xMax: number): { x: number; v: number } {
  const x = xMin + Math.random() * (xMax - xMin);
  const v = -0.5 + Math.random() * 1.0;
  return { x, v };
}

interface TrainingCurve {
  epoch: number;
  error: number;
  portCount: number;
}

async function evaluateError(
  bank: any,
  testStates: Array<{ x: number; v: number }>,
  dt: number
): Promise<number> {
  let totalError = 0;
  let count = 0;

  for (const state of testStates) {
    const truth = piecewiseSpringDynamics(state.x, state.v, dt);
    const truthDelta = [truth.x - state.x, truth.v - state.v];

    const pred = bank.predictFromState("default", { state, context: {} }, 1)[0];
    if (!pred) continue;

    const err = Math.sqrt(
      Math.pow(pred.delta[0]! - truthDelta[0]!, 2) +
      Math.pow(pred.delta[1]! - truthDelta[1]!, 2)
    );

    totalError += err;
    count++;
  }

  return count > 0 ? totalError / count : 1.0;
}

async function trainOnce(
  tam: TAM,
  domainName: string,
  domain: DomainSpec<{ x: number; v: number }> | any,
  totalEpochs: number,
  testStates: Array<{ x: number; v: number }>,
  dt: number,
  label: string
): Promise<{ bank: any; error: number; portCount: number }> {
  console.log(`\n${label}:`);
  console.log(`  Training for ${totalEpochs} epochs...`);

  // Train once for full duration
  const bank = await tam.learn(domainName, domain, {
    epochs: totalEpochs,
    samplesPerEpoch: 50,
    flushFrequency: 20,
  });

  const error = await evaluateError(bank, testStates, dt);
  const portCount = bank.getPortIds().length;

  console.log(`  ✓ Complete: Error ${error.toFixed(4)} | Ports ${portCount}`);

  return { bank, error, portCount };
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Composition for Online Adaptation");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("THE CORRECT TEST: Does composition accelerate learning?\n");
  console.log("Dynamics: Piecewise Spring");
  console.log("  Left  (x < -0.3): Stiff spring (k=2.0)");
  console.log("  Right (x > 0.3):  Soft spring (k=0.5)");
  console.log("  Gap   [-0.3,0.3]: Transition - TARGET for online learning\n");
  console.log("Strategy:");
  console.log("  1. Train primitives on left/right");
  console.log("  2. Discover functors (composition infrastructure)");
  console.log("  3. Train NEW domain on gap with composition available");
  console.log("  4. Compare: Composed (bootstrapped) vs From-scratch\n");

  const dt = 0.1;

  // ============================================================================
  // Phase 1: Train Primitive Actors
  // ============================================================================

  console.log("─────────────────────────────────────────────────────────");
  console.log("Phase 1: Training Primitive Actors");
  console.log("─────────────────────────────────────────────────────────");

  const tamPrimitives = new TAM();

  const leftDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(-1.0, -0.3),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  const rightDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(0.3, 1.0),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  console.log("\nTraining Left Actor (Stiff Spring)...");
  const bankLeft = await tamPrimitives.learn("left-stiff", leftDomain, {
    epochs: 200,
    samplesPerEpoch: 50,
    flushFrequency: 20,
  });
  console.log(`✓ Left Actor: ${bankLeft.getPortIds().length} ports`);

  console.log("\nTraining Right Actor (Soft Spring)...");
  const bankRight = await tamPrimitives.learn("right-soft", rightDomain, {
    epochs: 200,
    samplesPerEpoch: 50,
    flushFrequency: 20,
  });
  console.log(`✓ Right Actor: ${bankRight.getPortIds().length} ports`);

  // ============================================================================
  // Phase 2: Discover Functors (Composition Infrastructure)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Phase 2: Functor Discovery");
  console.log("─────────────────────────────────────────────────────────\n");

  const pathLtoR = await tamPrimitives.findPath("left-stiff", "right-soft", 1);
  const pathRtoL = await tamPrimitives.findPath("right-soft", "left-stiff", 1);

  if (pathLtoR && pathRtoL) {
    console.log("✓ Functors discovered between left ↔ right");
    console.log(`  L→R: ${(pathLtoR.totalBindingRate * 100).toFixed(1)}% binding`);
    console.log(`  R→L: ${(pathRtoL.totalBindingRate * 100).toFixed(1)}% binding`);
  } else {
    console.log("⚠ Functor discovery failed - composition may not help");
  }

  // ============================================================================
  // Phase 3: Online Adaptation in Gap Region
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Phase 3: Online Learning in Gap Region");
  console.log("─────────────────────────────────────────────────────────");
  console.log("\nTarget: Gap region [-0.3, 0.3] - UNSEEN by primitives");
  console.log("Training: 200 epochs with 20-epoch checkpoints\n");

  // Test states in gap region
  const testStatesGap: Array<{ x: number; v: number }> = [];
  for (let i = 0; i < 100; i++) {
    testStatesGap.push(sampleState(-0.3, 0.3));
  }

  const gapDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(-0.3, 0.3),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  // Approach A: Bootstrap from composition
  console.log("Approach A: Composed (bootstrap from left+right via functors)");
  const tamComposed = new TAM();

  // Register primitives in composed TAM
  await tamComposed.learn("left-stiff", leftDomain, { epochs: 1, samplesPerEpoch: 1 });
  await tamComposed.learn("right-soft", rightDomain, { epochs: 1, samplesPerEpoch: 1 });

  // Discover functors in composed TAM
  await tamComposed.findPath("left-stiff", "right-soft", 1);
  await tamComposed.findPath("right-soft", "left-stiff", 1);

  // Now train gap domain WITH composition available
  const resultComposed = await trainOnce(
    tamComposed,
    "gap-composed",
    {
      ...gapDomain,
      // This is the key: specify composition
      stateToRaw: (state) => [state.x, state.v],
      rawDim: 2,
      composeWith: ["left-stiff", "right-soft"],
    } as any,
    200,
    testStatesGap,
    dt,
    "  Composed"
  );

  // Approach B: From scratch (no composition)
  console.log("\nApproach B: From Scratch (no prior knowledge)");
  const tamScratch = new TAM();
  const resultScratch = await trainOnce(
    tamScratch,
    "gap-scratch",
    gapDomain,
    200,
    testStatesGap,
    dt,
    "  From-Scratch"
  );

  // ============================================================================
  // Analysis
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis: Final Performance Comparison");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Final Performance (Gap Region):");
  console.log(`  Composed:     ${resultComposed.error.toFixed(4)} | ${resultComposed.portCount} ports`);
  console.log(`  From-Scratch: ${resultScratch.error.toFixed(4)} | ${resultScratch.portCount} ports\n`);

  const improvement = ((resultScratch.error - resultComposed.error) / resultScratch.error * 100);

  if (improvement > 10) {
    console.log(`✓ Composition significantly improves performance by ${improvement.toFixed(1)}%`);
    console.log("  Transfer learning through functors helps adaptation!");
  } else if (improvement > 0) {
    console.log(`≈ Composition provides modest benefit: ${improvement.toFixed(1)}%`);
    console.log("  Some transfer, but primitives may not be optimal match.");
  } else {
    console.log(`⚠ Composition does not improve performance (${improvement.toFixed(1)}%)`);
    console.log("  Primitives may introduce bias or functors not reliable.");
  }

  // ============================================================================
  // Export Results
  // ============================================================================

  const exportData = {
    name: "Composition for Online Adaptation",
    config: {
      domain: "Piecewise Spring (k1=2.0, k2=0.5)",
      primitiveEpochs: 200,
      gapEpochs: 200,
      actors: {
        left: "Stiff spring (x < -0.3)",
        right: "Soft spring (x > 0.3)",
        gap: "Transition [-0.3, 0.3] - TARGET",
      },
      functors: {
        leftToRight: pathLtoR ? { found: true, binding: pathLtoR.totalBindingRate } : { found: false },
        rightToLeft: pathRtoL ? { found: true, binding: pathRtoL.totalBindingRate } : { found: false },
      },
    },
    results: {
      composed: {
        error: resultComposed.error,
        portCount: resultComposed.portCount,
      },
      fromScratch: {
        error: resultScratch.error,
        portCount: resultScratch.portCount,
      },
      improvement: improvement,
    },
    summary: {
      conclusion: improvement > 10
        ? "Composition significantly improves performance"
        : improvement > 0
        ? "Composition provides modest benefit"
        : "Composition does not improve performance",
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write(
    "examples/results/05d-composition-online-adaptation.json",
    JSON.stringify(exportData, null, 2)
  );
  console.log("\n✓ Results saved to examples/results/05d-composition-online-adaptation.json");

  // Cleanup
  tamPrimitives.dispose();
  tamComposed.dispose();
  tamScratch.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
