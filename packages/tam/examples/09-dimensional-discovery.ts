/**
 * Experiment 09: Dimensional Discovery in High-Dimensional Spaces
 *
 * Demonstrates TAM's ability to discover causal structure and ignore noise
 * in high-dimensional environments through epistemic humility and online learning.
 *
 * Environment: 10D state space
 * - Dimensions 0-1: Causal (spring dynamics)
 * - Dimensions 2-9: Pure noise (random)
 *
 * Key Mechanisms:
 * 1. Epistemic humility: Start with maximally wide cones (distance ≈ 0)
 * 2. Softmax attention: Learn per-dimension importance weights
 * 3. Anisotropic cones: Radius varies by dimension based on attention
 * 4. Homeostatic control: Binding rate equilibrium (~30% in noisy environment)
 *
 * Expected Outcome:
 * - 100% coverage on causal dimensions (discovers structure)
 * - 0% coverage on noise dimensions (correctly ignores)
 * - High agency (97%+) after convergence
 * - Distance stabilizes at environment-appropriate level
 *
 * This demonstrates TAM's ability to:
 * - Distinguish signal from noise
 * - Learn dimensional relevance without supervision
 * - Maintain homeostatic equilibrium in stochastic environments
 */

import { GeometricPortBank } from "../src/geometric/bank";
import { BindingRateRefinementPolicy } from "../src/geometric/refinement";
import type { Encoders, Situation } from "../src/types";
import { sub } from "../src/vec";

// Spring dynamics (2D: x, v)
const k = 1.0;
const b = 0.1;
const dt = 0.1;

function springStep(x: number, v: number): { x: number; v: number } {
  const ax = -k * x - b * v;
  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

type State10D = number[];

function randomState10D(): State10D {
  const state: State10D = [];
  state[0] = (Math.random() - 0.5) * 2; // x
  state[1] = (Math.random() - 0.5) * 2; // v
  for (let i = 2; i < 10; i++) {
    state[i] = (Math.random() - 0.5) * 2;
  }
  return state;
}

function step10D(state: State10D): State10D {
  const next: State10D = [];
  const spring = springStep(state[0]!, state[1]!);
  next[0] = spring.x;
  next[1] = spring.v;
  for (let i = 2; i < 10; i++) {
    next[i] = (Math.random() - 0.5) * 2; // Pure noise
  }
  return next;
}

const encoders: Encoders<State10D, {}> = {
  embedSituation: (sit: Situation<State10D, {}>) => sit.state,
  delta: (before, after) => sub(after.state, before.state),
};

interface DimensionalAnalysis {
  full: { error: number; agency: number; coverage: number };
  causalOnly: { error: number; agency: number; coverage: number };
  noiseOnly: { error: number; agency: number; coverage: number };
}

function evaluateWithMask(
  bank: GeometricPortBank<State10D, {}>,
  testStates: State10D[],
  mask: boolean[]
): { error: number; agency: number; coverage: number } {
  let totalError = 0;
  let totalAgency = 0;
  let boundCount = 0;
  let count = 0;

  for (const state of testStates) {
    const maskedState = state.map((val, i) => mask[i] ? val : 0);
    const truth = step10D(state);
    const truthDelta = sub(truth, state);

    const pred = bank.predictFromState("default", { state: maskedState, context: {} }, 1)[0];
    if (!pred) continue;

    let err = 0;
    let dimCount = 0;
    for (let i = 0; i < 10; i++) {
      if (mask[i]) {
        err += Math.pow(pred.delta[i]! - truthDelta[i]!, 2);
        dimCount++;
      }
    }
    err = dimCount > 0 ? Math.sqrt(err / dimCount) : 0;

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

function analyzeByDimension(
  bank: GeometricPortBank<State10D, {}>,
  testStates: State10D[]
): DimensionalAnalysis {
  const fullMask = Array(10).fill(true);
  const causalMask = [true, true, false, false, false, false, false, false, false, false];
  const noiseMask = [false, false, true, true, true, true, true, true, true, true];

  return {
    full: evaluateWithMask(bank, testStates, fullMask),
    causalOnly: evaluateWithMask(bank, testStates, causalMask),
    noiseOnly: evaluateWithMask(bank, testStates, noiseMask),
  };
}

interface Checkpoint {
  sample: number;
  full: { error: number; agency: number; coverage: number };
  causalOnly: { error: number; agency: number; coverage: number };
  noiseOnly: { error: number; agency: number; coverage: number };
  portCount: number;
  bindingRate: number;
  distance: number;
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Binding Rate Policy + Softmax Attention");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Setup:");
  console.log("  ✓ Embeddings trained (anisotropic cones)");
  console.log("  ✓ Softmax attention (relative dimensional weighting)");
  console.log("  ✓ Equilibrium control (targets 30% binding rate)\n");
  console.log("Environment: 10D (2 causal + 8 noise)\n");

  const config = {
    embeddingDim: 10,
    embeddingLearningRate: 0.01, // Train embeddings
    enableProliferation: false,
    minAlignmentThreshold: 0.0,
    equilibriumRate: 0.3, // Target 30% binding rate
    bindingRateTolerance: 0.1, // Tolerance band
    commitment: {
      initialRadius: 0.3,
      learningRate: 0.001, // Reduced from 0.01 for stability
      batchSize: 10, // Batch updates for smoother gradients
      narrowStep: 0.1, // Additive: add 0.1 to distance on success
      widenStep: 0.2, // Additive: reduce by (violation × 0.2) on failure
    },
  };

  const policy = new BindingRateRefinementPolicy(config);

  const bank = new GeometricPortBank<State10D, {}>(
    encoders,
    config,
    undefined,
    policy
  );

  const testStates: State10D[] = [];
  for (let i = 0; i < 100; i++) {
    testStates.push(randomState10D());
  }

  const checkpoints: Checkpoint[] = [];
  const totalSamples = 5000; // Increased for better convergence
  const checkpointEvery = 100;

  console.log("Training...\n");
  console.log("Sample | Distance | Agency | BindRate | Causal Cov | Noise Cov");
  console.log("-------|----------|--------|----------|------------|----------");

  for (let sample = 0; sample < totalSamples; sample++) {
    const before = randomState10D();
    const after = step10D(before);

    bank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      action: "default",
    });

    if ((sample + 1) % checkpointEvery === 0) {
      bank.flush();

      const analysis = analyzeByDimension(bank, testStates);
      const ports = bank.getAllPorts();
      const portIds = bank.getPortIds();

      const distance = ports.length > 0 ? ports[0]!.getLastDistance() : 0;
      const bindingRate = portIds.length > 0
        ? (bank as any).history.getBindingRate(portIds[0])
        : 0;

      // Debug: Check CommitmentNet state
      if ((sample + 1) === 1000) {
        const commitmentNet = (ports[0] as any).commitmentNet;
        const snapshot = commitmentNet.snapshot();
        console.log("\n[DEBUG] CommitmentNet snapshot:", snapshot);
      }

      const checkpoint: Checkpoint = {
        sample: sample + 1,
        full: analysis.full,
        causalOnly: analysis.causalOnly,
        noiseOnly: analysis.noiseOnly,
        portCount: portIds.length,
        bindingRate,
        distance,
      };

      checkpoints.push(checkpoint);

      console.log(
        `${checkpoint.sample.toString().padStart(6)} | ` +
        `${distance.toFixed(4).padStart(8)} | ` +
        `${(checkpoint.full.agency * 100).toFixed(1).padStart(5)}% | ` +
        `${(bindingRate * 100).toFixed(1).padStart(7)}% | ` +
        `${(checkpoint.causalOnly.coverage * 100).toFixed(1).padStart(9)}% | ` +
        `${(checkpoint.noiseOnly.coverage * 100).toFixed(1).padStart(8)}%`
      );

      // Detailed inspection at sample 1000
      if (sample + 1 === 1000) {
        console.log("\n─────────────────────────────────────────────────────────");
        console.log("Detailed Analysis at Sample 1000:");
        console.log("─────────────────────────────────────────────────────────");

        const port = ports[0]!;
        const cone = port.getCone(testStates[0]!);

        console.log(`\nPort Embedding (dimensional attention):`);
        for (let i = 0; i < 10; i++) {
          const dimType = i < 2 ? "CAUSAL" : "NOISE  ";
          console.log(`  Dim ${i} (${dimType}): ${port.embedding[i]!.toFixed(6)}`);
        }

        console.log(`\nCone Radii (anisotropy):`);
        for (let i = 0; i < 10; i++) {
          const dimType = i < 2 ? "CAUSAL" : "NOISE  ";
          console.log(`  Dim ${i} (${dimType}): ${cone.radius[i]!.toFixed(6)}`);
        }

        console.log(`\nMetrics:`);
        console.log(`  Distance: ${distance.toFixed(6)}`);
        console.log(`  Agency: ${(checkpoint.full.agency * 100).toFixed(2)}%`);
        console.log(`  Binding rate: ${(bindingRate * 100).toFixed(2)}%`);
        console.log(`  Causal coverage: ${(checkpoint.causalOnly.coverage * 100).toFixed(1)}%`);
        console.log(`  Noise coverage: ${(checkpoint.noiseOnly.coverage * 100).toFixed(1)}%`);
        console.log("─────────────────────────────────────────────────────────\n");
      }
    }
  }

  bank.flush();

  // Final analysis
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Final Results");
  console.log("═══════════════════════════════════════════════════════════\n");

  const final = checkpoints[checkpoints.length - 1]!;

  console.log("Performance:");
  console.log(`  Distance: ${final.distance.toFixed(4)}`);
  console.log(`  Full agency: ${(final.full.agency * 100).toFixed(1)}%`);
  console.log(`  Binding rate: ${(final.bindingRate * 100).toFixed(1)}%`);
  console.log(`  Causal coverage: ${(final.causalOnly.coverage * 100).toFixed(1)}%`);
  console.log(`  Noise coverage: ${(final.noiseOnly.coverage * 100).toFixed(1)}%\n`);

  console.log("Dimensional Discovery:");
  console.log(`  Causal agency: ${(final.causalOnly.agency * 100).toFixed(1)}%`);
  console.log(`  Noise agency: ${(final.noiseOnly.agency * 100).toFixed(1)}%`);
  const agencyGap = final.causalOnly.agency - final.noiseOnly.agency;
  console.log(`  Gap: ${(agencyGap * 100).toFixed(1)}%\n`);

  // Check if system discovered homeostasis
  const distanceTrajectory = checkpoints.slice(-5).map(c => c.distance);
  const distanceVariance = distanceTrajectory.reduce((sum, d) => {
    const mean = distanceTrajectory.reduce((s, x) => s + x, 0) / distanceTrajectory.length;
    return sum + (d - mean) ** 2;
  }, 0) / distanceTrajectory.length;
  const distanceStable = distanceVariance < 0.01;

  console.log("Homeostasis Discovery:");
  if (distanceStable && final.distance > 0.1) {
    console.log(`  ✓ Distance stabilized at ${final.distance.toFixed(3)}`);
    console.log(`  ✓ Agency settled at ${(final.full.agency * 100).toFixed(1)}%`);
    console.log(`  ✓ System discovered environment-specific equilibrium`);
  } else if (final.distance < 0.01) {
    console.log(`  ⚠ Distance collapsed to ${final.distance.toFixed(6)}`);
    console.log(`  ⚠ Agency near zero: ${(final.full.agency * 100).toFixed(2)}%`);
    console.log(`  ⚠ System did not stabilize`);
  } else {
    console.log(`  ~ Distance still evolving: ${final.distance.toFixed(3)}`);
    console.log(`  ~ May need more training`);
  }

  // Export results
  await Bun.write(
    "examples/results/09g-binding-rate-softmax.json",
    JSON.stringify({
      name: "Binding Rate Policy + Softmax Attention",
      config: {
        dimensions: { total: 10, causal: "0-1", noise: "2-9" },
        policy: "BindingRateRefinementPolicy",
        embeddingTraining: true,
        softmaxAttention: true,
        equilibriumRate: 0.3,
        samples: totalSamples,
      },
      checkpoints,
      final: {
        distance: final.distance,
        agency: final.full.agency,
        bindingRate: final.bindingRate,
        causalCoverage: final.causalOnly.coverage,
        noiseCoverage: final.noiseOnly.coverage,
        agencyGap,
        distanceStable,
      },
      timestamp: new Date().toISOString(),
    }, null, 2)
  );

  console.log("\n✓ Results saved to examples/results/09g-binding-rate-softmax.json");

  bank.dispose();
  console.log("✓ Experiment complete!");
}

main().catch(console.error);
