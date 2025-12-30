/**
 * Multi-Horizon Prediction: Addressing Long-Horizon Rollout Failure
 *
 * CONCEPT: Instead of training only on one-step transitions s_t → s_{t+1},
 * train on multiple horizons simultaneously:
 *   - 1-step:  s_t → s_{t+1}   (precise, short-term)
 *   - 5-step:  s_t → s_{t+5}   (tactical, medium-term)
 *   - 20-step: s_t → s_{t+20}  (strategic, long-term)
 *
 * WHY THIS HELPS:
 * 1. Long-horizon ports learn to compensate for error accumulation
 * 2. Different specialists for different time scales
 * 3. Agency selects appropriate horizon for current state
 * 4. Can discover non-Markovian structure (momentum, delayed effects)
 * 5. Enables compositional planning: Chain or direct?
 *
 * TAM'S UNIQUE ADVANTAGE:
 * Unlike standard models that must choose a single horizon, TAM naturally
 * supports multiple horizons as different ports, with agency-based selection
 * and potential functor relationships between them.
 *
 * IMPLEMENTATION STATUS: Sketch / Proof-of-concept
 * This demonstrates the API design and expected behavior.
 * Full implementation would require:
 * - Multi-horizon transition collection
 * - Separate port banks per horizon (or multi-action bank)
 * - Horizon-aware prediction interface
 */

import {
  GeometricPortBank,
  type Situation,
  type Transition,
  type GeometricPortConfigInput,
  createEncoderBridge,
} from "../../src";
import { norm, sub } from "../../src/vec";

// ============================================================================
// Domain: 1D Damped Spring (same as before)
// ============================================================================

interface SpringState {
  x: number;
  v: number;
}

const k = 1.0;
const b = 0.1;
const dt = 0.1;

function simulate(state: SpringState): SpringState {
  const ax = -k * state.x - b * state.v;
  const newV = state.v + ax * dt;
  const newX = state.x + newV * dt;
  return { x: newX, v: newV };
}

function simulateKSteps(state: SpringState, k: number): SpringState {
  let current = state;
  for (let i = 0; i < k; i++) {
    current = simulate(current);
  }
  return current;
}

function randomState(xRange: [number, number], vRange: [number, number]): SpringState {
  const x = xRange[0] + Math.random() * (xRange[1] - xRange[0]);
  const v = vRange[0] + Math.random() * (vRange[1] - vRange[0]);
  return { x, v };
}

function embedState(state: SpringState): number[] {
  return [state.x, state.v];
}

// ============================================================================
// Multi-Horizon Training
// ============================================================================

/**
 * Train separate banks for different horizons.
 * Each horizon has its own specialist ports that learn the k-step dynamics.
 */
async function trainMultiHorizon(
  episodes: number,
  horizons: number[] = [1, 5, 10, 20]
): Promise<Map<number, GeometricPortBank<SpringState, unknown>>> {
  const bridge = createEncoderBridge<SpringState>({ extractRaw: embedState });
  const config: GeometricPortConfigInput = { embeddingDim: 2 };

  // Create separate bank for each horizon
  const banks = new Map<number, GeometricPortBank<SpringState, unknown>>();
  for (const h of horizons) {
    banks.set(h, new GeometricPortBank(bridge.encoders, config));
  }

  console.log(`Training multi-horizon banks for horizons: ${horizons.join(", ")}`);

  for (let i = 0; i < episodes; i++) {
    const before = randomState([-1, 1], [-1, 1]);

    // Train each horizon with appropriate k-step transition
    for (const h of horizons) {
      const after = simulateKSteps(before, h);

      const transition: Transition<SpringState, unknown> = {
        before: { state: before, context: null },
        after: { state: after, context: null },
        action: `step_${h}`, // Different action per horizon
      };

      await banks.get(h)!.observe(transition);
    }

    if ((i + 1) % 50 === 0) {
      for (const bank of banks.values()) {
        bank.flush();
      }
    }

    if ((i + 1) % 100 === 0) {
      console.log(`  [${i + 1}/${episodes}] Training...`);
    }
  }

  for (const bank of banks.values()) {
    bank.flush();
  }

  return banks;
}

// ============================================================================
// Evaluation: Compare Approaches
// ============================================================================

/**
 * Evaluate rollout error for different prediction strategies.
 */
async function evaluateRollout(
  banks: Map<number, GeometricPortBank<SpringState, unknown>>,
  testStates: SpringState[],
  targetHorizon: number
) {
  const horizons = Array.from(banks.keys()).sort((a, b) => a - b);

  console.log(`\nEvaluating ${targetHorizon}-step rollout on ${testStates.length} test states:\n`);

  // Strategy 1: Iterative 1-step (baseline)
  const bank1 = banks.get(1);
  if (bank1) {
    let totalError = 0;
    for (const state of testStates) {
      let current = state;
      for (let step = 0; step < targetHorizon; step++) {
        const sit: Situation<SpringState, unknown> = { state: current, context: null };
        const pred = bank1.predict("step_1", sit, 1)[0];
        if (!pred) break;
        current = {
          x: current.x + pred.delta[0]!,
          v: current.v + pred.delta[1]!,
        };
      }
      const actual = simulateKSteps(state, targetHorizon);
      const error = Math.sqrt(
        (current.x - actual.x) ** 2 + (current.v - actual.v) ** 2
      );
      totalError += error;
    }
    const avgError = totalError / testStates.length;
    console.log(`Strategy 1: Iterative 1-step predictions`);
    console.log(`  Error: ${avgError.toFixed(4)}`);
    console.log(`  Method: Chain ${targetHorizon} × 1-step predictions`);
    console.log(`  Problem: Error compounds at each step\n`);
  }

  // Strategy 2: Direct k-step (if trained for this horizon)
  const bankK = banks.get(targetHorizon);
  if (bankK) {
    let totalError = 0;
    let totalAgency = 0;
    for (const state of testStates) {
      const sit: Situation<SpringState, unknown> = { state, context: null };
      const pred = bankK.predict(`step_${targetHorizon}`, sit, 1)[0];
      if (!pred) continue;
      const predicted = {
        x: state.x + pred.delta[0]!,
        v: state.v + pred.delta[1]!,
      };
      const actual = simulateKSteps(state, targetHorizon);
      const error = Math.sqrt(
        (predicted.x - actual.x) ** 2 + (predicted.v - actual.v) ** 2
      );
      totalError += error;
      totalAgency += pred.agency;
    }
    const avgError = totalError / testStates.length;
    const avgAgency = totalAgency / testStates.length;
    console.log(`Strategy 2: Direct ${targetHorizon}-step prediction`);
    console.log(`  Error: ${avgError.toFixed(4)}`);
    console.log(`  Agency: ${(avgAgency * 100).toFixed(1)}%`);
    console.log(`  Method: Single ${targetHorizon}-step port`);
    console.log(`  Advantage: Learns to compensate for dynamics\n`);
  }

  // Strategy 3: Adaptive horizon (use agency to select)
  const allBanks = Array.from(banks.entries());
  if (allBanks.length > 1) {
    let totalError = 0;
    let totalAgency = 0;
    const horizonCounts = new Map<number, number>();

    for (const state of testStates) {
      // Try each horizon, pick highest agency
      let bestHorizon = 1;
      let bestAgency = 0;
      let bestPred: { delta: number[]; agency: number } | null = null;

      for (const [h, bank] of banks) {
        if (h > targetHorizon) continue; // Can't use longer horizon than target
        const sit: Situation<SpringState, unknown> = { state, context: null };
        const pred = bank.predict(`step_${h}`, sit, 1)[0];
        if (pred && pred.agency > bestAgency) {
          bestAgency = pred.agency;
          bestHorizon = h;
          bestPred = pred;
        }
      }

      if (bestPred) {
        // If selected horizon < target, need to chain predictions
        let current = state;
        let stepsRemaining = targetHorizon;
        while (stepsRemaining > 0 && bestPred) {
          const h = Math.min(bestHorizon, stepsRemaining);
          current = {
            x: current.x + bestPred.delta[0]!,
            v: current.v + bestPred.delta[1]!,
          };
          stepsRemaining -= h;

          if (stepsRemaining > 0) {
            const sit: Situation<SpringState, unknown> = { state: current, context: null };
            bestPred = banks.get(h)!.predict(`step_${h}`, sit, 1)[0] || null;
          }
        }

        const actual = simulateKSteps(state, targetHorizon);
        const error = Math.sqrt(
          (current.x - actual.x) ** 2 + (current.v - actual.v) ** 2
        );
        totalError += error;
        totalAgency += bestAgency;
        horizonCounts.set(bestHorizon, (horizonCounts.get(bestHorizon) || 0) + 1);
      }
    }

    const avgError = totalError / testStates.length;
    const avgAgency = totalAgency / testStates.length;
    console.log(`Strategy 3: Adaptive horizon (agency-based selection)`);
    console.log(`  Error: ${avgError.toFixed(4)}`);
    console.log(`  Agency: ${(avgAgency * 100).toFixed(1)}%`);
    console.log(`  Method: Select horizon based on confidence`);
    console.log(`  Horizon usage:`);
    for (const [h, count] of Array.from(horizonCounts.entries()).sort((a, b) => a[0] - b[0])) {
      console.log(`    ${h}-step: ${count} times (${(100 * count / testStates.length).toFixed(0)}%)`);
    }
  }
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Multi-Horizon Prediction: Concept Demonstration");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Problem: Long-horizon rollouts fail due to error compounding\n");
  console.log("Solution: Train separate specialists for different horizons\n");
  console.log("TAM Advantage: Multiple ports + Agency-based selection\n");

  // Train multi-horizon banks
  const banks = await trainMultiHorizon(500, [1, 5, 10, 20]);

  // Generate test states
  const testStates = Array.from({ length: 50 }, () =>
    randomState([-1, 1], [-1, 1])
  );

  // Evaluate different target horizons
  for (const target of [5, 10, 20]) {
    console.log("\n" + "─".repeat(61));
    await evaluateRollout(banks, testStates, target);
  }

  // Cleanup
  for (const bank of banks.values()) {
    bank.dispose();
  }

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Insights");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Expected Results:");
  console.log("  1. Direct k-step should outperform iterative 1-step");
  console.log("     (Learns actual k-step dynamics, not error accumulation)\n");

  console.log("  2. Adaptive horizon should be competitive");
  console.log("     (Uses agency to select reliable predictions)\n");

  console.log("  3. Different horizons have different agency patterns");
  console.log("     (Short-term high agency everywhere, long-term selective)\n");

  console.log("TAM-Specific Benefits:");
  console.log("  • Multiple time scales as different specialists");
  console.log("  • Agency naturally selects appropriate horizon");
  console.log("  • Can discover functor relationships: F(short) → long?");
  console.log("  • Compositional planning: Chain when needed");
  console.log("  • Adaptive replanning: Replan when long-term fails\n");

  console.log("Next Steps:");
  console.log("  • Implement hierarchical planning with horizon selection");
  console.log("  • Discover functors between horizons (temporal composition)");
  console.log("  • Learn when to replan (binding violation on long-term)");
  console.log("  • Apply to non-Markovian domains (momentum, delayed effects)");

  console.log("\n✓ Concept demonstration complete!");
}

main().catch(console.error);
