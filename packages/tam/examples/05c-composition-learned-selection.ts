/**
 * Experiment 05c: Composition with Learned Port Selection
 *
 * Tests whether TAM-based meta-learning can improve composition by learning
 * WHEN to trust different ports/functors vs raw agency values.
 *
 * Key improvement over 05b:
 * - Use TAM-based port selection instead of naive MaxAgency
 * - Meta-learner should discover that left actor's high agency in gap is misleading
 * - More training cycles (300 epochs) to ensure convergence
 *
 * Domain: Piecewise Spring with Boundary Discontinuity
 * - Left region (x < -0.3): Stiff spring (k=2.0, b=0.3)
 * - Right region (x > 0.3): Soft spring (k=0.5, b=0.1)
 * - Gap region [-0.3, 0.3]: Transition zone (NEVER SEEN)
 */

import { TAM, type DomainSpec } from "../src";
import type { Vec } from "../src/types";

// Discontinuous spring dynamics
function piecewiseSpringDynamics(x: number, v: number, dt: number): { x: number; v: number } {
  let k: number, b: number;

  if (x < -0.3) {
    // Left: Stiff spring
    k = 2.0;
    b = 0.3;
  } else if (x > 0.3) {
    // Right: Soft spring
    k = 0.5;
    b = 0.1;
  } else {
    // Gap: Blend (but actors never see this!)
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

interface RegionResults {
  left: { error: number; count: number; agency?: number };
  gap: { error: number; count: number; agency?: number };
  right: { error: number; count: number; agency?: number };
}

function evaluateByRegion(
  predictFn: (state: { x: number; v: number }) => { delta: number[]; agency?: number },
  testStates: Array<{ x: number; v: number }>,
  label: string
): RegionResults {
  const results: RegionResults = {
    left: { error: 0, count: 0, agency: 0 },
    gap: { error: 0, count: 0, agency: 0 },
    right: { error: 0, count: 0, agency: 0 },
  };

  const dt = 0.1;

  for (const state of testStates) {
    const truth = piecewiseSpringDynamics(state.x, state.v, dt);
    const truthDelta = [truth.x - state.x, truth.v - state.v];

    const pred = predictFn(state);
    const err = Math.sqrt(
      Math.pow(pred.delta[0]! - truthDelta[0]!, 2) +
      Math.pow(pred.delta[1]! - truthDelta[1]!, 2)
    );

    let region: "left" | "gap" | "right";
    if (state.x < -0.3) region = "left";
    else if (state.x > 0.3) region = "right";
    else region = "gap";

    results[region].error += err;
    results[region].count++;
    results[region].agency! += pred.agency || 0;
  }

  for (const region of ["left", "gap", "right"] as const) {
    if (results[region].count > 0) {
      results[region].error /= results[region].count;
      results[region].agency! /= results[region].count;
    }
  }

  console.log(`\n${label}:`);
  console.log(`  Left  (x<-0.3):   Error ${results.left.error.toFixed(4)} | Agency ${((results.left.agency || 0) * 100).toFixed(1)}% | n=${results.left.count}`);
  console.log(`  Gap   [-0.3,0.3]: Error ${results.gap.error.toFixed(4)} | Agency ${((results.gap.agency || 0) * 100).toFixed(1)}% | n=${results.gap.count}`);
  console.log(`  Right (x>0.3):    Error ${results.right.error.toFixed(4)} | Agency ${((results.right.agency || 0) * 100).toFixed(1)}% | n=${results.right.count}`);

  return results;
}

async function trainActor(
  tam: TAM,
  domainName: string,
  domain: DomainSpec<{ x: number; v: number }>,
  epochs: number
): Promise<any> {
  const bank = await tam.learn(domainName, domain, {
    epochs,
    samplesPerEpoch: 50,
    flushFrequency: 20,
  });
  return bank;
}

// Selection state for meta-learner
interface SelectionState {
  stateEmb: Vec;
  candidates: Array<{ portId: string; agency: number }>;
  candidateIndex: number;
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Composition with Learned Port Selection");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Question: Can meta-learning improve composition?\n");
  console.log("Dynamics: Piecewise Spring");
  console.log("  Left  (x < -0.3): Stiff spring (k=2.0, b=0.3)");
  console.log("  Right (x > 0.3):  Soft spring (k=0.5, b=0.1)");
  console.log("  Gap   [-0.3,0.3]: Transition (NEVER SEEN)\n");
  console.log("Key improvement: TAM-based port selection");
  console.log("  - Meta-learner discovers when to trust each actor");
  console.log("  - Should correct overconfident agency in gap\n");

  const tam = new TAM();
  const dt = 0.1;

  // Domains
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

  const fullDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(-1.0, 1.0),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  // Train actors (more epochs for convergence)
  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 1: Learning Left Actor (Stiff Spring) - 300 epochs");
  console.log("─────────────────────────────────────────────────────────");
  const bankLeft = await trainActor(tam, "left-stiff", leftDomain, 300);
  console.log(`  ✓ Trained: ${bankLeft.getPortIds().length} ports\n`);

  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 2: Learning Right Actor (Soft Spring) - 300 epochs");
  console.log("─────────────────────────────────────────────────────────");
  const bankRight = await trainActor(tam, "right-soft", rightDomain, 300);
  console.log(`  ✓ Trained: ${bankRight.getPortIds().length} ports\n`);

  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 3: Baseline (Full Range) - 300 epochs");
  console.log("─────────────────────────────────────────────────────────");
  const bankBaseline = await trainActor(tam, "full-range", fullDomain, 300);
  console.log(`  ✓ Trained: ${bankBaseline.getPortIds().length} ports\n`);

  // Functor discovery
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Functor Discovery");
  console.log("═══════════════════════════════════════════════════════════\n");

  const pathLeftToRight = await tam.findPath("left-stiff", "right-soft", 1);
  const pathRightToLeft = await tam.findPath("right-soft", "left-stiff", 1);

  if (pathLeftToRight) {
    console.log(`✓ Found functor: left → right (${(pathLeftToRight.totalBindingRate * 100).toFixed(1)}%)`);
  }
  if (pathRightToLeft) {
    console.log(`✓ Found functor: right → left (${(pathRightToLeft.totalBindingRate * 100).toFixed(1)}%)`);
  }
  console.log();

  // Generate test states
  const testStates: Array<{ x: number; v: number }> = [];
  for (let i = 0; i < 150; i++) {
    testStates.push(sampleState(-1.0, 1.0));
  }

  // Evaluate with standard MaxAgency selection
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Baseline Performance (MaxAgency Selection)");
  console.log("═══════════════════════════════════════════════════════════");

  const regionResultsLeft = evaluateByRegion(
    (state) => {
      const pred = bankLeft.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Left Actor"
  );

  const regionResultsRight = evaluateByRegion(
    (state) => {
      const pred = bankRight.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Right Actor"
  );

  const regionResultsBase = evaluateByRegion(
    (state) => {
      const pred = bankBaseline.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Baseline"
  );

  // Train meta-learner for port selection
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Training Meta-Learner (TAM-based Selection)");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Training selection TAM to predict binding success...");
  console.log("Episodes: 500 (allowing meta-learner to converge)\n");

  // Create selection domain
  const selectionTAM = new TAM();
  const selectionDomain: DomainSpec<SelectionState> = {
    randomState: () => {
      // Sample random state from full range
      const state = sampleState(-1.0, 1.0);
      const stateEmb = [state.x, state.v];

      // Get predictions from both actors
      const predLeft = bankLeft.predictFromState("default", { state, context: {} }, 1)[0];
      const predRight = bankRight.predictFromState("default", { state, context: {} }, 1)[0];

      const candidates = [
        { portId: "left", agency: predLeft?.agency || 0 },
        { portId: "right", agency: predRight?.agency || 0 },
      ];

      const candidateIndex = Math.floor(Math.random() * candidates.length);

      return { stateEmb, candidates, candidateIndex };
    },

    simulate: (selState) => {
      // Reconstruct state from embedding
      const state = { x: selState.stateEmb[0]!, v: selState.stateEmb[1]! };

      // Simulate one step
      const nextState = piecewiseSpringDynamics(state.x, state.v, dt);

      // Get selected candidate's prediction
      const selectedCandidate = selState.candidates[selState.candidateIndex]!;
      let prediction;

      if (selectedCandidate.portId === "left") {
        prediction = bankLeft.predictFromState("default", { state, context: {} }, 1)[0];
      } else {
        prediction = bankRight.predictFromState("default", { state, context: {} }, 1)[0];
      }

      if (!prediction) {
        return selState; // No change if prediction fails
      }

      // Apply delta to get predicted next state
      const predictedNext = {
        x: state.x + prediction.delta[0]!,
        v: state.v + prediction.delta[1]!,
      };

      // Compute binding (how accurate was the prediction?)
      const error = Math.sqrt(
        Math.pow(predictedNext.x - nextState.x, 2) +
        Math.pow(predictedNext.v - nextState.v, 2)
      );

      const binding = Math.max(0, 1 - error * 10); // Convert error to [0,1] binding

      // Return next selection state with same structure but updated embedding
      return {
        stateEmb: [nextState.x, nextState.v],
        candidates: selState.candidates,
        candidateIndex: selState.candidateIndex,
      };
    },

    embedder: (selState) => {
      // Embed: [normalized state (2D), candidate features (4D)]
      const stateNorm = Math.sqrt(selState.stateEmb[0]! ** 2 + selState.stateEmb[1]! ** 2);
      const normState = stateNorm > 1e-8
        ? [selState.stateEmb[0]! / stateNorm, selState.stateEmb[1]! / stateNorm]
        : [0, 0];

      // Encode candidate info: [agency_left, agency_right, is_left_selected, is_right_selected]
      const agencyLeft = selState.candidates[0]?.agency || 0;
      const agencyRight = selState.candidates[1]?.agency || 0;
      const isLeft = selState.candidateIndex === 0 ? 1 : 0;
      const isRight = selState.candidateIndex === 1 ? 1 : 0;

      return [...normState, agencyLeft, agencyRight, isLeft, isRight];
    },

    embeddingDim: 6,
  };

  const selectionBank = await selectionTAM.learn("port-selection", selectionDomain, {
    epochs: 500,
    samplesPerEpoch: 50,
    flushFrequency: 20,
  });

  console.log(`✓ Selection meta-learner trained: ${selectionBank.getPortIds().length} ports\n`);

  // Evaluate with learned selection
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Composed Performance (Learned Selection)");
  console.log("═══════════════════════════════════════════════════════════\n");

  const composedResults = {
    left: { error: 0, count: 0, leftCount: 0, rightCount: 0, ltorCount: 0, rtolCount: 0 },
    gap: { error: 0, count: 0, leftCount: 0, rightCount: 0, ltorCount: 0, rtolCount: 0 },
    right: { error: 0, count: 0, leftCount: 0, rightCount: 0, ltorCount: 0, rtolCount: 0 },
  };

  for (const state of testStates) {
    const truth = piecewiseSpringDynamics(state.x, state.v, dt);
    const truthDelta = [truth.x - state.x, truth.v - state.v];

    // Get predictions from both actors
    const predLeft = bankLeft.predictFromState("default", { state, context: {} }, 1)[0];
    const predRight = bankRight.predictFromState("default", { state, context: {} }, 1)[0];

    if (!predLeft && !predRight) continue;

    // Use TAM-based selection
    const stateEmb = [state.x, state.v];
    const candidates = [
      { portId: "left", agency: predLeft?.agency || 0 },
      { portId: "right", agency: predRight?.agency || 0 },
    ];

    let bestIndex = 0;
    let bestBindingProb = 0;

    for (let i = 0; i < candidates.length; i++) {
      const selState: SelectionState = { stateEmb, candidates, candidateIndex: i };
      const selPreds = selectionBank.predictFromState("default", { state: selState, context: {} }, 1);

      if (selPreds.length > 0) {
        const bindingProb = selPreds[0]!.agency || 0.5;
        if (bindingProb > bestBindingProb) {
          bestBindingProb = bindingProb;
          bestIndex = i;
        }
      }
    }

    const selectedPred = bestIndex === 0 ? predLeft : predRight;
    if (!selectedPred) continue;

    const err = Math.sqrt(
      Math.pow(selectedPred.delta[0]! - truthDelta[0]!, 2) +
      Math.pow(selectedPred.delta[1]! - truthDelta[1]!, 2)
    );

    let region: "left" | "gap" | "right";
    if (state.x < -0.3) region = "left";
    else if (state.x > 0.3) region = "right";
    else region = "gap";

    composedResults[region].error += err;
    composedResults[region].count++;

    if (bestIndex === 0) composedResults[region].leftCount++;
    else composedResults[region].rightCount++;
  }

  // Average
  for (const region of ["left", "gap", "right"] as const) {
    if (composedResults[region].count > 0) {
      composedResults[region].error /= composedResults[region].count;
    }
  }

  console.log("Composed with Learned Selection:");
  console.log(`  Left:  ${composedResults.left.error.toFixed(4)}`);
  console.log(`    Selected: L=${composedResults.left.leftCount}, R=${composedResults.left.rightCount}`);
  console.log(`  Gap:   ${composedResults.gap.error.toFixed(4)}`);
  console.log(`    Selected: L=${composedResults.gap.leftCount}, R=${composedResults.gap.rightCount}`);
  console.log(`  Right: ${composedResults.right.error.toFixed(4)}`);
  console.log(`    Selected: L=${composedResults.right.leftCount}, R=${composedResults.right.rightCount}`);

  // Analysis
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis: Gap Region Comparison");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Gap Performance:");
  console.log(`  Left Actor (MaxAgency):    ${regionResultsLeft.gap.error.toFixed(4)} | Agency ${(regionResultsLeft.gap.agency! * 100).toFixed(1)}%`);
  console.log(`  Right Actor (MaxAgency):   ${regionResultsRight.gap.error.toFixed(4)} | Agency ${(regionResultsRight.gap.agency! * 100).toFixed(1)}%`);
  console.log(`  Baseline (Full Training):  ${regionResultsBase.gap.error.toFixed(4)}`);
  console.log(`  Learned Selection:         ${composedResults.gap.error.toFixed(4)}\n`);

  const bestIndividual = Math.min(regionResultsLeft.gap.error, regionResultsRight.gap.error);
  const improvement = ((bestIndividual - composedResults.gap.error) / bestIndividual * 100);

  if (composedResults.gap.error < bestIndividual) {
    console.log(`✓ Learned selection improves over best individual by ${improvement.toFixed(1)}%`);
  } else if (composedResults.gap.error < bestIndividual * 1.1) {
    console.log(`≈ Learned selection matches best individual (within 10%)`);
  } else {
    console.log(`⚠ Learned selection does not improve over individual actors`);
  }

  console.log(`\nSelection breakdown in gap:`);
  console.log(`  Left selected:  ${composedResults.gap.leftCount}/${composedResults.gap.count} (${(composedResults.gap.leftCount / composedResults.gap.count * 100).toFixed(1)}%)`);
  console.log(`  Right selected: ${composedResults.gap.rightCount}/${composedResults.gap.count} (${(composedResults.gap.rightCount / composedResults.gap.count * 100).toFixed(1)}%)`);

  const rightSelectionRate = composedResults.gap.rightCount / composedResults.gap.count;
  if (rightSelectionRate > 0.7) {
    console.log(`\n✓ Meta-learner learned to prefer right actor in gap`);
    console.log(`  (Despite left actor's higher agency)`);
  } else if (rightSelectionRate < 0.3) {
    console.log(`\n⚠ Meta-learner still prefers left actor`);
    console.log(`  (May need more training or better reward signal)`);
  } else {
    console.log(`\n≈ Meta-learner uses mixed strategy`);
  }

  // Export
  const exportData = {
    name: "Composition with Learned Selection",
    config: {
      domain: "Piecewise Spring (k1=2.0, k2=0.5)",
      epochs: 300,
      selectionEpochs: 500,
      actors: {
        left: "Stiff spring (x < -0.3)",
        right: "Soft spring (x > 0.3)",
        baseline: "Full range",
        gap: "Transition [-0.3, 0.3] - UNSEEN",
      },
      portCounts: {
        left: bankLeft.getPortIds().length,
        right: bankRight.getPortIds().length,
        baseline: bankBaseline.getPortIds().length,
        selection: selectionBank.getPortIds().length,
      },
    },
    functors: {
      leftToRight: pathLeftToRight ? { found: true, bindingRate: pathLeftToRight.totalBindingRate } : { found: false },
      rightToLeft: pathRightToLeft ? { found: true, bindingRate: pathRightToLeft.totalBindingRate } : { found: false },
    },
    performance: {
      gap: {
        leftActor: { error: regionResultsLeft.gap.error, agency: regionResultsLeft.gap.agency },
        rightActor: { error: regionResultsRight.gap.error, agency: regionResultsRight.gap.agency },
        baseline: { error: regionResultsBase.gap.error, agency: regionResultsBase.gap.agency },
        learnedSelection: {
          error: composedResults.gap.error,
          leftCount: composedResults.gap.leftCount,
          rightCount: composedResults.gap.rightCount,
          rightSelectionRate: rightSelectionRate,
        },
        improvement: improvement > 0 ? `${improvement.toFixed(1)}%` : "none",
      },
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write("examples/results/05c-composition-learned-selection.json", JSON.stringify(exportData, null, 2));
  console.log("\n✓ Results saved to examples/results/05c-composition-learned-selection.json");

  // Cleanup
  tam.dispose();
  selectionTAM.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
