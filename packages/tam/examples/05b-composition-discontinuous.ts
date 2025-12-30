/**
 * Experiment 05b: Composition with Discontinuous Dynamics
 *
 * Tests whether functor composition can bridge discontinuities that
 * simple extrapolation cannot handle.
 *
 * Domain: Piecewise Spring with Boundary Discontinuity
 * - Left region (x < -0.3): Stiff spring (k=2.0, b=0.3)
 * - Right region (x > 0.3): Soft spring (k=0.5, b=0.1)
 * - Gap region [-0.3, 0.3]: Transition zone (NEVER SEEN)
 *
 * Key difference from 05: Dynamics change sharply at boundaries
 * - Simple extrapolation should fail in gap
 * - Composition should be necessary to handle transition
 */

import { TAM, type DomainSpec } from "../src";

// Discontinuous spring dynamics
function piecewiseSpringDynamics(x: number, v: number, dt: number): { x: number; v: number } {
  let k: number, b: number;

  // Piecewise dynamics with sharp transition
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
    const blend = (x + 0.3) / 0.6; // 0 at left boundary, 1 at right
    k = 2.0 + blend * (0.5 - 2.0);
    b = 0.3 + blend * (0.1 - 0.3);
  }

  const ax = -k * x - b * v;
  const newV = v + ax * dt;
  const newX = x + newV * dt;
  return { x: newX, v: newV };
}

// Sample state uniformly in range
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

    // Categorize by region
    let region: "left" | "gap" | "right";
    if (state.x < -0.3) region = "left";
    else if (state.x > 0.3) region = "right";
    else region = "gap";

    results[region].error += err;
    results[region].count++;
    results[region].agency! += pred.agency || 0;
  }

  // Average
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

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Discontinuous Dynamics Composition");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Question: Can composition bridge discontinuities?\n");
  console.log("Dynamics: Piecewise Spring");
  console.log("  Left  (x < -0.3): Stiff spring (k=2.0, b=0.3)");
  console.log("  Right (x > 0.3):  Soft spring (k=0.5, b=0.1)");
  console.log("  Gap   [-0.3,0.3]: Transition (NEVER SEEN)\n");
  console.log("Hypothesis: Extrapolation fails, composition helps\n");

  const tam = new TAM();
  const dt = 0.1;

  // Domain A: Left region (stiff spring)
  const leftDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(-1.0, -0.3),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  // Domain B: Right region (soft spring)
  const rightDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(0.3, 1.0),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  // Baseline: Full range (including gap)
  const fullDomain: DomainSpec<{ x: number; v: number }> = {
    randomState: () => sampleState(-1.0, 1.0),
    simulate: (state) => piecewiseSpringDynamics(state.x, state.v, dt),
    embedder: (state) => [state.x, state.v],
    embeddingDim: 2,
  };

  // Train actors
  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 1: Learning Left Actor (Stiff Spring)");
  console.log("─────────────────────────────────────────────────────────");
  const bankLeft = await trainActor(tam, "left-stiff", leftDomain, 150);
  console.log(`  ✓ Trained: ${bankLeft.getPortIds().length} ports\n`);

  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 2: Learning Right Actor (Soft Spring)");
  console.log("─────────────────────────────────────────────────────────");
  const bankRight = await trainActor(tam, "right-soft", rightDomain, 150);
  console.log(`  ✓ Trained: ${bankRight.getPortIds().length} ports\n`);

  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 3: Baseline (Full Range Including Gap)");
  console.log("─────────────────────────────────────────────────────────");
  const bankBaseline = await trainActor(tam, "full-range", fullDomain, 150);
  console.log(`  ✓ Trained: ${bankBaseline.getPortIds().length} ports\n`);

  // Functor discovery
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Functor Discovery");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Attempting to discover functors between regimes...\n");

  const pathLeftToRight = await tam.findPath("left-stiff", "right-soft", 1);
  const pathRightToLeft = await tam.findPath("right-soft", "left-stiff", 1);

  if (pathLeftToRight) {
    console.log(`✓ Found functor: left → right`);
    console.log(`  Binding rate: ${(pathLeftToRight.totalBindingRate * 100).toFixed(1)}%`);
  } else {
    console.log(`✗ No functor: left → right`);
  }

  if (pathRightToLeft) {
    console.log(`✓ Found functor: right → left`);
    console.log(`  Binding rate: ${(pathRightToLeft.totalBindingRate * 100).toFixed(1)}%`);
  } else {
    console.log(`✗ No functor: right → left`);
  }

  console.log();

  // Generate test states across all regions
  const testStates: Array<{ x: number; v: number }> = [];
  for (let i = 0; i < 150; i++) {
    testStates.push(sampleState(-1.0, 1.0));
  }

  // Evaluate individual actors
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Performance by Region");
  console.log("═══════════════════════════════════════════════════════════");

  const regionResultsLeft = evaluateByRegion(
    (state) => {
      const pred = bankLeft.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Left Actor (Stiff)"
  );

  const regionResultsRight = evaluateByRegion(
    (state) => {
      const pred = bankRight.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Right Actor (Soft)"
  );

  const regionResultsBase = evaluateByRegion(
    (state) => {
      const pred = bankBaseline.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Baseline (Full)"
  );

  // Composed performance
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Composed Performance");
  console.log("═══════════════════════════════════════════════════════════\n");

  let composedResults: {
    left: { error: number; count: number; leftCount: number; rightCount: number; ltorCount: number; rtolCount: number };
    gap: { error: number; count: number; leftCount: number; rightCount: number; ltorCount: number; rtolCount: number };
    right: { error: number; count: number; leftCount: number; rightCount: number; ltorCount: number; rtolCount: number };
  } | null = null;

  if (pathLeftToRight && pathRightToLeft) {
    console.log("Strategy: ComposedPort with cross-regime functors\n");

    const composedLtoR = tam.compose(pathLeftToRight);
    const composedRtoL = tam.compose(pathRightToLeft);

    composedResults = {
      left: { error: 0, count: 0, leftCount: 0, rightCount: 0, ltorCount: 0, rtolCount: 0 },
      gap: { error: 0, count: 0, leftCount: 0, rightCount: 0, ltorCount: 0, rtolCount: 0 },
      right: { error: 0, count: 0, leftCount: 0, rightCount: 0, ltorCount: 0, rtolCount: 0 },
    };

    for (const state of testStates) {
      try {
        const predLeft = bankLeft.predictFromState("default", { state, context: {} }, 1)[0];
        const predRight = bankRight.predictFromState("default", { state, context: {} }, 1)[0];

        let composedLR, composedRL;
        try {
          composedLR = composedLtoR.predict(state);
          composedRL = composedRtoL.predict(state);
        } catch {
          composedLR = null;
          composedRL = null;
        }

        const candidates = [
          predLeft ? { delta: predLeft.delta, agency: predLeft.agency, source: "left" } : null,
          predRight ? { delta: predRight.delta, agency: predRight.agency, source: "right" } : null,
          composedLR ? { delta: composedLR.delta, agency: composedLR.agency || 0, source: "L→R" } : null,
          composedRL ? { delta: composedRL.delta, agency: composedRL.agency || 0, source: "R→L" } : null,
        ].filter((c) => c !== null) as Array<{ delta: number[]; agency: number; source: string }>;

        if (candidates.length === 0) continue;

        const best = candidates.reduce((a, b) => (a.agency > b.agency ? a : b));

        const truth = piecewiseSpringDynamics(state.x, state.v, dt);
        const truthDelta = [truth.x - state.x, truth.v - state.v];
        const err = Math.sqrt(
          Math.pow(best.delta[0]! - truthDelta[0]!, 2) +
          Math.pow(best.delta[1]! - truthDelta[1]!, 2)
        );

        let region: "left" | "gap" | "right";
        if (state.x < -0.3) region = "left";
        else if (state.x > 0.3) region = "right";
        else region = "gap";

        composedResults[region].error += err;
        composedResults[region].count++;

        if (best.source === "left") composedResults[region].leftCount++;
        else if (best.source === "right") composedResults[region].rightCount++;
        else if (best.source === "L→R") composedResults[region].ltorCount++;
        else composedResults[region].rtolCount++;
      } catch (e) {
        console.error(`Error: ${e}`);
      }
    }

    // Average
    for (const region of ["left", "gap", "right"] as const) {
      if (composedResults[region].count > 0) {
        composedResults[region].error /= composedResults[region].count;
      }
    }

    console.log("Composed via ComposedPort (All 4 Candidates):");
    console.log(`  Left:  ${composedResults.left.error.toFixed(4)}`);
    console.log(`    Direct:   L=${composedResults.left.leftCount}, R=${composedResults.left.rightCount}`);
    console.log(`    Composed: L→R=${composedResults.left.ltorCount}, R→L=${composedResults.left.rtolCount}`);
    console.log(`  Gap:   ${composedResults.gap.error.toFixed(4)}`);
    console.log(`    Direct:   L=${composedResults.gap.leftCount}, R=${composedResults.gap.rightCount}`);
    console.log(`    Composed: L→R=${composedResults.gap.ltorCount}, R→L=${composedResults.gap.rtolCount}`);
    console.log(`  Right: ${composedResults.right.error.toFixed(4)}`);
    console.log(`    Direct:   L=${composedResults.right.leftCount}, R=${composedResults.right.rightCount}`);
    console.log(`    Composed: L→R=${composedResults.right.ltorCount}, R→L=${composedResults.right.rtolCount}`);
  } else {
    console.log("⚠ Functors not discovered, skipping composition\n");
  }

  // Analysis
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis: Gap Region (Discontinuity Test)");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Gap Region Performance (Never Seen by Specialists):");
  console.log(`  Left Actor alone:  ${regionResultsLeft.gap.error.toFixed(4)}`);
  console.log(`  Right Actor alone: ${regionResultsRight.gap.error.toFixed(4)}`);
  console.log(`  Baseline:          ${regionResultsBase.gap.error.toFixed(4)}`);
  if (composedResults) {
    console.log(`  Composed:          ${composedResults.gap.error.toFixed(4)}\n`);

    const bestIndividual = Math.min(regionResultsLeft.gap.error, regionResultsRight.gap.error);
    const improvement = ((bestIndividual - composedResults.gap.error) / bestIndividual * 100);

    if (composedResults.gap.error < bestIndividual && improvement > 5) {
      console.log(`✓ Composition improves over best individual by ${improvement.toFixed(1)}%`);
    } else if (composedResults.gap.error < regionResultsBase.gap.error) {
      console.log(`✓ Composition competitive with baseline`);
    } else {
      console.log(`⚠ Composition does not improve performance`);
    }

    const composedUsed = composedResults.gap.ltorCount + composedResults.gap.rtolCount;
    if (composedUsed > 0) {
      console.log(`✓ ComposedPort selected ${composedUsed}/${composedResults.gap.count} times in gap`);
      console.log(`  Shows functors are trusted for bridging regimes`);
    } else {
      console.log(`⚠ ComposedPort never selected (direct predictions preferred)`);
    }
  }

  // Export
  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Exporting Results");
  console.log("─────────────────────────────────────────────────────────");

  const exportData = {
    name: "Discontinuous Dynamics Composition",
    config: {
      domain: "Piecewise Spring (k1=2.0, k2=0.5)",
      epochs: 150,
      actors: {
        left: "Stiff spring (x < -0.3)",
        right: "Soft spring (x > 0.3)",
        baseline: "Full range (x ∈ [-1, 1])",
        gap: "Transition (x ∈ [-0.3, 0.3]) - UNSEEN",
      },
      finalPortCounts: {
        left: bankLeft.getPortIds().length,
        right: bankRight.getPortIds().length,
        baseline: bankBaseline.getPortIds().length,
      },
    },
    functors: {
      leftToRight: pathLeftToRight
        ? { found: true, bindingRate: pathLeftToRight.totalBindingRate, steps: pathLeftToRight.steps.length }
        : { found: false },
      rightToLeft: pathRightToLeft
        ? { found: true, bindingRate: pathRightToLeft.totalBindingRate, steps: pathRightToLeft.steps.length }
        : { found: false },
      totalAttempted: 2,
    },
    regionalPerformance: {
      left: { left: regionResultsLeft.left, gap: regionResultsLeft.gap, right: regionResultsLeft.right },
      right: { left: regionResultsRight.left, gap: regionResultsRight.gap, right: regionResultsRight.right },
      baseline: { left: regionResultsBase.left, gap: regionResultsBase.gap, right: regionResultsBase.right },
      composed: composedResults ? { left: composedResults.left, gap: composedResults.gap, right: composedResults.right } : null,
    },
    summary: {
      functorsDiscovered: (pathLeftToRight ? 1 : 0) + (pathRightToLeft ? 1 : 0),
      compositionUsed: composedResults ? (composedResults.gap.ltorCount + composedResults.gap.rtolCount) > 0 : false,
      gapPerformance: composedResults
        ? {
            left: regionResultsLeft.gap.error,
            right: regionResultsRight.gap.error,
            baseline: regionResultsBase.gap.error,
            composed: composedResults.gap.error,
            improvement:
              composedResults.gap.error < Math.min(regionResultsLeft.gap.error, regionResultsRight.gap.error)
                ? ((1 - composedResults.gap.error / Math.min(regionResultsLeft.gap.error, regionResultsRight.gap.error)) * 100).toFixed(1) + "%"
                : "none",
          }
        : null,
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write("examples/results/05b-composition-discontinuous.json", JSON.stringify(exportData, null, 2));
  console.log("✓ Results saved to examples/results/05b-composition-discontinuous.json\n");

  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Key Takeaway");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  Discontinuous dynamics test composition necessity:");
  console.log("  - Sharp boundary between stiff/soft spring regimes");
  console.log("  - Gap region requires understanding both dynamics");
  console.log("  - Direct extrapolation should fail");
  console.log("  - Composition should bridge the discontinuity\n");

  // Cleanup
  tam.dispose();

  console.log("✓ Experiment complete!");
}

main().catch(console.error);
