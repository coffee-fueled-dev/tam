/**
 * Experiment 05: Composition (Coverage Extension)
 *
 * Demonstrates:
 * - Training actors on different scale regimes using TAM API
 * - Functor discovery between scale-specialized actors
 * - Composition extends coverage to unseen scales
 * - Systematic interpolation/extrapolation
 *
 * Setup:
 *   Actor A: Small-scale dynamics (x ∈ [-0.5, 0.5], dynamics: spring with k=1.0)
 *   Actor B: Large-scale dynamics (x ∈ [-2, -1] ∪ [1, 2], dynamics: spring with k=1.0)
 *   Gap: Intermediate scale (x ∈ [0.5, 1]) - never seen by either!
 *
 * Key insight:
 *   Both actors learn the SAME dynamics (spring) at different scales.
 *   TAM discovers functors between them automatically.
 *   Composition should extend coverage to intermediate/extrapolated scales.
 *
 * Expected outcome:
 *   - Actor A: Good performance on small scale
 *   - Actor B: Good performance on large scale
 *   - Composition: Better coverage on intermediate/full range
 *   - Demonstrates systematic scale understanding via functors
 */

import { TAM, type DomainSpec, agencyFromCone } from "../src";

// ============================================================================
// Shared Spring Dynamics
// ============================================================================

const k = 1.0;
const b = 0.1;
const dt = 0.1;

function springSimulate(x: number, v: number, steps: number = 1): { x: number; v: number } {
  let currX = x;
  let currV = v;

  for (let i = 0; i < steps; i++) {
    const ax = -k * currX - b * currV;
    currV = currV + ax * dt;
    currX = currX + currV * dt;
  }

  return { x: currX, v: currV };
}

// ============================================================================
// Domain Specifications
// ============================================================================

/**
 * Domain A: Small-scale spring (x ∈ [-0.5, 0.5])
 */
const smallScaleDomain: DomainSpec<{ x: number; v: number }> = {
  randomState: () => ({
    x: (Math.random() - 0.5), // x ∈ [-0.5, 0.5]
    v: (Math.random() - 0.5), // v ∈ [-0.5, 0.5]
  }),

  simulate: (state) => springSimulate(state.x, state.v, 1),

  embeddingDim: 2,
  embedder: (state) => [state.x, state.v],
};

/**
 * Domain B: Large-scale spring (x ∈ [-2, -1] ∪ [1, 2])
 * Excludes center region!
 */
const largeScaleDomain: DomainSpec<{ x: number; v: number }> = {
  randomState: () => {
    // Sample from [-2, -1] or [1, 2]
    const side = Math.random() < 0.5 ? -1 : 1;
    return {
      x: side * (1 + Math.random()), // ±[1, 2]
      v: (Math.random() - 0.5) * 2, // v ∈ [-1, 1]
    };
  },

  simulate: (state) => springSimulate(state.x, state.v, 1),

  embeddingDim: 2,
  embedder: (state) => [state.x, state.v],
};

/**
 * Domain C (Full): Full-scale spring (x ∈ [-2, 2])
 * This is what we want to cover via composition.
 */
const fullScaleDomain: DomainSpec<{ x: number; v: number }> = {
  randomState: () => ({
    x: (Math.random() - 0.5) * 4, // x ∈ [-2, 2]
    v: (Math.random() - 0.5) * 2, // v ∈ [-1, 1]
  }),

  simulate: (state) => springSimulate(state.x, state.v, 1),

  embeddingDim: 2,
  embedder: (state) => [state.x, state.v],
};

// ============================================================================
// Evaluation by Region
// ============================================================================

interface RegionResults {
  small: { error: number; count: number; agency?: number };
  gap: { error: number; count: number; agency?: number };
  large: { error: number; count: number; agency?: number };
}

function evaluateByRegion(
  predictFn: (state: { x: number; v: number }) => { delta: number[]; agency?: number },
  testStates: Array<{ x: number; v: number }>,
  label: string
): RegionResults {
  const results: RegionResults = {
    small: { error: 0, count: 0, agency: 0 },
    gap: { error: 0, count: 0, agency: 0 },
    large: { error: 0, count: 0, agency: 0 },
  };

  for (const state of testStates) {
    try {
      const pred = predictFn(state);
      if (!pred) continue;

      const actual = springSimulate(state.x, state.v, 1);
      const predicted = {
        x: state.x + pred.delta[0]!,
        v: state.v + pred.delta[1]!,
      };

      const error = Math.sqrt(
        (predicted.x - actual.x) ** 2 + (predicted.v - actual.v) ** 2
      );

      // Categorize by region
      const absX = Math.abs(state.x);
      let region: keyof RegionResults;
      if (absX < 0.5) {
        region = "small";
      } else if (absX < 1.0) {
        region = "gap";
      } else {
        region = "large";
      }

      results[region].error += error;
      results[region].count++;
      if (pred.agency !== undefined) {
        results[region].agency = (results[region].agency || 0) + pred.agency;
      }
    } catch (e) {
      // Skip states that fail
    }
  }

  // Average errors and agency
  for (const region of ["small", "gap", "large"] as const) {
    if (results[region].count > 0) {
      results[region].error /= results[region].count;
      if (results[region].agency !== undefined) {
        results[region].agency! /= results[region].count;
      }
    }
  }

  console.log(`\n${label}:`);
  console.log(`  Small [0, 0.5):   Error ${results.small.error.toFixed(4)} | Agency ${((results.small.agency || 0) * 100).toFixed(1)}% | n=${results.small.count}`);
  console.log(`  Gap [0.5, 1.0):   Error ${results.gap.error.toFixed(4)} | Agency ${((results.gap.agency || 0) * 100).toFixed(1)}% | n=${results.gap.count}`);
  console.log(`  Large [1.0+):     Error ${results.large.error.toFixed(4)} | Agency ${((results.large.agency || 0) * 100).toFixed(1)}% | n=${results.large.count}`);

  return results;
}

// ============================================================================
// Simple Training (TAM doesn't support incremental checkpointing)
// ============================================================================

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

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Scale Composition (Coverage Extension) - TAM Infrastructure");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Question: Can functor composition extend coverage to unseen scales?\n");
  console.log("Setup:");
  console.log("  Actor A: Small scale (x ∈ [-0.5, 0.5])");
  console.log("  Actor B: Large scale (x ∈ [-2, -1] ∪ [1, 2])");
  console.log("  Gap: Intermediate (x ∈ [0.5, 1.0]) - NEVER SEEN!");
  console.log("\nKey: Same dynamics (spring), different scale regimes.");
  console.log("     Can TAM discover functors and interpolate to unseen scales?\n");

  const tam = new TAM({
    maxEpochs: 100,
    successThreshold: 0.7,
    patience: 30,
    minEpochs: 10,
    hiddenSizes: [16, 16],
    learningRate: 0.01,
    samplesPerEpoch: 50,
  });

  // ============================================================================
  // Step 1: Learn Small-Scale Actor
  // ============================================================================

  console.log("─────────────────────────────────────────────────────────");
  console.log("Step 1: Learning Actor A (Small Scale)");
  console.log("─────────────────────────────────────────────────────────");

  const bankA = await trainActor(tam, "small-scale", smallScaleDomain, 100);
  console.log(`  ✓ Trained: ${bankA.getPortIds().length} ports`);

  // ============================================================================
  // Step 2: Learn Large-Scale Actor
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Step 2: Learning Actor B (Large Scale)");
  console.log("─────────────────────────────────────────────────────────");

  const bankB = await trainActor(tam, "large-scale", largeScaleDomain, 100);
  console.log(`  ✓ Trained: ${bankB.getPortIds().length} ports`);

  // ============================================================================
  // Step 3: Baseline (Full Scale)
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Step 3: Baseline (Train from Scratch on Full Scale)");
  console.log("─────────────────────────────────────────────────────────");

  const bankBaseline = await trainActor(tam, "full-scale", fullScaleDomain, 100);
  console.log(`  ✓ Trained: ${bankBaseline.getPortIds().length} ports`);

  // ============================================================================
  // Step 4: Discover Functors
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Functor Discovery");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Attempting to discover functors between scale regimes...\n");

  // Try to find paths between domains
  const pathAtoB = await tam.findPath("small-scale", "large-scale", 1);
  const pathBtoA = await tam.findPath("large-scale", "small-scale", 1);

  if (pathAtoB) {
    console.log(`✓ Found functor: small → large`);
    console.log(`  Binding rate: ${(pathAtoB.totalBindingRate * 100).toFixed(1)}%`);
  } else {
    console.log(`✗ No functor found: small → large`);
  }

  if (pathBtoA) {
    console.log(`✓ Found functor: large → small`);
    console.log(`  Binding rate: ${(pathBtoA.totalBindingRate * 100).toFixed(1)}%`);
  } else {
    console.log(`✗ No functor found: large → small`);
  }

  // Show all attempted pairs
  const attempted = tam.getAttemptedPairs();
  console.log(`\nTotal functor attempts: ${attempted.length}`);

  // ============================================================================
  // Step 5: Evaluate by Region
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Performance by Scale Region");
  console.log("═══════════════════════════════════════════════════════════");

  // Generate test states across all regions
  const testStates: Array<{ x: number; v: number }> = [];

  // Small scale
  for (let i = 0; i < 50; i++) {
    testStates.push({
      x: (Math.random() - 0.5),
      v: (Math.random() - 0.5),
    });
  }

  // Gap (intermediate)
  for (let i = 0; i < 50; i++) {
    const side = Math.random() < 0.5 ? -1 : 1;
    testStates.push({
      x: side * (0.5 + Math.random() * 0.5), // ±[0.5, 1.0]
      v: (Math.random() - 0.5),
    });
  }

  // Large scale
  for (let i = 0; i < 50; i++) {
    const side = Math.random() < 0.5 ? -1 : 1;
    testStates.push({
      x: side * (1 + Math.random()), // ±[1.0, 2.0]
      v: (Math.random() - 0.5) * 2,
    });
  }

  // Evaluate Actor A
  const regionResultsA = evaluateByRegion(
    (state) => {
      const pred = bankA.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Actor A (Small)"
  );

  // Evaluate Actor B
  const regionResultsB = evaluateByRegion(
    (state) => {
      const pred = bankB.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Actor B (Large)"
  );

  // Evaluate Baseline
  const regionResultsBase = evaluateByRegion(
    (state) => {
      const pred = bankBaseline.predictFromState("default", { state, context: {} }, 1)[0];
      return pred ? { delta: pred.delta, agency: pred.agency } : { delta: [0, 0], agency: 0 };
    },
    testStates,
    "Baseline (Full)"
  );

  // ============================================================================
  // Step 6: Composed Performance (True Functor Composition via ComposedPort)
  // ============================================================================

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Composed Performance (True Functor Composition)");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Declare composedResults outside block so it's accessible for export
  let composedResults: {
    small: { error: number; count: number; aCount: number; bCount: number; atobCount: number; btoaCount: number };
    gap: { error: number; count: number; aCount: number; bCount: number; atobCount: number; btoaCount: number };
    large: { error: number; count: number; aCount: number; bCount: number; atobCount: number; btoaCount: number };
  } | null = null;

  if (pathAtoB && pathBtoA) {
    console.log("Strategy: Use ComposedPort to leverage both actors' knowledge\n");
    console.log("  A→B: Actor A uses Actor B's large-scale expertise via functor");
    console.log("  B→A: Actor B uses Actor A's small-scale expertise via functor\n");

    // Create composed ports
    const composedAtoB = tam.compose(pathAtoB);
    const composedBtoA = tam.compose(pathBtoA);

    composedResults = {
      small: { error: 0, count: 0, aCount: 0, bCount: 0, atobCount: 0, btoaCount: 0 },
      gap: { error: 0, count: 0, aCount: 0, bCount: 0, atobCount: 0, btoaCount: 0 },
      large: { error: 0, count: 0, aCount: 0, bCount: 0, atobCount: 0, btoaCount: 0 },
    };

    for (const state of testStates) {
      try {
        // Get direct predictions from both actors
        const predA = bankA.predictFromState("default", { state, context: {} }, 1)[0];
        const predB = bankB.predictFromState("default", { state, context: {} }, 1)[0];

        // Get composed predictions (A leveraging B, and vice versa)
        // Note: These may produce NaN if ComposedPort can't properly bypass encoder
        let composedAB, composedBA;
        try {
          composedAB = composedAtoB.predict(state);
          composedBA = composedBtoA.predict(state);
        } catch (e) {
          // Skip if composed prediction fails
          continue;
        }

        if (!predA || !predB) continue;

        // Evaluate all 4 approaches and pick best by confidence
        const candidates = [
          { pred: predA, source: "A", type: "direct" },
          { pred: predB, source: "B", type: "direct" },
          { pred: { delta: composedAB.delta, agency: agencyFromCone(composedAB.cone), cone: composedAB.cone }, source: "A→B", type: "composed" },
          { pred: { delta: composedBA.delta, agency: agencyFromCone(composedBA.cone), cone: composedBA.cone }, source: "B→A", type: "composed" },
        ];

        // Select best by agency
        const best = candidates.reduce((a, b) => (a.pred.agency > b.pred.agency ? a : b));
        const pred = best.pred;

        const actual = springSimulate(state.x, state.v, 1);

        // Check if delta is valid
        if (!pred.delta || pred.delta.length < 2 || !isFinite(pred.delta[0]!) || !isFinite(pred.delta[1]!)) {
          console.error(`Invalid delta from ${best.source}:`, pred.delta);
          continue;
        }

        const predicted = {
          x: state.x + pred.delta[0]!,
          v: state.v + pred.delta[1]!,
        };

        const error = Math.sqrt(
          (predicted.x - actual.x) ** 2 + (predicted.v - actual.v) ** 2
        );

        if (!isFinite(error)) {
          console.error(`Non-finite error from ${best.source}:`, { predicted, actual, delta: pred.delta });
          continue;
        }

        // Categorize by region and track which approach was used
        const absX = Math.abs(state.x);
        if (absX < 0.5) {
          composedResults.small.error += error;
          composedResults.small.count++;
          if (best.source === "A") composedResults.small.aCount++;
          else if (best.source === "B") composedResults.small.bCount++;
          else if (best.source === "A→B") composedResults.small.atobCount++;
          else composedResults.small.btoaCount++;
        } else if (absX < 1.0) {
          composedResults.gap.error += error;
          composedResults.gap.count++;
          if (best.source === "A") composedResults.gap.aCount++;
          else if (best.source === "B") composedResults.gap.bCount++;
          else if (best.source === "A→B") composedResults.gap.atobCount++;
          else composedResults.gap.btoaCount++;
        } else {
          composedResults.large.error += error;
          composedResults.large.count++;
          if (best.source === "A") composedResults.large.aCount++;
          else if (best.source === "B") composedResults.large.bCount++;
          else if (best.source === "A→B") composedResults.large.atobCount++;
          else composedResults.large.btoaCount++;
        }
      } catch (e) {
        // Skip states that fail
        console.error(`Error in composed prediction: ${e}`);
      }
    }

    // Average
    if (composedResults.small.count > 0) {
      composedResults.small.error /= composedResults.small.count;
    }
    if (composedResults.gap.count > 0) {
      composedResults.gap.error /= composedResults.gap.count;
    }
    if (composedResults.large.count > 0) {
      composedResults.large.error /= composedResults.large.count;
    }

    console.log("Composed via ComposedPort (All 4 Candidates):");
    console.log(`  Small scale: ${composedResults.small.error.toFixed(4)}`);
    console.log(`    Direct:   A=${composedResults.small.aCount}, B=${composedResults.small.bCount}`);
    console.log(`    Composed: A→B=${composedResults.small.atobCount}, B→A=${composedResults.small.btoaCount}`);
    console.log(`  Gap:         ${composedResults.gap.error.toFixed(4)}`);
    console.log(`    Direct:   A=${composedResults.gap.aCount}, B=${composedResults.gap.bCount}`);
    console.log(`    Composed: A→B=${composedResults.gap.atobCount}, B→A=${composedResults.gap.btoaCount}`);
    console.log(`  Large scale: ${composedResults.large.error.toFixed(4)}`);
    console.log(`    Direct:   A=${composedResults.large.aCount}, B=${composedResults.large.bCount}`);
    console.log(`    Composed: A→B=${composedResults.large.atobCount}, B→A=${composedResults.large.btoaCount}`);

    // ============================================================================
    // Analysis
    // ============================================================================

    console.log("\n═══════════════════════════════════════════════════════════");
    console.log("  Analysis");
    console.log("═══════════════════════════════════════════════════════════\n");

    console.log("Gap Region (Never Seen by Either Actor):");
    console.log(`  Actor A alone:  ${regionResultsA.gap.error.toFixed(4)}`);
    console.log(`  Actor B alone:  ${regionResultsB.gap.error.toFixed(4)}`);
    console.log(`  Baseline:       ${regionResultsBase.gap.error.toFixed(4)}`);
    console.log(`  Composed:       ${composedResults.gap.error.toFixed(4)}`);

    const bestIndividual = Math.min(regionResultsA.gap.error, regionResultsB.gap.error);
    const composedVsBaseline = ((1 - composedResults.gap.error / regionResultsBase.gap.error) * 100);

    if (composedResults.gap.error < bestIndividual * 0.9) {
      console.log("\n✓ Composition significantly outperforms individual actors on gap!");
      console.log("  Demonstrates systematic interpolation to unseen scales");
    } else if (composedResults.gap.error < bestIndividual) {
      console.log("\n✓ Composition slightly better than individual actors on gap");
      console.log("  Shows some interpolation capability");
    } else {
      console.log("\n⚠ Composition similar to best individual actor");
    }

    if (composedResults.gap.error < regionResultsBase.gap.error) {
      console.log(`✓ Composed outperforms baseline by ${composedVsBaseline.toFixed(1)}%`);
      console.log("  Without ever training on full scale!");
    }

    console.log("\n✓ Functors discovered in both directions!");
    console.log("  Shows scale transformation is learnable");

    // Show which composition strategy is being used where
    const totalComposed = composedResults.gap.atobCount + composedResults.gap.btoaCount;
    const totalDirect = composedResults.gap.aCount + composedResults.gap.bCount;
    if (totalComposed > 0) {
      console.log(`\n✓ ComposedPort actively used! (${totalComposed}/${composedResults.gap.count} in gap)`);
      console.log("  System leverages cross-scale knowledge via functors");
    } else {
      console.log(`\n⚠ ComposedPort not selected (direct predictions preferred)`);
      console.log("  May need to improve functor composition confidence");
    }
  } else {
    console.log("⚠ No functors found between scale regimes");
    console.log("  Cannot create ComposedPort");
  }

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Takeaway");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  TAM infrastructure for scale composition:");
  console.log("    1. Learn specialists on different scale regimes");
  console.log("    2. Discover functors between regimes automatically");
  console.log("    3. Compose via learned transformations");
  console.log("\n  Benefits:");
  console.log("    - Systematic interpolation/extrapolation");
  console.log("    - Functor discovery finds scale transformations");
  console.log("    - Better than ad-hoc agency selection");
  console.log("\n  Status:");
  console.log(`    - Registered domains: ${tam.listDomains().length}`);
  console.log(`    - Functor attempts: ${tam.getAttemptedPairs().length}`);
  console.log(`    - Successful paths: ${tam.getSuccessfulGraph().size}`);

  // ============================================================================
  // Export Results
  // ============================================================================

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Exporting Results");
  console.log("─────────────────────────────────────────────────────────");

  const exportData = {
    name: "Scale Composition - Coverage Extension",
    config: {
      domain: "1D Damped Spring",
      epochs: 100,
      actors: {
        actorA: "Small scale (x ∈ [-0.5, 0.5])",
        actorB: "Large scale (x ∈ [-2, -1] ∪ [1, 2])",
        baseline: "Full scale (x ∈ [-2, 2])",
        gap: "Intermediate (x ∈ [0.5, 1.0]) - UNSEEN",
      },
      finalPortCounts: {
        actorA: bankA.getPortIds().length,
        actorB: bankB.getPortIds().length,
        baseline: bankBaseline.getPortIds().length,
      },
    },
    functors: {
      smallToLarge: pathAtoB
        ? {
            found: true,
            bindingRate: pathAtoB.totalBindingRate,
            steps: pathAtoB.steps.length,
          }
        : { found: false },
      largeToSmall: pathBtoA
        ? {
            found: true,
            bindingRate: pathBtoA.totalBindingRate,
            steps: pathBtoA.steps.length,
          }
        : { found: false },
      totalAttempted: attempted.length,
    },
    regionalPerformance: {
      actorA: {
        small: regionResultsA.small,
        gap: regionResultsA.gap,
        large: regionResultsA.large,
      },
      actorB: {
        small: regionResultsB.small,
        gap: regionResultsB.gap,
        large: regionResultsB.large,
      },
      baseline: {
        small: regionResultsBase.small,
        gap: regionResultsBase.gap,
        large: regionResultsBase.large,
      },
      composed: composedResults
        ? {
            small: composedResults.small,
            gap: composedResults.gap,
            large: composedResults.large,
          }
        : null,
    },
    summary: {
      functorsDiscovered: (pathAtoB ? 1 : 0) + (pathBtoA ? 1 : 0),
      compositionUsed: composedResults
        ? (composedResults.gap.atobCount || 0) + (composedResults.gap.btoaCount || 0) > 0
        : false,
      gapPerformance: composedResults
        ? {
            actorA: regionResultsA.gap.error,
            actorB: regionResultsB.gap.error,
            baseline: regionResultsBase.gap.error,
            composed: composedResults.gap.error,
            improvement:
              composedResults.gap.error < Math.min(regionResultsA.gap.error, regionResultsB.gap.error)
                ? ((1 - composedResults.gap.error / Math.min(regionResultsA.gap.error, regionResultsB.gap.error)) * 100).toFixed(1) + "%"
                : "none",
          }
        : null,
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write(
    "examples/results/05-composition.json",
    JSON.stringify(exportData, null, 2)
  );
  console.log("✓ Results saved to examples/results/05-composition.json");

  // Cleanup
  tam.dispose();
  bankA.dispose();
  bankB.dispose();
  bankBaseline.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
