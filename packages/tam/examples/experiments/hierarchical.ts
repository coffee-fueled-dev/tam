/**
 * Hierarchical Composition Experiment
 *
 * Demonstrates compositional learning using await tam.learn() and functor discovery.
 *
 * Architecture:
 *   Level 0: Raw domains (shift, scale)
 *   Level 1: Learn each domain with await tam.learn()
 *   Level 2: Discover functors between domains
 *
 * This uses the TAM library directly rather than ad-hoc utilities.
 */

import { TAM, type TrainingConfig } from "../../src";
import { shift, scale, power, affine, shiftDeterministic } from "../domains";

// ============================================================================
// Configuration
// ============================================================================

const TRAINING: TrainingConfig = {
  epochs: 100,
  samplesPerEpoch: 50,
  flushFrequency: 10,
};

// ============================================================================
// Main
// ============================================================================

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Hierarchical Composition Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log(
    "Using await tam.learn() to train domains and discover functors.\n"
  );

  // Create TAM instance
  const tam = new TAM({
    maxEpochs: 200,
    patience: 50,
    successThreshold: 0.5,
    samplesPerEpoch: 50,
  });

  // Track domain metrics
  const domainMetrics: Array<{ name: string; agency: number }> = [];

  // Level 1: Learn primitive domains
  console.log("Level 1: Learning primitive domains...\n");

  console.log("  Training shift-deterministic domain (x → x+1)...");
  const shiftDetPort = await tam.learn(
    "shift-det",
    shiftDeterministic,
    TRAINING
  );
  domainMetrics.push({ name: "shift-det", agency: shiftDetPort.getAgency() });
  console.log(`    Agency: ${(shiftDetPort.getAgency() * 100).toFixed(1)}%`);

  console.log("  Training shift domain (stochastic)...");
  const shiftPort = await tam.learn("shift", shift, TRAINING);
  domainMetrics.push({ name: "shift", agency: shiftPort.getAgency() });
  console.log(`    Agency: ${(shiftPort.getAgency() * 100).toFixed(1)}%`);

  console.log("  Training scale domain...");
  const scalePort = await tam.learn("scale", scale, TRAINING);
  domainMetrics.push({ name: "scale", agency: scalePort.getAgency() });
  console.log(`    Agency: ${(scalePort.getAgency() * 100).toFixed(1)}%`);

  console.log("  Training power domain...");
  const powerPort = await tam.learn("power", power, TRAINING);
  domainMetrics.push({ name: "power", agency: powerPort.getAgency() });
  console.log(`    Agency: ${(powerPort.getAgency() * 100).toFixed(1)}%`);

  console.log("  Training affine domain...");
  const affinePort = await tam.learn("affine", affine, TRAINING);
  domainMetrics.push({ name: "affine", agency: affinePort.getAgency() });
  console.log(`    Agency: ${(affinePort.getAgency() * 100).toFixed(1)}%`);

  // Level 2: Discover functors
  console.log("\nLevel 2: Discovering functors...\n");

  const pairs = [
    ["shift", "scale"],
    ["shift", "affine"],
    ["scale", "affine"],
    ["affine", "power"],
  ];

  const results: Array<{
    source: string;
    target: string;
    status: string;
    bindingRate: number;
  }> = [];

  for (const pair of pairs) {
    const source = pair[0]!;
    const target = pair[1]!;
    console.log(`  Discovering ${source} → ${target}...`);
    const path = await tam.findPath(source, target, 1);

    if (path && path.steps.length > 0) {
      const rate = path.totalBindingRate;
      console.log(`    ✓ Found! Binding rate: ${(rate * 100).toFixed(1)}%`);
      results.push({ source, target, status: "found", bindingRate: rate });
    } else {
      const cached = tam.getCachedResult(source, target);
      console.log(`    ✗ Not found (${cached?.reason ?? "unknown"})`);
      results.push({
        source,
        target,
        status: "not_found",
        bindingRate: cached?.bindingRate ?? 0,
      });
    }
  }

  // Summary
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Results Summary");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Domain metrics table
  console.log("Domain Agency:");
  console.log("| Domain  | Agency |");
  console.log("|---------|--------|");
  for (const d of domainMetrics) {
    console.log(
      `| ${d.name.padEnd(7)} | ${(d.agency * 100).toFixed(1).padStart(5)}% |`
    );
  }

  // Functor metrics table
  console.log("\nFunctor Binding:");
  console.log("| Source  | Target  | Status    | Binding |");
  console.log("|---------|---------|-----------|---------|");
  for (const r of results) {
    const status = r.status === "found" ? "found" : "not found";
    console.log(
      `| ${r.source.padEnd(7)} | ${r.target.padEnd(7)} | ${status.padEnd(
        9
      )} | ${(r.bindingRate * 100).toFixed(1).padStart(5)}% |`
    );
  }

  // Aggregate metrics
  const avgAgency =
    domainMetrics.reduce((sum, d) => sum + d.agency, 0) / domainMetrics.length;
  const avgBinding =
    results.reduce((sum, r) => sum + r.bindingRate, 0) / results.length;
  const foundCount = results.filter((r) => r.status === "found").length;

  console.log("\nAggregate Metrics:");
  console.log(`  Average domain agency:  ${(avgAgency * 100).toFixed(1)}%`);
  console.log(`  Average binding rate:   ${(avgBinding * 100).toFixed(1)}%`);
  console.log(`  Functors discovered:    ${foundCount}/${results.length}`);

  // Check composition graph
  console.log("\nComposition Graph:");
  const graph = tam.getSuccessfulGraph();
  if (graph.size > 0) {
    for (const [source, targets] of graph) {
      console.log(`  ${source} → ${targets.join(", ")}`);
    }
  } else {
    console.log("  (empty - no functors discovered)");
  }

  // Success criteria
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Success Criteria");
  console.log("═══════════════════════════════════════════════════════════\n");

  if (foundCount >= 2) {
    console.log("  ✓ SUCCESS: Multiple composition paths discovered!");
  } else {
    console.log("  ✗ Limited composition paths. May need more training.");
  }

  // Cleanup
  tam.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
