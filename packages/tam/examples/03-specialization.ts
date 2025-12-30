/**
 * Experiment 03: Multi-Mode Specialization
 *
 * Demonstrates:
 * - Port proliferation for different behaviors
 * - Specialist discovery via agency-based selection
 * - Multiple modes in a single domain
 *
 * Expected outcome:
 * - 4 specialist ports (one per drift mode)
 * - High agency when mode is clear
 * - Good coverage (reality honors commitments)
 */

import { runExperiment, driftModes, saveResultsToJson } from "./harness";

async function main() {
  const result = await runExperiment("Multi-Mode Specialization", {
    domain: driftModes,
    episodes: 500,
    checkpointEvery: 100,
    testSize: 100,
    bankConfig: {
      embeddingDim: driftModes.embeddingDim,
      specialistThreshold: 0.7, // Encourage specialization
    },
  });

  // Save results to JSON
  await saveResultsToJson(result, "examples/results/03-specialization.json");

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Port Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  const ports = result.bank!.getAllPorts();
  console.log(`Total ports discovered: ${ports.length}`);
  console.log(`Expected: 4 (one per mode: left/right/up/down)\n`);

  if (ports.length >= 4) {
    console.log("✓ Successfully discovered specialists for different modes");
  } else if (ports.length >= 2) {
    console.log("⚠ Partial specialization (may need more training)");
  } else {
    console.log("✗ No specialization (single generalist port)");
  }

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Takeaway");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("  TAM automatically discovers distinct behaviors.");
  console.log("  No manual mode switching - agency selects specialists.");
  console.log("  Each specialist commits to narrow cones for its mode.");

  // Cleanup
  result.bank?.dispose();

  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
