/**
 * Analyze Port Organization
 *
 * Investigates what ports are specializing on by examining:
 * 1. Raw port embeddings (4D for double pendulum)
 * 2. Which test states each port handles best
 * 3. Correlation with physical quantities (energy, angle, etc.)
 *
 * Usage: bun examples/analyze-port-organization.ts <path-to-latent.jsonl>
 */

import { readFile } from "node:fs/promises";

interface LatentCheckpoint {
  sample: number;
  ports: Array<{
    portIdx: number;
    embedding: number[];
    commitments: Array<{
      stateIdx: number;
      concentration: number;
      selected: boolean;
    }>;
  }>;
}

// Double pendulum state interpretation
interface PendulumState {
  theta1: number;
  theta2: number;
  omega1: number;
  omega2: number;
}

function computeEnergy(s: PendulumState): number {
  const m1 = 1.0;
  const m2 = 1.0;
  const L1 = 1.0;
  const L2 = 1.0;
  const g = 9.81;

  // Kinetic energy (rotational)
  const KE = 0.5 * m1 * (L1 * s.omega1) ** 2 + 0.5 * m2 * ((L1 * s.omega1) ** 2 + (L2 * s.omega2) ** 2);

  // Potential energy (height)
  const h1 = -L1 * Math.cos(s.theta1);
  const h2 = h1 - L2 * Math.cos(s.theta2);
  const PE = m1 * g * h1 + m2 * g * h2;

  return KE + PE;
}

async function analyzeLatent(latentPath: string) {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Port Organization Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  const latentText = await readFile(latentPath, "utf-8");
  const checkpoints: LatentCheckpoint[] = latentText
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line));

  // Analyze final checkpoint
  const final = checkpoints[checkpoints.length - 1]!;
  console.log(`Final checkpoint: Sample ${final.sample}`);
  console.log(`Port count: ${final.ports.length}\n`);

  if (final.ports.length === 0) {
    console.log("No ports to analyze!");
    return;
  }

  // Check dimensionality
  const embeddingDim = final.ports[0]!.embedding.length;
  console.log(`Embedding dimension: ${embeddingDim}D\n`);

  // 1. Analyze embedding distribution
  console.log("Port Embeddings:");
  console.log("Port | Embedding (first 4 dims)");
  console.log("-----|-------------------------");
  for (const port of final.ports) {
    const embStr = port.embedding.slice(0, 4).map(v => v.toFixed(3)).join(", ");
    console.log(`  ${port.portIdx.toString().padStart(2)} | [${embStr}]`);
  }

  // 2. Compute pairwise distances
  if (final.ports.length > 1) {
    console.log("\nPairwise Port Distances (Euclidean):");
    for (let i = 0; i < final.ports.length; i++) {
      for (let j = i + 1; j < final.ports.length; j++) {
        const emb1 = final.ports[i]!.embedding;
        const emb2 = final.ports[j]!.embedding;
        const dist = Math.sqrt(
          emb1.reduce((sum, v, k) => sum + (v - emb2[k]!) ** 2, 0)
        );
        console.log(`  Port ${i} ↔ Port ${j}: ${dist.toFixed(3)}`);
      }
    }
  }

  // 3. Analyze commitment patterns (which states each port handles)
  console.log("\nPort Commitment Patterns:");
  console.log("Port | Selected States | Avg Concentration");
  console.log("-----|-----------------|------------------");
  for (const port of final.ports) {
    const selectedCommits = port.commitments.filter(c => c.selected);
    const avgConc = selectedCommits.length > 0
      ? selectedCommits.reduce((sum, c) => sum + c.concentration, 0) / selectedCommits.length
      : 0;
    console.log(`  ${port.portIdx.toString().padStart(2)} | ${selectedCommits.length.toString().padStart(15)} | ${avgConc.toFixed(3).padStart(16)}`);
  }

  // 4. For 4D embeddings (double pendulum), correlate with physical quantities
  if (embeddingDim === 4) {
    console.log("\nPhysical Quantity Analysis (interpreting as [θ1, θ2, ω1, ω2]):");
    console.log("Port | Avg θ1   | Avg θ2   | Avg ω1   | Avg ω2   | Avg Energy");
    console.log("-----|----------|----------|----------|----------|------------");

    for (const port of final.ports) {
      const emb = port.embedding;
      const state: PendulumState = {
        theta1: emb[0]!,
        theta2: emb[1]!,
        omega1: emb[2]!,
        omega2: emb[3]!,
      };
      const energy = computeEnergy(state);

      console.log(
        `  ${port.portIdx.toString().padStart(2)} | ` +
        `${state.theta1.toFixed(3).padStart(8)} | ` +
        `${state.theta2.toFixed(3).padStart(8)} | ` +
        `${state.omega1.toFixed(3).padStart(8)} | ` +
        `${state.omega2.toFixed(3).padStart(8)} | ` +
        `${energy.toFixed(3).padStart(10)}`
      );
    }
  }

  // 5. Check if embeddings lie on a line (1D manifold)
  if (final.ports.length >= 3) {
    console.log("\nDimensionality Check:");

    // Compute variance along first principal component vs others
    const embeddings = final.ports.map(p => p.embedding);
    const mean = embeddings[0]!.map((_, d) =>
      embeddings.reduce((sum, emb) => sum + emb[d]!, 0) / embeddings.length
    );

    const centered = embeddings.map(emb =>
      emb.map((v, d) => v - mean[d]!)
    );

    // Simple variance computation per dimension
    const variances = mean.map((_, d) =>
      centered.reduce((sum, emb) => sum + emb[d]! ** 2, 0) / centered.length
    );

    const totalVar = variances.reduce((sum, v) => sum + v, 0);
    const sortedVars = [...variances].sort((a, b) => b - a);

    console.log(`Total variance: ${totalVar.toFixed(6)}`);
    console.log(`Top 3 dimension variances: ${sortedVars.slice(0, 3).map(v => v.toFixed(6)).join(", ")}`);
    console.log(`Variance explained by top dimension: ${((sortedVars[0]! / totalVar) * 100).toFixed(1)}%`);

    if ((sortedVars[0]! / totalVar) > 0.9) {
      console.log("\n✓ Ports are approximately 1D (>90% variance in one direction)");
      console.log("  Interpretation: Ports organize along a single axis");
    } else if ((sortedVars[0]! / totalVar) > 0.7) {
      console.log("\n• Ports are mostly 1D (>70% variance in one direction)");
    } else {
      console.log("\n• Ports use multiple dimensions");
    }
  }

  // 6. Track evolution across checkpoints
  if (checkpoints.length > 1) {
    console.log("\n\nPort Count Evolution:");
    const samples = checkpoints.map(c => c.sample);
    const portCounts = checkpoints.map(c => c.ports.length);

    console.log(`Samples: ${samples.slice(0, 5).join(", ")} ... ${samples.slice(-3).join(", ")}`);
    console.log(`Ports:   ${portCounts.slice(0, 5).join(", ")}     ... ${portCounts.slice(-3).join(", ")}`);

    const proliferationPoints = [];
    for (let i = 1; i < checkpoints.length; i++) {
      if (portCounts[i]! > portCounts[i - 1]!) {
        proliferationPoints.push(`Sample ${samples[i]}: ${portCounts[i - 1]} → ${portCounts[i]}`);
      }
    }

    if (proliferationPoints.length > 0) {
      console.log("\nProliferation events:");
      for (const event of proliferationPoints) {
        console.log(`  ${event}`);
      }
    } else {
      console.log("\nNo proliferation events (stayed at 1 port)");
    }
  }

  console.log("\n✓ Analysis complete!");
}

// CLI usage
if (import.meta.main) {
  const latentPath = process.argv[2];
  if (!latentPath) {
    console.error("Usage: bun examples/analyze-port-organization.ts <path-to-latent.jsonl>");
    console.error("Example: bun examples/analyze-port-organization.ts runs/v2-double-pendulum-123456/latent.jsonl");
    process.exit(1);
  }

  analyzeLatent(latentPath).catch((error) => {
    console.error("Error analyzing latent space:", error);
    process.exit(1);
  });
}

export { analyzeLatent };
