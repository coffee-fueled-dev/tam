/**
 * Encoder Learning Experiment
 *
 * Demonstrates composition-aware encoder learning:
 * 1. Learn primitive domains (shift, scale) with hand-crafted encoders
 * 2. Learn a composite domain (affine: ax + b) by:
 *    a. Providing only raw features (stateToRaw)
 *    b. Specifying composition targets (shift, scale)
 *    c. Letting the encoder learn automatically
 * 3. Compare agency against hand-crafted encoder baseline
 */

import { TAM, type DomainSpec } from "../../src";

// ============================================================================
// Primitive Domains (with hand-crafted encoders)
// ============================================================================

const RANGE = 10;

const shift: DomainSpec<number> = {
  randomState: () => (Math.random() - 0.5) * 2 * RANGE,
  simulate: (x) => x + (Math.random() - 0.5) * 2, // +/- 1
  embedder: (x) => [x / RANGE], // Normalized
  embeddingDim: 1,
};

const scale: DomainSpec<number> = {
  randomState: () => 0.5 + Math.random() * 4.5, // [0.5, 5]
  simulate: (x) => Math.max(0.5, Math.min(5, x * (0.8 + Math.random() * 0.4))), // ±20%
  embedder: (x) => [(x - 2.75) / 2.25], // Normalized around [0.5, 5] center
  embeddingDim: 1,
};

// ============================================================================
// Affine Domain: ax + b
//
// Two versions:
// 1. Hand-crafted encoder (baseline)
// 2. Composition-learned encoder (experimental)
// ============================================================================

const AFFINE_RANGE = 10;

// Shared dynamics
const affineSimulate = (x: number): number => {
  const a = 0.8 + Math.random() * 0.4; // [0.8, 1.2]
  const b = (Math.random() - 0.5) * 2; // [-1, 1]
  return Math.max(-AFFINE_RANGE, Math.min(AFFINE_RANGE, a * x + b));
};

// Baseline: hand-crafted encoder
const affineHandcrafted: DomainSpec<number> = {
  randomState: () => (Math.random() - 0.5) * 2 * AFFINE_RANGE,
  simulate: affineSimulate,
  embedder: (x) => [x / AFFINE_RANGE],
  embeddingDim: 1,
};

// Experimental: composition-learned encoder
const affineComposition: DomainSpec<number> = {
  randomState: () => (Math.random() - 0.5) * 2 * AFFINE_RANGE,
  simulate: affineSimulate,
  // No embedder! We learn it via composition
  stateToRaw: (x) => [x], // Raw state value
  rawDim: 1,
  composeWith: ["shift", "scale"], // Learn encoder relative to these
  embeddingDim: 1,
};

// ============================================================================
// Run Experiment
// ============================================================================

async function main() {
  console.log("=".repeat(60));
  console.log("ENCODER LEARNING EXPERIMENT");
  console.log("=".repeat(60));
  console.log();

  const TRAINING = { epochs: 200, samplesPerEpoch: 50, flushFrequency: 20 };

  // Create TAM instance
  const tam = new TAM();

  // Step 1: Learn primitive domains
  console.log("Step 1: Learning primitive domains...");
  console.log("-".repeat(40));

  const shiftPort = await tam.learn("shift", shift, TRAINING);
  console.log(`  shift:  Agency = ${(shiftPort.getAgency() * 100).toFixed(1)}%`);

  const scalePort = await tam.learn("scale", scale, TRAINING);
  console.log(`  scale:  Agency = ${(scalePort.getAgency() * 100).toFixed(1)}%`);
  console.log();

  // Step 2a: Learn affine with hand-crafted encoder (baseline)
  console.log("Step 2a: Learning affine (hand-crafted encoder)...");
  console.log("-".repeat(40));

  const affineHandPort = await tam.learn(
    "affine-handcrafted",
    affineHandcrafted,
    TRAINING
  );
  const handcraftedAgency = affineHandPort.getAgency();
  console.log(`  affine-handcrafted: Agency = ${(handcraftedAgency * 100).toFixed(1)}%`);
  console.log();

  // Step 2b: Learn affine with composition-learned encoder
  console.log("Step 2b: Learning affine (composition-learned encoder)...");
  console.log("-".repeat(40));
  console.log("  Targets: shift, scale");
  console.log("  Learning encoder from composition objective...");

  const affineCompPort = await tam.learn(
    "affine-composed",
    affineComposition,
    TRAINING
  );
  const composedAgency = affineCompPort.getAgency();
  console.log(`  affine-composed: Agency = ${(composedAgency * 100).toFixed(1)}%`);
  console.log();

  // Step 3: Compare results
  console.log("Step 3: Comparison");
  console.log("-".repeat(40));
  console.log(`  Hand-crafted encoder: ${(handcraftedAgency * 100).toFixed(1)}% agency`);
  console.log(`  Composition-learned:  ${(composedAgency * 100).toFixed(1)}% agency`);

  const diff = composedAgency - handcraftedAgency;
  const verdict =
    Math.abs(diff) < 0.05
      ? "≈ comparable"
      : diff > 0
        ? "✓ composition-learned better"
        : "✗ hand-crafted better";
  console.log(`  Difference: ${(diff * 100).toFixed(1)}% (${verdict})`);
  console.log();

  // Step 4: Test functor discovery
  console.log("Step 4: Functor Discovery");
  console.log("-".repeat(40));

  const shiftToAffine = await tam.findPath("shift", "affine-composed");
  console.log(
    `  shift → affine-composed: ${shiftToAffine ? `Found (${(shiftToAffine.totalBindingRate * 100).toFixed(1)}%)` : "Not found"}`
  );

  const scaleToAffine = await tam.findPath("scale", "affine-composed");
  console.log(
    `  scale → affine-composed: ${scaleToAffine ? `Found (${(scaleToAffine.totalBindingRate * 100).toFixed(1)}%)` : "Not found"}`
  );
  console.log();

  console.log("=".repeat(60));
  console.log("EXPERIMENT COMPLETE");
  console.log("=".repeat(60));

  // Clean up
  tam.dispose();
}

main().catch(console.error);

