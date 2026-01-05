/**
 * V2 Anisotropic Cone Demonstration
 *
 * Demonstrates context-dependent anisotropic cones in v2:
 * - Dimension 0: Deterministic (x' = -0.5*x)
 * - Dimension 1: Noisy (random)
 *
 * Expected behavior:
 * - Cone radius[0] should be NARROW (low variance, predictable)
 * - Cone radius[1] should be WIDE (high variance, unpredictable)
 * - No global dimensional attention needed - emerges from per-prediction variance
 *
 * This shows v2's approach: anisotropy is context-dependent rather than
 * a global property of ports (unlike v1's learned attention weights).
 */

import { Actor } from "../src/v2";

type State = { x: number; noise: number };

function step(state: State): State {
  return {
    x: state.x * 0.5,           // Deterministic decay
    noise: (Math.random() - 0.5) * 10, // Much wider noise range
  };
}

function randomState(): State {
  return {
    x: (Math.random() - 0.5) * 2,
    noise: (Math.random() - 0.5) * 2,
  };
}

const embedState = (s: State) => [s.x, s.noise];

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  V2 Anisotropic Cone Demonstration");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Environment:");
  console.log("  Dim 0 (x):     Deterministic (x' = 0.5*x)");
  console.log("  Dim 1 (noise): Random noise\n");
  console.log("Expected: Cone should be narrow in x, wide in noise\n");

  const actor = new Actor<State>(embedState, {
    proliferation: { enabled: false },
    causal: {
      minVariance: 0.1,  // Lower floor to allow narrow cones on predictable dimension
    },
  });

  // Train for a bit
  console.log("Training 2000 samples...\n");
  for (let i = 0; i < 2000; i++) {
    const before = randomState();
    const after = step(before);
    await actor.observe({
      before: { state: before },
      after: { state: after },
    });
  }
  await actor.flush();

  // Show predictions at different states
  console.log("Predictions and cone anisotropy:\n");
  console.log("State (x, noise)  | Cone Radius (x, noise) | Agency");
  console.log("------------------|-------------------------|-------");

  for (let i = 0; i < 5; i++) {
    const testState = randomState();
    const pred = actor.predict(testState);

    console.log(
      `(${testState.x.toFixed(2).padStart(5)}, ${testState.noise.toFixed(2).padStart(5)}) | ` +
      `(${pred.cone.radius[0]!.toFixed(3).padStart(5)}, ${pred.cone.radius[1]!.toFixed(3).padStart(5)})      | ` +
      `${(pred.agency * 100).toFixed(1)}%`
    );
  }

  // Compute average anisotropy ratio
  let totalRatio = 0;
  const samples = 50;
  for (let i = 0; i < samples; i++) {
    const testState = randomState();
    const pred = actor.predict(testState);
    const ratio = pred.cone.radius[1]! / Math.max(pred.cone.radius[0]!, 1e-6);
    totalRatio += ratio;
  }
  const avgRatio = totalRatio / samples;

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log(`Average anisotropy ratio (noise_radius / x_radius): ${avgRatio.toFixed(2)}x`);
  console.log();

  if (avgRatio > 2.0) {
    console.log("✓ Strong anisotropy detected!");
    console.log("  Cone is wider in unpredictable dimension (noise)");
    console.log("  Cone is narrower in predictable dimension (x)");
  } else if (avgRatio > 1.2) {
    console.log("✓ Moderate anisotropy detected");
    console.log("  System recognizes difference between dimensions");
  } else {
    console.log("⚠ Weak anisotropy (nearly isotropic)");
    console.log("  Cones are similar width in both dimensions");
  }

  console.log("\nKey insight:");
  console.log("  V2's anisotropy is CONTEXT-DEPENDENT (per-prediction)");
  console.log("  Not a global port property like v1's attention weights");
  console.log("  Emerges naturally from CausalNet's per-dimension variance");

  actor.dispose();
  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
