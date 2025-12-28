/**
 * Simple test for ContextualMixtureDeltaPort:
 * - Same action "UP" has TWO modes depending on context feature "blockedAbove":
 *    blockedAbove = 0  => delta (0, -1)
 *    blockedAbove = 1  => delta (0,  0)
 *
 * The gating model should learn to route contexts to the right component.
 *
 * How to run (Node 18+):
 *   1) Put this file next to your implementation (or adjust import paths)
 *   2) tsc && node dist/test.js
 *
 * This uses only console output + basic assertions.
 */

import {
  ContextualPortBank,
  simpleGridEncoders,
  type SimpleGridState,
  type Situation,
} from "../index"; // <-- change to your file name

function approxEq(a: number, b: number, eps = 0.25): boolean {
  return Math.abs(a - b) <= eps;
}

function vecApproxEq(v: number[], w: number[], eps = 0.25): boolean {
  if (v.length !== w.length) return false;
  for (let i = 0; i < v.length; i++)
    if (!approxEq(v[i]!, w[i]!, eps)) return false;
  return true;
}

function makeSit(
  x: number,
  y: number,
  blockedAbove: boolean
): Situation<SimpleGridState, {}> {
  return { state: { x, y, blockedAbove }, context: {} };
}

function runTest() {
  // Config tuned for the split heuristic
  const bank = new ContextualPortBank(simpleGridEncoders, {
    maxComponents: 4,
    goodFitLogLikeThreshold: -6,
    noveltyLogLikeThreshold: -20,
    gateLr: 0.25,
    meanLr: 0.15,
    varLr: 0.1,
    minComponentSeparationL2: 0.25,
    minVar: 1e-3,
    maxVar: 10,
    // Split heuristic settings
    reservoirSize: 20,
    splitSseRatio: 0.6,
    splitMinSamples: 6,
    splitCheckInterval: 3,
  });

  // Generate training data
  // Mode A: normal move up => (0,-1)
  // Mode B: blocked => (0,0)
  // We interleave contexts so gating must learn the dependence.
  const N = 200;
  for (let i = 0; i < N; i++) {
    const blocked = i % 2 === 0; // alternate
    const x = (i * 7) % 13;
    const y = blocked ? 0 : 5 + ((i * 3) % 10);

    const before = makeSit(x, y, blocked);

    const after = blocked
      ? makeSit(x, y, blocked) // no movement
      : makeSit(x, y - 1, blocked); // move up by 1

    bank.observe({ action: "UP", before, after });
  }

  // Query predictions in each context
  const testFree = makeSit(7, 10, false);
  const testBlocked = makeSit(7, 0, true);

  const predFree = bank.predict("UP", testFree, 2);
  const predBlocked = bank.predict("UP", testBlocked, 2);

  console.log("Predictions (free):", predFree);
  console.log("Predictions (blocked):", predBlocked);

  // Expect top prediction to be near [0,-1] in free context
  if (predFree.length === 0) throw new Error("No predictions for free context");
  const topFree = predFree[0]!.delta;

  // Expect top prediction to be near [0,0] in blocked context
  if (predBlocked.length === 0)
    throw new Error("No predictions for blocked context");
  const topBlocked = predBlocked[0]!.delta;

  const okFree = vecApproxEq(topFree, [0, -1], 0.35);
  const okBlocked = vecApproxEq(topBlocked, [0, 0], 0.35);

  console.log("Top free delta:", topFree, "OK?", okFree);
  console.log("Top blocked delta:", topBlocked, "OK?", okBlocked);

  if (!okFree)
    throw new Error(
      `Expected free-context UP ~ [0,-1], got ${JSON.stringify(topFree)}`
    );
  if (!okBlocked)
    throw new Error(
      `Expected blocked-context UP ~ [0,0], got ${JSON.stringify(topBlocked)}`
    );

  // Optional: sanity check that the two contexts score their correct deltas higher
  const scoreFreeCorrect = bank.score("UP", testFree, [0, -1]);
  const scoreFreeWrong = bank.score("UP", testFree, [0, 0]);
  const scoreBlockedCorrect = bank.score("UP", testBlocked, [0, 0]);
  const scoreBlockedWrong = bank.score("UP", testBlocked, [0, -1]);

  console.log("Scores:");
  console.log(
    " free  correct [0,-1]:",
    scoreFreeCorrect,
    " vs wrong [0,0]:",
    scoreFreeWrong
  );
  console.log(
    " block correct [0,0]:",
    scoreBlockedCorrect,
    " vs wrong [0,-1]:",
    scoreBlockedWrong
  );

  if (!(scoreFreeCorrect > scoreFreeWrong)) {
    throw new Error("Expected free context to score [0,-1] higher than [0,0]");
  }
  if (!(scoreBlockedCorrect > scoreBlockedWrong)) {
    throw new Error(
      "Expected blocked context to score [0,0] higher than [0,-1]"
    );
  }

  console.log("✅ Test passed.");
}

runTest();
