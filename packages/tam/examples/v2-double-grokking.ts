/**
 * V2 Double Grokking Experiment
 *
 * Tests if v2 exhibits the same two-phase learning as v1:
 * 1. First grokking: Dynamics learning (error drops suddenly)
 * 2. Second grokking: Epistemic learning (agency calibrates)
 *
 * Key mechanisms:
 * - Geometric binding: trajectory must fall within affordance cone
 * - Agency maximization: drives cones to narrow (increase concentration)
 * - Binding failures: provide feedback pressure to improve predictions
 *
 * Key metric: Calibration gap = |agency - binding_rate|
 * Perfect calibration: agency matches actual binding success
 */

import {
  Actor,
  StatsLogger,
  LatentLogger,
  evaluate,
  generateVisualization,
  generateLatentVisualization,
  ProportionalRefinementPolicy,
} from "../src/v2";

// 1D damped spring (same as v1)
const k = 1.0;
const b = 0.1;
const dt = 0.1;

type State = { x: number; v: number };

function step(state: State): State {
  const ax = -k * state.x - b * state.v;
  const newV = state.v + ax * dt;
  const newX = state.x + newV * dt;
  return { x: newX, v: newV };
}

// In-distribution: [-1, 1] x [-1, 1]
function randomStateInDist(): State {
  return {
    x: (Math.random() - 0.5) * 2,
    v: (Math.random() - 0.5) * 2,
  };
}

// Out-of-distribution: [1.5, 2.0] or [-2.0, -1.5]
function randomStateOutDist(): State {
  const sign = Math.random() < 0.5 ? -1 : 1;
  return {
    x: sign * (1.5 + Math.random() * 0.5),
    v: sign * (1.5 + Math.random() * 0.5),
  };
}

const embedState = (s: State) => [s.x, s.v];

interface Checkpoint {
  sample: number;
  // Dynamics learning
  errorInDist: number;
  errorOutDist: number;
  // Epistemic learning
  agencyInDist: number;
  agencyOutDist: number;
  bindingRateInDist: number;
  bindingRateOutDist: number;
  // Calibration quality
  calibrationGap: number;
  // System state
  portCount: number;
  trainBindingRate: number;
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  V2 Double Grokking Experiment");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Tracking two phases of learning:");
  console.log("  1. First grokking:  Learning WHAT happens (error drops)");
  console.log("  2. Second grokking: Learning CONFIDENCE (agency calibrates)\n");

  const samples = 10_000;
  const checkpointEvery = 100;

  // Set up logging
  const logger = await StatsLogger.create({
    experiment: "V2 Double Grokking",
    config: { samples, checkpointEvery },
  });

  // Set up latent space logger
  const latentLogger = LatentLogger.create(
    logger.getRunDir(),
    randomStateInDist,
    embedState,
    { numStates: 10 }
  );

  // Create actor with proportional refinement policy
  // - Geometric controller: adjusts concentration proportionally to binding violation
  // - Outside cone: widen proportionally to (normalizedDistance - 1.0)
  // - Inside cone: narrow proportionally to (1.0 - normalizedDistance)
  // - This creates strong narrowing pressure when binding succeeds consistently
  const actor = new Actor<State>(embedState, {
    proliferation: { enabled: true },
    // Note: Concentration now derived geometrically from CausalNet's uncertainty
    // refinementPolicy is deprecated in unified geometric architecture
    refinementPolicy: new ProportionalRefinementPolicy(
      0.5,   // narrowGain: (unused - kept for interface compatibility)
      1.0,   // widenGain: (unused - kept for interface compatibility)
      0.2    // minMargin: (unused - kept for interface compatibility)
    ),
  });

  // Generate fixed test sets (never used for training)
  const testSetInDist: State[] = [];
  for (let i = 0; i < 100; i++) {
    testSetInDist.push(randomStateInDist());
  }
  const testSetOutDist: State[] = [];
  for (let i = 0; i < 50; i++) {
    testSetOutDist.push(randomStateOutDist());
  }

  const checkpoints: Checkpoint[] = [];

  console.log("Sample | Error (In/Out) | Agency (In/Out) | Binding (In/Out) | Cal.Gap | Ports");
  console.log("-------|----------------|-----------------|------------------|---------|------");

  for (let i = 0; i < samples; i++) {
    const before = randomStateInDist();
    const after = step(before);

    await actor.observe({
      before: { state: before },
      after: { state: after },
    });

    if ((i + 1) % checkpointEvery === 0) {
      await actor.flush();

      // Evaluate on FIXED test sets (never trained on)
      let testIdx = 0;
      const evalInDist = evaluate(
        actor,
        () => {
          const b = testSetInDist[testIdx % testSetInDist.length]!;
          testIdx++;
          return { before: b, after: step(b) };
        },
        embedState,
        testSetInDist.length
      );

      let testIdx2 = 0;
      const evalOutDist = evaluate(
        actor,
        () => {
          const b = testSetOutDist[testIdx2 % testSetOutDist.length]!;
          testIdx2++;
          return { before: b, after: step(b) };
        },
        embedState,
        testSetOutDist.length
      );

      // Compute calibration gap (how well does agency predict binding?)
      const calibrationGap = Math.abs(
        evalInDist.avgAgency - evalInDist.testBindingRate
      );

      // Get training metrics from analytics (actual training data)
      const analytics = actor.getAnalytics();
      let totalSamples = 0;
      let weightedBindingRate = 0;
      let weightedAgency = 0;
      let weightedError = 0;

      for (const port of analytics.ports) {
        totalSamples += port.samples;
        weightedBindingRate += port.bindingRate * port.samples;
        weightedAgency += port.avgAgency * port.samples;
        weightedError += port.avgError * port.samples;
      }

      const trainBindingRate = totalSamples > 0 ? weightedBindingRate / totalSamples : 0;
      const trainAgency = totalSamples > 0 ? weightedAgency / totalSamples : 0;
      const trainError = totalSamples > 0 ? weightedError / totalSamples : 0;

      const checkpoint: Checkpoint = {
        sample: i + 1,
        errorInDist: evalInDist.avgError,
        errorOutDist: evalOutDist.avgError,
        agencyInDist: evalInDist.avgAgency,
        agencyOutDist: evalOutDist.avgAgency,
        bindingRateInDist: evalInDist.testBindingRate,
        bindingRateOutDist: evalOutDist.testBindingRate,
        calibrationGap,
        portCount: actor.getPortCount(),
        trainBindingRate,
      };

      checkpoints.push(checkpoint);

      // Compute calibration gaps
      const testCalibrationGap = Math.abs(evalInDist.avgAgency - evalInDist.testBindingRate);
      const trainCalibrationGap = Math.abs(trainAgency - trainBindingRate);

      // Compute average cone radius to track narrowing pressure
      let totalRadius = 0;
      let radiusCount = 0;
      for (let j = 0; j < 10; j++) {
        const testState = randomStateInDist();
        const pred = actor.predict(testState);
        for (const r of pred.cone.radius) {
          totalRadius += r;
          radiusCount++;
        }
      }
      const avgConeRadius = radiusCount > 0 ? totalRadius / radiusCount : 0;

      // Log checkpoint metrics
      logger.logCheckpoint({
        sample: i + 1,
        portCount: actor.getPortCount(),
        testAgency: evalInDist.avgAgency,
        testError: evalInDist.avgError,
        testBindingRate: evalInDist.testBindingRate,
        trainAgency,
        trainError,
        trainBindingRate,
        testCalibrationGap,
        trainCalibrationGap,
        avgConeRadius,
      });

      // Log per-port metrics
      logger.logPorts(
        analytics.ports.map((port) => ({
          portIdx: port.portIdx,
          sample: i + 1,
          samples: port.samples,
          bindings: port.bindings,
          bindingRate: port.bindingRate,
          avgAgency: port.avgAgency,
          agencyStdDev: port.agencyStdDev,
          avgError: port.avgError,
          errorStdDev: port.errorStdDev,
        }))
      );

      // Log latent space state
      latentLogger.logCheckpoint(i + 1, actor);

      console.log(
        `${checkpoint.sample.toString().padStart(6)} | ` +
          `${checkpoint.errorInDist.toFixed(4)}/${checkpoint.errorOutDist.toFixed(4)} | ` +
          `${(checkpoint.agencyInDist * 100).toFixed(1).padStart(4)}%/${(checkpoint.agencyOutDist * 100).toFixed(1).padStart(4)}% | ` +
          `${(checkpoint.bindingRateInDist * 100).toFixed(1).padStart(5)}%/${(checkpoint.bindingRateOutDist * 100).toFixed(1).padStart(5)}% | ` +
          `${checkpoint.calibrationGap.toFixed(3).padStart(7)} | ` +
          `${checkpoint.portCount.toString().padStart(5)}`
      );
    }
  }

  logger.close();
  latentLogger.close();

  await generateVisualization(logger.getRunDir());
  await generateLatentVisualization(logger.getRunDir());

  // Cleanup
  actor.dispose();

  // Analyze grokking moments
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Grokking Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  // First grokking: sudden error drop
  let firstGrokkingSample: number | null = null;
  let maxErrorDrop = 0;

  for (let i = 1; i < checkpoints.length; i++) {
    const errorDrop = checkpoints[i - 1]!.errorInDist - checkpoints[i]!.errorInDist;
    if (errorDrop > maxErrorDrop) {
      maxErrorDrop = errorDrop;
      firstGrokkingSample = checkpoints[i]!.sample;
    }
  }

  // Second grokking: sudden agency improvement
  let secondGrokkingSample: number | null = null;
  let maxAgencyJump = 0;

  for (let i = 1; i < checkpoints.length; i++) {
    const agencyJump = checkpoints[i]!.agencyInDist - checkpoints[i - 1]!.agencyInDist;
    if (agencyJump > maxAgencyJump && checkpoints[i]!.sample > (firstGrokkingSample || 0)) {
      maxAgencyJump = agencyJump;
      secondGrokkingSample = checkpoints[i]!.sample;
    }
  }

  // Calibration convergence: when calibration gap becomes consistently small
  let calibrationConvergenceSample: number | null = null;
  const calibrationThreshold = 0.1;
  let consecutiveGood = 0;

  for (let i = 0; i < checkpoints.length; i++) {
    if (checkpoints[i]!.calibrationGap < calibrationThreshold) {
      consecutiveGood++;
      if (consecutiveGood >= 3 && !calibrationConvergenceSample) {
        calibrationConvergenceSample = checkpoints[i]!.sample;
      }
    } else {
      consecutiveGood = 0;
    }
  }

  if (firstGrokkingSample) {
    console.log(`✓ First Grokking (Dynamics): Sample ~${firstGrokkingSample}`);
    console.log(`  Error dropped by ${maxErrorDrop.toFixed(4)}`);
    console.log(`  Interpretation: Model learned the spring dynamics\n`);
  } else {
    console.log(`⚠ First grokking not clearly observed`);
    console.log(`  Max error drop: ${maxErrorDrop.toFixed(4)}\n`);
  }

  if (secondGrokkingSample) {
    console.log(`✓ Second Grokking (Epistemics): Sample ~${secondGrokkingSample}`);
    console.log(`  Agency jumped by ${(maxAgencyJump * 100).toFixed(1)}%`);
    console.log(`  Interpretation: Model learned when to be confident\n`);
  } else {
    console.log(`⚠ Second grokking not clearly observed`);
    console.log(`  Max agency jump: ${(maxAgencyJump * 100).toFixed(1)}%\n`);
  }

  if (calibrationConvergenceSample) {
    console.log(`✓ Calibration Convergence: Sample ~${calibrationConvergenceSample}`);
    console.log(`  Calibration gap fell below ${calibrationThreshold}`);
    console.log(`  Interpretation: Model's confidence matches actual performance\n`);
  } else {
    console.log(`⚠ Calibration not yet converged`);
    console.log(`  Final gap: ${checkpoints[checkpoints.length - 1]!.calibrationGap.toFixed(3)}\n`);
  }

  // Phase summary
  const initial = checkpoints[0]!;
  const final = checkpoints[checkpoints.length - 1]!;

  console.log("Phase Summary:");
  console.log(`  Initial → Final:`);
  console.log(`    Error (in-dist):     ${initial.errorInDist.toFixed(4)} → ${final.errorInDist.toFixed(4)}`);
  console.log(`    Agency (in-dist):    ${(initial.agencyInDist * 100).toFixed(1)}% → ${(final.agencyInDist * 100).toFixed(1)}%`);
  console.log(`    Binding (in-dist):   ${(initial.bindingRateInDist * 100).toFixed(1)}% → ${(final.bindingRateInDist * 100).toFixed(1)}%`);
  console.log(`    Calibration Gap:     ${initial.calibrationGap.toFixed(3)} → ${final.calibrationGap.toFixed(3)}`);
  console.log(`    Ports:               ${initial.portCount} → ${final.portCount}`);

  console.log(`\nVisualizations:`);
  console.log(`  Metrics:      ${logger.getRunDir()}/visualization.html`);
  console.log(`  Latent Space: ${logger.getRunDir()}/latent-visualization.html`);
  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
