/**
 * V2 Forced Proliferation - Piecewise Dynamics
 *
 * Tests whether complex piecewise dynamics force port proliferation.
 *
 * System: Piecewise linear dynamics with hard boundaries
 * - Region 1 (x >= 0): Expand in x, contract in y
 * - Region 2 (x < 0):  Contract in x, expand in y
 *
 * Hypothesis: Clustering detects "barbell" shape in trajectory distribution
 * and triggers proliferation to cover each region separately.
 */

import {
  Actor,
  StatsLogger,
  LatentLogger,
  evaluate,
  generateVisualization,
  generateLatentVisualization,
} from "../src/v2";

// State: 2D position
interface State {
  x: number;
  y: number;
}

/**
 * Piecewise linear dynamics with sharp boundary at x=0.
 * Different dynamics in each region create discontinuity.
 */
function step(state: State): State {
  const dt = 0.1;

  // Base movement
  let dx = 0;
  let dy = 0;

  // Apply piecewise dynamics based on current position
  if (state.x >= 0) {
    // Region 1: Expand in x, contract in y
    dx = 0.5 * state.x * dt;
    dy = -0.3 * state.y * dt;
  } else {
    // Region 2: Contract in x, expand in y
    dx = -0.3 * state.x * dt;
    dy = 0.5 * state.y * dt;
  }

  return {
    x: state.x + dx,
    y: state.y + dy,
  };
}

function randomState(): State {
  return {
    x: (Math.random() - 0.5) * 4.0,  // [-2, 2]
    y: (Math.random() - 0.5) * 4.0,
  };
}

const embedState = (s: State) => [s.x, s.y];

interface Checkpoint {
  sample: number;
  errorRegion1: number;
  errorRegion2: number;
  agencyRegion1: number;
  agencyRegion2: number;
  bindingRateRegion1: number;
  bindingRateRegion2: number;
  calibrationGapRegion1: number;
  calibrationGapRegion2: number;
  portCount: number;
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  V2 Forced Proliferation - Piecewise Dynamics");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("System: Piecewise linear with discontinuity at x=0");
  console.log("  Region 1 (x >= 0): dx/dt = +0.5x, dy/dt = -0.3y");
  console.log("  Region 2 (x <  0): dx/dt = -0.3x, dy/dt = +0.5y");
  console.log("\nHypothesis: Clustering detects barbell shape and triggers proliferation\n");

  const samples = 2000;
  const checkpointEvery = 100;

  // Set up logging
  const logger = await StatsLogger.create({
    experiment: "V2 Forced Proliferation",
    config: { samples, checkpointEvery },
  });

  // Set up latent space logger
  const latentLogger = LatentLogger.create(
    logger.getRunDir(),
    randomState,
    embedState,
    { numStates: 15 }
  );

  const actor = new Actor<State>(embedState, {
    proliferation: { enabled: true },
  });

  // Generate fixed test sets for each region
  const testSetRegion1: State[] = [];
  for (let i = 0; i < 50; i++) {
    testSetRegion1.push({
      x: Math.random() * 2.0,      // [0, 2]
      y: (Math.random() - 0.5) * 4.0,
    });
  }
  const testSetRegion2: State[] = [];
  for (let i = 0; i < 50; i++) {
    testSetRegion2.push({
      x: -Math.random() * 2.0,     // [-2, 0]
      y: (Math.random() - 0.5) * 4.0,
    });
  }

  const checkpoints: Checkpoint[] = [];

  console.log("Sample | Error (R1/R2) | Agency (R1/R2) | Binding (R1/R2) | Cal.Gap (R1/R2) | Ports");
  console.log("-------|---------------|----------------|-----------------|-----------------|------");

  for (let i = 0; i < samples; i++) {
    const before = randomState();
    const after = step(before);

    await actor.observe({
      before: { state: before },
      after: { state: after },
    });

    if ((i + 1) % checkpointEvery === 0) {
      await actor.flush();

      // Evaluate on FIXED test sets (never trained on)
      let testIdx1 = 0;
      const evalRegion1 = evaluate(
        actor,
        () => {
          const b = testSetRegion1[testIdx1 % testSetRegion1.length]!;
          testIdx1++;
          return { before: b, after: step(b) };
        },
        embedState,
        testSetRegion1.length
      );

      let testIdx2 = 0;
      const evalRegion2 = evaluate(
        actor,
        () => {
          const b = testSetRegion2[testIdx2 % testSetRegion2.length]!;
          testIdx2++;
          return { before: b, after: step(b) };
        },
        embedState,
        testSetRegion2.length
      );

      // Compute calibration gaps
      const calibrationGapRegion1 = Math.abs(evalRegion1.avgAgency - evalRegion1.testBindingRate);
      const calibrationGapRegion2 = Math.abs(evalRegion2.avgAgency - evalRegion2.testBindingRate);

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
        errorRegion1: evalRegion1.avgError,
        errorRegion2: evalRegion2.avgError,
        agencyRegion1: evalRegion1.avgAgency,
        agencyRegion2: evalRegion2.avgAgency,
        bindingRateRegion1: evalRegion1.testBindingRate,
        bindingRateRegion2: evalRegion2.testBindingRate,
        calibrationGapRegion1,
        calibrationGapRegion2,
        portCount: actor.getPortCount(),
      };

      checkpoints.push(checkpoint);

      // Compute calibration gaps
      const testCalibrationGap = (calibrationGapRegion1 + calibrationGapRegion2) / 2;
      const trainCalibrationGap = Math.abs(trainAgency - trainBindingRate);

      // Compute average cone radius
      let totalRadius = 0;
      let radiusCount = 0;
      for (let j = 0; j < 10; j++) {
        const testState = randomState();
        const pred = actor.predict(testState);
        for (const r of pred.cone.radius) {
          totalRadius += r;
          radiusCount++;
        }
      }
      const avgConeRadius = radiusCount > 0 ? totalRadius / radiusCount : 0;

      // Log to StatsLogger
      logger.logCheckpoint({
        sample: i + 1,
        portCount: actor.getPortCount(),
        // Test metrics (average of both regions)
        testAgency: (evalRegion1.avgAgency + evalRegion2.avgAgency) / 2,
        testError: (evalRegion1.avgError + evalRegion2.avgError) / 2,
        testBindingRate: (evalRegion1.testBindingRate + evalRegion2.testBindingRate) / 2,
        // Training metrics (from actual training data)
        trainAgency,
        trainError,
        trainBindingRate,
        // Calibration metrics
        testCalibrationGap,
        trainCalibrationGap,
        avgConeRadius,
      });

      logger.logPorts(
        actor.getAnalytics().ports.map((p) => ({
          portIdx: p.portIdx,
          sample: i + 1,
          samples: p.samples,
          bindings: Math.round(p.bindingRate * p.samples),
          bindingRate: p.bindingRate,
          avgAgency: p.avgAgency,
          agencyStdDev: p.agencyStdDev,
          avgError: p.avgError,
          errorStdDev: p.errorStdDev,
        }))
      );

      // Log latent space
      latentLogger.logCheckpoint(i + 1, actor);

      // Console output
      console.log(
        `${String(i + 1).padStart(6)} | ` +
        `${evalRegion1.avgError.toFixed(4)}/${evalRegion2.avgError.toFixed(4)} | ` +
        `${(evalRegion1.avgAgency * 100).toFixed(1)}%/${(evalRegion2.avgAgency * 100).toFixed(1)}% | ` +
        `${(evalRegion1.testBindingRate * 100).toFixed(0)}%/${(evalRegion2.testBindingRate * 100).toFixed(0)}% | ` +
        `${(calibrationGapRegion1 * 100).toFixed(1)}%/${(calibrationGapRegion2 * 100).toFixed(1)}% | ` +
        `${actor.getPortCount()}`
      );
    }
  }

  logger.close();
  latentLogger.close();

  await generateVisualization(logger.getRunDir());
  await generateLatentVisualization(logger.getRunDir());

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Final Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  const finalCheckpoint = checkpoints[checkpoints.length - 1]!;

  console.log("Region 1 (x >= 0):");
  console.log(`  Error:    ${finalCheckpoint.errorRegion1.toFixed(4)}`);
  console.log(`  Agency:   ${(finalCheckpoint.agencyRegion1 * 100).toFixed(1)}%`);
  console.log(`  Binding:  ${(finalCheckpoint.bindingRateRegion1 * 100).toFixed(1)}%`);
  console.log(`  Cal.Gap:  ${(finalCheckpoint.calibrationGapRegion1 * 100).toFixed(1)}%`);

  console.log("\nRegion 2 (x < 0):");
  console.log(`  Error:    ${finalCheckpoint.errorRegion2.toFixed(4)}`);
  console.log(`  Agency:   ${(finalCheckpoint.agencyRegion2 * 100).toFixed(1)}%`);
  console.log(`  Binding:  ${(finalCheckpoint.bindingRateRegion2 * 100).toFixed(1)}%`);
  console.log(`  Cal.Gap:  ${(finalCheckpoint.calibrationGapRegion2 * 100).toFixed(1)}%`);

  console.log(`\nFinal port count: ${finalCheckpoint.portCount}`);

  console.log("\n═══════════════════════════════════════════════════════════");

  if (actor.getPortCount() === 1) {
    console.log("⚠️  Single port covers both regions (no proliferation)");
    console.log("   Anisotropic cones stretched across the gap");
  } else {
    console.log(`✓ ${actor.getPortCount()} ports created (proliferation occurred)`);
    console.log("   Clustering detected barbell shape and triggered specialization");
  }

  console.log("═══════════════════════════════════════════════════════════");
  console.log("\n📊 Visualizations:");
  console.log(`  Metrics:      ${logger.getRunDir()}/visualization.html`);
  console.log(`  Latent Space: ${logger.getRunDir()}/latent-visualization.html`);
  console.log("\n✓ Forced proliferation experiment complete!\n");
}

main().catch(console.error);
