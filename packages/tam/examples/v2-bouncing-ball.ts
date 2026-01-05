/**
 * V2 Bouncing Ball - Steep Nonlinearity Test
 *
 * Tests how the unified geometric system handles sharp but continuous transitions:
 * - Smooth region: Free fall under gravity (linear, predictable)
 * - Steep region: Ground potential well (nonlinear, strong forces)
 *
 * Key questions:
 * 1. Does variance learn to be higher in steep nonlinear regions?
 * 2. Does the system maintain calibration across different dynamic regimes?
 * 3. Can a single port handle both, or does proliferation create specialized ports?
 *
 * Implementation notes:
 * - Uses smooth potential well: F = -k * y² when y < 0 (continuous but steep)
 * - No hard discontinuities - dynamics are differentiable everywhere
 * - Damping near ground creates energy loss without velocity jumps
 *
 * Expected behavior:
 * - Higher variance (wider cones) near y=0 (steep forces)
 * - Lower variance (narrow cones) in free fall region
 * - Binding rate reflects complexity of local dynamics
 */

import {
  Actor,
  StatsLogger,
  LatentLogger,
  evaluate,
  generateVisualization,
  generateLatentVisualization,
} from "../src/v2";

// 1D ball with smooth ground potential (continuous dynamics)
const g = 9.8;  // gravity
const dt = 0.05;
const k_ground = 500;  // ground stiffness (steep potential well)
const damping = 0.8;   // velocity damping near ground

type State = { y: number; vy: number };

function step(state: State): State {
  // Ground force: smooth potential well instead of hard collision
  // F = -k * max(0, -y)^2 creates steep but continuous repulsion
  const groundForce = state.y < 0 ? -k_ground * state.y * state.y : 0;

  // Apply gravity + ground force
  const ay = -g + groundForce;
  let newVy = state.vy + ay * dt;

  // Damping when near ground (models energy loss)
  if (state.y < 0.1) {
    newVy *= damping;
  }

  const newY = state.y + newVy * dt;

  return { y: newY, vy: newVy };
}

// In-distribution: [0.1, 5] x [-10, 10]
function randomStateInDist(): State {
  return {
    y: 0.1 + Math.random() * 4.9,
    vy: (Math.random() - 0.5) * 20,
  };
}

// Out-of-distribution: higher altitude
function randomStateOutDist(): State {
  return {
    y: 5 + Math.random() * 5,
    vy: (Math.random() - 0.5) * 20,
  };
}

const embedState = (s: State) => [s.y, s.vy];

interface Checkpoint {
  sample: number;
  errorInDist: number;
  errorOutDist: number;
  agencyInDist: number;
  agencyOutDist: number;
  bindingRateInDist: number;
  bindingRateOutDist: number;
  calibrationGap: number;
  portCount: number;
  trainBindingRate: number;
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  V2 Bouncing Ball - Steep Nonlinearity");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Testing uncertainty learning across dynamic regimes:");
  console.log("  - Smooth: Free fall (linear dynamics)");
  console.log("  - Steep: Ground potential (nonlinear forces)\n");

  const samples = 5_000;
  const checkpointEvery = 100;

  // Set up logging
  const logger = await StatsLogger.create({
    experiment: "V2 Bouncing Ball",
    config: { samples, checkpointEvery },
  });

  // Set up latent space logger
  const latentLogger = LatentLogger.create(
    logger.getRunDir(),
    randomStateInDist,
    embedState,
    { numStates: 10 }
  );

  const actor = new Actor<State>(embedState, {
    proliferation: { enabled: true },
  });

  // Generate fixed test sets
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

      // Evaluate on FIXED test sets
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

      const calibrationGap = Math.abs(
        evalInDist.avgAgency - evalInDist.testBindingRate
      );

      // Get training metrics
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

      const testCalibrationGap = Math.abs(evalInDist.avgAgency - evalInDist.testBindingRate);
      const trainCalibrationGap = Math.abs(trainAgency - trainBindingRate);

      // Compute average cone radius
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

  actor.dispose();

  // Analysis
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Nonlinearity Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  const initial = checkpoints[0]!;
  const final = checkpoints[checkpoints.length - 1]!;

  console.log("Steep Dynamics Handling:");
  console.log(`  Final binding rate: ${(final.bindingRateInDist * 100).toFixed(1)}%`);
  console.log(`  Final agency: ${(final.agencyInDist * 100).toFixed(1)}%`);
  console.log(`  Final error: ${final.errorInDist.toFixed(4)}`);
  console.log(`  Ports created: ${final.portCount}`);

  if (final.portCount > 1) {
    console.log(`\n  ✓ System created ${final.portCount} ports`);
    console.log(`    Interpretation: Specialized ports for different dynamic regimes`);
  } else {
    console.log(`\n  • System used 1 port`);
    console.log(`    Interpretation: Single port handles both regimes with adaptive variance`);
  }

  console.log(`\nPhase Summary:`);
  console.log(`  Initial → Final:`);
  console.log(`    Error (in-dist):     ${initial.errorInDist.toFixed(4)} → ${final.errorInDist.toFixed(4)}`);
  console.log(`    Agency (in-dist):    ${(initial.agencyInDist * 100).toFixed(1)}% → ${(final.agencyInDist * 100).toFixed(1)}%`);
  console.log(`    Binding (in-dist):   ${(initial.bindingRateInDist * 100).toFixed(1)}% → ${(final.bindingRateInDist * 100).toFixed(1)}%`);
  console.log(`    Calibration Gap:     ${initial.calibrationGap.toFixed(3)} → ${final.calibrationGap.toFixed(3)}`);

  console.log(`\nVisualizations:`);
  console.log(`  Metrics:      ${logger.getRunDir()}/visualization.html`);
  console.log(`  Latent Space: ${logger.getRunDir()}/latent-visualization.html`);
  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
