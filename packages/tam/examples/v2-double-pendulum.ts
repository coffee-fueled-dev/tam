/**
 * V2 Double Pendulum - High-Dimensional Chaotic Test
 *
 * Tests how the unified geometric system handles:
 * - Higher dimensionality: 4D state space (θ1, θ2, ω1, ω2)
 * - Chaotic dynamics: Sensitivity to initial conditions
 * - Complex coupling: Both pendulums affect each other
 *
 * Key questions:
 * 1. Does variance correctly reflect chaotic unpredictability?
 * 2. Can the system maintain calibration in higher dimensions?
 * 3. Does proliferation create specialized ports for different motion patterns?
 * 4. How does binding rate reflect intrinsic chaos vs learnable structure?
 *
 * Expected behavior:
 * - Higher variance (wider cones) in chaotic regions
 * - Lower binding rate than simple systems (chaos is hard to predict)
 * - Multiple ports may emerge for different motion regimes (pendulum-like vs chaotic)
 */

import {
  Actor,
  StatsLogger,
  LatentLogger,
  evaluate,
  generateVisualization,
  generateLatentVisualization,
} from "../src/v2";

// Double pendulum parameters
const m1 = 1.0;  // mass of first bob
const m2 = 1.0;  // mass of second bob
const L1 = 1.0;  // length of first rod
const L2 = 1.0;  // length of second rod
const g = 9.81;  // gravity
const dt = 0.02; // small timestep for stability

type State = {
  theta1: number;  // angle of first pendulum
  theta2: number;  // angle of second pendulum
  omega1: number;  // angular velocity of first
  omega2: number;  // angular velocity of second
};

/**
 * Double pendulum dynamics (Lagrangian mechanics)
 * See: https://www.myphysicslab.com/pendulum/double-pendulum-en.html
 */
function derivatives(s: State): { dtheta1: number; dtheta2: number; domega1: number; domega2: number } {
  const { theta1, theta2, omega1, omega2 } = s;
  const dtheta = theta2 - theta1;

  // Common terms
  const denominator = 2 * m1 + m2 - m2 * Math.cos(2 * dtheta);

  // Angular accelerations (from Lagrangian)
  const domega1 = (
    -g * (2 * m1 + m2) * Math.sin(theta1) -
    m2 * g * Math.sin(theta1 - 2 * theta2) -
    2 * Math.sin(dtheta) * m2 * (omega2 * omega2 * L2 + omega1 * omega1 * L1 * Math.cos(dtheta))
  ) / (L1 * denominator);

  const domega2 = (
    2 * Math.sin(dtheta) * (
      omega1 * omega1 * L1 * (m1 + m2) +
      g * (m1 + m2) * Math.cos(theta1) +
      omega2 * omega2 * L2 * m2 * Math.cos(dtheta)
    )
  ) / (L2 * denominator);

  return {
    dtheta1: omega1,
    dtheta2: omega2,
    domega1,
    domega2,
  };
}

function step(state: State): State {
  // RK4 integration for better stability
  const k1 = derivatives(state);

  const k2 = derivatives({
    theta1: state.theta1 + 0.5 * dt * k1.dtheta1,
    theta2: state.theta2 + 0.5 * dt * k1.dtheta2,
    omega1: state.omega1 + 0.5 * dt * k1.domega1,
    omega2: state.omega2 + 0.5 * dt * k1.domega2,
  });

  const k3 = derivatives({
    theta1: state.theta1 + 0.5 * dt * k2.dtheta1,
    theta2: state.theta2 + 0.5 * dt * k2.dtheta2,
    omega1: state.omega1 + 0.5 * dt * k2.domega1,
    omega2: state.omega2 + 0.5 * dt * k2.domega2,
  });

  const k4 = derivatives({
    theta1: state.theta1 + dt * k3.dtheta1,
    theta2: state.theta2 + dt * k3.dtheta2,
    omega1: state.omega1 + dt * k3.domega1,
    omega2: state.omega2 + dt * k3.domega2,
  });

  return {
    theta1: state.theta1 + (dt / 6) * (k1.dtheta1 + 2 * k2.dtheta1 + 2 * k3.dtheta1 + k4.dtheta1),
    theta2: state.theta2 + (dt / 6) * (k1.dtheta2 + 2 * k2.dtheta2 + 2 * k3.dtheta2 + k4.dtheta2),
    omega1: state.omega1 + (dt / 6) * (k1.domega1 + 2 * k2.domega1 + 2 * k3.domega1 + k4.domega1),
    omega2: state.omega2 + (dt / 6) * (k1.domega2 + 2 * k2.domega2 + 2 * k3.domega2 + k4.domega2),
  };
}

// In-distribution: moderate angles and velocities
function randomStateInDist(): State {
  return {
    theta1: (Math.random() - 0.5) * Math.PI,     // [-π/2, π/2]
    theta2: (Math.random() - 0.5) * Math.PI,     // [-π/2, π/2]
    omega1: (Math.random() - 0.5) * 4,           // [-2, 2]
    omega2: (Math.random() - 0.5) * 4,           // [-2, 2]
  };
}

// Out-of-distribution: higher energy (larger angles)
function randomStateOutDist(): State {
  return {
    theta1: (Math.random() - 0.5) * 2 * Math.PI,  // [-π, π]
    theta2: (Math.random() - 0.5) * 2 * Math.PI,  // [-π, π]
    omega1: (Math.random() - 0.5) * 8,            // [-4, 4]
    omega2: (Math.random() - 0.5) * 8,            // [-4, 4]
  };
}

const embedState = (s: State) => [s.theta1, s.theta2, s.omega1, s.omega2];

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
  console.log("  V2 Double Pendulum - High-Dimensional Chaos");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Testing 4D chaotic dynamics:");
  console.log("  - State space: (θ1, θ2, ω1, ω2)");
  console.log("  - Dynamics: Coupled, nonlinear, sensitive to ICs");
  console.log("  - Challenge: Learn structure in intrinsic chaos\n");

  const samples = 10_000;
  const checkpointEvery = 150;

  // Set up logging
  const logger = await StatsLogger.create({
    experiment: "V2 Double Pendulum",
    config: { samples, checkpointEvery },
  });

  // Set up latent space logger (will use PCA for visualization)
  const latentLogger = LatentLogger.create(
    logger.getRunDir(),
    randomStateInDist,
    embedState,
    { numStates: 10 }
  );

  const actor = new Actor<State>(embedState, {
    proliferation: { enabled: true },
    causal: {
      minVariance: 2.0,  // Lower floor for chaotic dynamics (default 5.0)
    },
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
  console.log("  High-Dimensional Chaos Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  const initial = checkpoints[0]!;
  const final = checkpoints[checkpoints.length - 1]!;

  console.log("Chaos Handling:");
  console.log(`  Final binding rate: ${(final.bindingRateInDist * 100).toFixed(1)}%`);
  console.log(`  Final agency: ${(final.agencyInDist * 100).toFixed(1)}%`);
  console.log(`  Final error: ${final.errorInDist.toFixed(4)}`);
  console.log(`  Ports created: ${final.portCount}`);

  if (final.bindingRateInDist < 0.5) {
    console.log(`\n  ✓ Low binding rate reflects intrinsic chaos`);
    console.log(`    Interpretation: System correctly recognizes unpredictability`);
  }

  if (final.portCount > 1) {
    console.log(`\n  ✓ System created ${final.portCount} ports`);
    console.log(`    Interpretation: Specialized ports for different motion regimes`);
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
  console.log(`    (Note: 4D embeddings projected to 2D via PCA)`);
  console.log("\n✓ Experiment complete!");
}

main().catch(console.error);
