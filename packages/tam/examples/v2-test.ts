/**
 * Test TAM v2 with simple 1D damped spring
 *
 * Verifies:
 * - Basic training loop works
 * - Port proliferation happens
 * - Networks learn dynamics
 * - Agency increases with binding
 */

import { Actor, StatsLogger, LatentLogger, evaluate, generateVisualization, generateLatentVisualization } from "../src/v2";

// 1D damped spring: x'' = -kx - bv
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

function randomState(): State {
  return {
    x: (Math.random() - 0.5) * 4,
    v: (Math.random() - 0.5) * 4,
  };
}

// State embedder
const embedState = (s: State): number[] => [s.x, s.v];

async function main() {
  console.log("═══════════════════════════════════════════════════════");
  console.log("  TAM v2 Test: 1D Damped Spring");
  console.log("═══════════════════════════════════════════════════════\n");

  const samples = 2000;
  const checkpointEvery = 50;

  // Set up stats logger
  const logger = await StatsLogger.create({
    experiment: "1D Damped Spring",
    config: {
      k,
      b,
      dt,
      samples,
      checkpointEvery,
    },
  });

  // Set up latent space logger
  const latentLogger = LatentLogger.create(
    logger.getRunDir(),
    randomState,
    embedState,
    { numStates: 10 }
  );

  const actor = new Actor<State>(embedState, {
    proliferation: {
      enabled: true,
    },
  });

  console.log("Training...\n");
  console.log("Sample | Ports | Test (Agency / Error) | Train (Agency / Error / BindRate)");
  console.log("-------|-------|------------------------|----------------------------------");

  for (let i = 0; i < samples; i++) {
    const before = randomState();
    const after = step(before);

    await actor.observe({
      before: { state: before },
      after: { state: after },
    });

    if ((i + 1) % checkpointEvery === 0) {
      await actor.flush();

      // Evaluate on test samples
      const { avgAgency, avgError, testBindingRate } = evaluate(
        actor,
        () => {
          const before = randomState();
          const after = step(before);
          return { before, after };
        },
        embedState
      );

      // Get detailed analytics
      const analytics = actor.getAnalytics();

      // Compute aggregate training metrics (weighted by samples per port)
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

      // Log checkpoint metrics (both training and test)
      logger.logCheckpoint({
        sample: i + 1,
        portCount: analytics.portCount,
        // Test metrics (held-out samples)
        testAgency: avgAgency,
        testError: avgError,
        testBindingRate,
        // Training metrics (aggregate across all ports)
        trainAgency,
        trainError,
        trainBindingRate,
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
        `${(i + 1).toString().padStart(6)} | ` +
          `${actor.getPortCount().toString().padStart(5)} | ` +
          `Test: ${(avgAgency * 100).toFixed(1).padStart(4)}% / ${avgError.toFixed(4)} | ` +
          `Train: ${(trainAgency * 100).toFixed(1).padStart(4)}% / ${trainError.toFixed(4)} / BR ${(trainBindingRate * 100).toFixed(1)}%`
      );
    }
  }

  await actor.flush();

  // Close loggers
  logger.close();
  latentLogger.close();

  console.log("\n✓ Test complete!");
  console.log(`Final port count: ${actor.getPortCount()}`);
  console.log(`Analytics written to: ${logger.getRunDir()}/`);

  // Generate visualizations
  await generateVisualization(logger.getRunDir());
  await generateLatentVisualization(logger.getRunDir());

  actor.dispose();
}

main().catch(console.error);
