/**
 * Homeostasis Experiment
 *
 * Tests whether TAM's refinement loop produces calibrated models:
 * - High agency should correlate with high binding success
 * - Out-of-domain situations should show low agency (epistemic humility)
 *
 * Homeostasis Score = Agency × BindingRate
 * A crystallized model maximizes this across diverse in-domain situations.
 */

import {
  TAM,
  GeometricPortBank,
  evaluateBinding,
  type DomainSpec,
} from "../../src";
import type { Vec } from "../../src/vec";

// ============================================================================
// Test Configuration
// ============================================================================

interface TestResult {
  situation: string;
  agency: number;
  bindingSuccess: boolean;
  homeostasis: number; // agency × (success ? 1 : 0)
}

interface DomainTestConfig<S> {
  name: string;
  spec: DomainSpec<S>;
  inDomainSampler: () => S; // Same as training distribution
  outOfDomainSampler: () => S; // Extrapolation / novel situations
  stateToString: (s: S) => string;
}

// ============================================================================
// Shift Domain: x → x + δ
// ============================================================================

const SHIFT_TRAIN_MIN = -10;
const SHIFT_TRAIN_MAX = 10;

const shiftConfig: DomainTestConfig<number> = {
  name: "shift",
  spec: {
    randomState: () =>
      SHIFT_TRAIN_MIN + Math.random() * (SHIFT_TRAIN_MAX - SHIFT_TRAIN_MIN),
    simulate: (x) => x + (Math.random() - 0.5) * 2, // ±1
    embedder: (x) => [x / 10], // Normalized to training range
    embeddingDim: 1,
  },
  inDomainSampler: () =>
    SHIFT_TRAIN_MIN + Math.random() * (SHIFT_TRAIN_MAX - SHIFT_TRAIN_MIN),
  outOfDomainSampler: () => {
    // Far outside training range
    const sign = Math.random() > 0.5 ? 1 : -1;
    return sign * (50 + Math.random() * 50); // ±[50, 100]
  },
  stateToString: (x) => x.toFixed(2),
};

// ============================================================================
// Scale Domain: x → x * k
// ============================================================================

const SCALE_TRAIN_MIN = 0.5;
const SCALE_TRAIN_MAX = 5;

const scaleConfig: DomainTestConfig<number> = {
  name: "scale",
  spec: {
    randomState: () =>
      SCALE_TRAIN_MIN + Math.random() * (SCALE_TRAIN_MAX - SCALE_TRAIN_MIN),
    simulate: (x) => Math.max(0.1, x * (0.8 + Math.random() * 0.4)), // ±20%
    embedder: (x) => [(x - 2.75) / 2.25], // Normalized
    embeddingDim: 1,
  },
  inDomainSampler: () =>
    SCALE_TRAIN_MIN + Math.random() * (SCALE_TRAIN_MAX - SCALE_TRAIN_MIN),
  outOfDomainSampler: () => {
    // Far outside training range
    return 100 + Math.random() * 100; // [100, 200]
  },
  stateToString: (x) => x.toFixed(2),
};

// ============================================================================
// Test Harness
// ============================================================================

async function testDomain<S>(
  tam: TAM,
  config: DomainTestConfig<S>,
  trainEpochs: number,
  testSamples: number
): Promise<{
  inDomain: TestResult[];
  outOfDomain: TestResult[];
  summary: {
    inDomainAgency: number;
    inDomainBindingRate: number;
    inDomainHomeostasis: number;
    outOfDomainAgency: number;
    outOfDomainBindingRate: number;
    outOfDomainHomeostasis: number;
  };
}> {
  // Train the domain
  const port = await tam.learn(config.name, config.spec, {
    epochs: trainEpochs,
    samplesPerEpoch: 50,
    flushFrequency: 20,
  });

  // Test in-domain
  const inDomain = testSituations(
    port,
    config.spec.simulate,
    config.spec.embedder!,
    config.inDomainSampler,
    config.stateToString,
    testSamples
  );

  // Test out-of-domain
  const outOfDomain = testSituations(
    port,
    config.spec.simulate,
    config.spec.embedder!,
    config.outOfDomainSampler,
    config.stateToString,
    testSamples
  );

  // Compute summary stats
  const summarize = (results: TestResult[]) => ({
    agency: results.reduce((s, r) => s + r.agency, 0) / results.length,
    bindingRate:
      results.filter((r) => r.bindingSuccess).length / results.length,
    homeostasis:
      results.reduce((s, r) => s + r.homeostasis, 0) / results.length,
  });

  const inSummary = summarize(inDomain);
  const outSummary = summarize(outOfDomain);

  return {
    inDomain,
    outOfDomain,
    summary: {
      inDomainAgency: inSummary.agency,
      inDomainBindingRate: inSummary.bindingRate,
      inDomainHomeostasis: inSummary.homeostasis,
      outOfDomainAgency: outSummary.agency,
      outOfDomainBindingRate: outSummary.bindingRate,
      outOfDomainHomeostasis: outSummary.homeostasis,
    },
  };
}

function testSituations<S>(
  portBank: GeometricPortBank<S, {}>,
  simulate: (s: S) => S,
  embedder: (s: S) => Vec,
  sampler: () => S,
  stateToString: (s: S) => string,
  n: number
): TestResult[] {
  const results: TestResult[] = [];

  // Get the default port
  const port = portBank.get("default");

  for (let i = 0; i < n; i++) {
    const before = sampler();
    const after = simulate(before);

    // Create situation object
    const situation = { state: before, context: {} };

    // Get cone and prediction
    const cone = port.getCone(situation);
    const predictions = port.predict(situation);
    const agency = predictions.length > 0 ? predictions[0]!.agency : 0;

    // Compute actual delta
    const beforeEmb = embedder(before);
    const afterEmb = embedder(after);
    const actualDelta = afterEmb.map((a, j) => a - beforeEmb[j]!);

    // Use canonical binding predicate (ellipsoid distance, same as fibration.ts)
    const bindingOutcome = evaluateBinding(actualDelta, cone);
    const bindingSuccess = bindingOutcome.success;

    results.push({
      situation: stateToString(before),
      agency,
      bindingSuccess,
      homeostasis: agency * (bindingSuccess ? 1 : 0),
    });
  }

  return results;
}

// ============================================================================
// Adaptation Test: Does the model learn from out-of-domain failures?
// ============================================================================

async function testAdaptation<S>(
  tam: TAM,
  config: DomainTestConfig<S>,
  portBank: GeometricPortBank<S, {}>,
  adaptationSamples: number
): Promise<{
  agencyBefore: number;
  agencyAfter: number;
  bindingRateBefore: number;
  bindingRateAfter: number;
}> {
  const port = portBank.get("default");

  // Measure initial out-of-domain agency and binding rate
  const beforeResults = testSituations(
    portBank,
    config.spec.simulate,
    config.spec.embedder!,
    config.outOfDomainSampler,
    config.stateToString,
    20
  );
  const agencyBefore =
    beforeResults.reduce((s, r) => s + r.agency, 0) / beforeResults.length;
  const bindingRateBefore =
    beforeResults.filter((r) => r.bindingSuccess).length / beforeResults.length;

  // Expose model to out-of-domain samples (online learning)
  for (let i = 0; i < adaptationSamples; i++) {
    const before = config.outOfDomainSampler();
    const after = config.spec.simulate(before);

    // Observe the transition - this should trigger widening on failure
    portBank.observe({
      before: { state: before, context: {} },
      after: { state: after, context: {} },
      action: "default",
    });

    // Flush periodically to train networks
    if ((i + 1) % 10 === 0) {
      portBank.flush();
    }
  }
  portBank.flush();

  // Measure agency and binding rate after adaptation
  const afterResults = testSituations(
    portBank,
    config.spec.simulate,
    config.spec.embedder!,
    config.outOfDomainSampler,
    config.stateToString,
    20
  );
  const agencyAfter =
    afterResults.reduce((s, r) => s + r.agency, 0) / afterResults.length;
  const bindingRateAfter =
    afterResults.filter((r) => r.bindingSuccess).length / afterResults.length;

  return { agencyBefore, agencyAfter, bindingRateBefore, bindingRateAfter };
}

// ============================================================================
// Main Experiment
// ============================================================================

async function main() {
  console.log("=".repeat(70));
  console.log("HOMEOSTASIS EXPERIMENT");
  console.log("=".repeat(70));
  console.log();
  console.log("Part 1: In-domain vs Out-of-domain performance");
  console.log(
    "Part 2: Online adaptation - does the model learn from failures?"
  );
  console.log();

  const TRAIN_EPOCHS = 300;
  const TEST_SAMPLES = 100;
  const ADAPTATION_SAMPLES = 100;

  const tam = new TAM();

  // Test each domain
  for (const config of [shiftConfig, scaleConfig]) {
    console.log("-".repeat(70));
    console.log(`Domain: ${config.name.toUpperCase()}`);
    console.log("-".repeat(70));

    const results = await testDomain(tam, config, TRAIN_EPOCHS, TEST_SAMPLES);
    const s = results.summary;

    console.log();
    console.log("PART 1: Initial Performance");
    console.log("                    Agency    BindingRate   Homeostasis");
    console.log(
      `  In-Domain:        ${(s.inDomainAgency * 100)
        .toFixed(1)
        .padStart(5)}%` +
        `      ${(s.inDomainBindingRate * 100).toFixed(1).padStart(5)}%` +
        `        ${(s.inDomainHomeostasis * 100).toFixed(1).padStart(5)}%`
    );
    console.log(
      `  Out-of-Domain:    ${(s.outOfDomainAgency * 100)
        .toFixed(1)
        .padStart(5)}%` +
        `      ${(s.outOfDomainBindingRate * 100).toFixed(1).padStart(5)}%` +
        `        ${(s.outOfDomainHomeostasis * 100).toFixed(1).padStart(5)}%`
    );
    console.log();

    // Part 2: Test adaptation
    console.log("PART 2: Online Adaptation (learning from OOD failures)");

    // Get the port bank that was trained
    const portBank = await tam.learn(`${config.name}-adapt`, config.spec, {
      epochs: TRAIN_EPOCHS,
      samplesPerEpoch: 50,
      flushFrequency: 20,
    });

    const adaptation = await testAdaptation(
      tam,
      config,
      portBank,
      ADAPTATION_SAMPLES
    );

    console.log(
      `  Before adaptation: Agency = ${(adaptation.agencyBefore * 100).toFixed(
        1
      )}%, ` + `Binding = ${(adaptation.bindingRateBefore * 100).toFixed(1)}%`
    );
    console.log(
      `  After adaptation:  Agency = ${(adaptation.agencyAfter * 100).toFixed(
        1
      )}%, ` + `Binding = ${(adaptation.bindingRateAfter * 100).toFixed(1)}%`
    );

    const agencyDelta = adaptation.agencyAfter - adaptation.agencyBefore;
    const bindingDelta =
      adaptation.bindingRateAfter - adaptation.bindingRateBefore;

    console.log();
    if (bindingDelta > 0.1) {
      console.log("  ✓ Model adapted: Binding rate improved after exposure");
    }
    if (agencyDelta < -0.1 && adaptation.bindingRateAfter < 0.5) {
      console.log(
        "  ✓ Model widened cones: Agency decreased (epistemic humility)"
      );
    } else if (agencyDelta > 0 && adaptation.bindingRateAfter > 0.5) {
      console.log(
        "  ✓ Model crystallized: Agency increased with better binding"
      );
    } else if (Math.abs(agencyDelta) < 0.1) {
      console.log("  ~ Agency stable during adaptation");
    }

    const homeostasisAfter =
      adaptation.agencyAfter * adaptation.bindingRateAfter;
    console.log(
      `  Homeostasis after adaptation: ${(homeostasisAfter * 100).toFixed(1)}%`
    );
    console.log();
  }

  // Summary
  console.log("=".repeat(70));
  console.log("INTERPRETATION");
  console.log("=".repeat(70));
  console.log();
  console.log("TAM is an online learning framework. What matters:");
  console.log(
    "  1. Initial OOD confidence is fine (hasn't been proven wrong yet)"
  );
  console.log("  2. After failures, model should adapt:");
  console.log(
    "     - Widen cones (lower agency) if dynamics are unpredictable"
  );
  console.log("     - Narrow cones (higher agency) once dynamics are learned");
  console.log("  3. Homeostasis = high agency + high binding in steady state");
  console.log();

  tam.dispose();
}

main().catch(console.error);
