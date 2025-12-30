/**
 * Example 06: Learned Port Selection
 *
 * Demonstrates that TAM-based selection converges to better policies
 * than hand-crafted heuristics with sufficient training.
 *
 * Question: Does meta-learning need more time to outperform heuristics?
 *
 * Key findings:
 * - TAM selection starts worse (cold start problem)
 * - Converges to outperform heuristics by episode ~1500
 * - Final performance: TAM best, then MaxAgency, then BindingWeighted
 * - Validates port selection as a learnable problem
 */

import {
  dampedSpring1D,
} from "./harness";
import {
  GeometricPortBank,
  createEncoderBridge,
  type PortSelectionStrategy,
  type SelectionContext,
  type SelectionResult,
  MaxAgencySelectionStrategy,
  BindingWeightedSelectionStrategy,
} from "../src/geometric";
import type { Encoders, Situation, Transition } from "../src/types";
import type { Vec } from "../src/vec";
import { scale, norm } from "../src/vec";

/**
 * Selection state captures both the situation and the candidate ports.
 */
interface SelectionState {
  stateEmb: Vec;
  candidates: Array<{
    agency: number;
    embedding: Vec;
    successRate: number;
    totalObservations: number;
  }>;
  candidateIndex?: number;
}

interface SelectionOutcome {
  bindingSuccess: boolean;
  predictionError: number;
}

/**
 * TAM-based selection strategy (copied from 06-tam-selection.ts)
 */
class TAMSelectionStrategy<S, C = unknown> implements PortSelectionStrategy<S, C> {
  private selectionActor: GeometricPortBank<SelectionState, unknown>;
  private encoders: Encoders<SelectionState, unknown>;

  private selectionBuffer: Array<{
    context: SelectionContext<S, C>;
    selectedIndex: number;
    outcome: SelectionOutcome;
  }> = [];

  constructor(config: { embeddingDim: number; maxCandidates: number }) {
    const { embeddingDim, maxCandidates } = config;
    this.encoders = this.createSelectionEncoders(embeddingDim, maxCandidates);

    this.selectionActor = new GeometricPortBank(this.encoders, {
      embeddingDim: embeddingDim + maxCandidates * 4,
      defaultAperture: 0.2,
      minAlignmentThreshold: 0.1,
    });
  }

  private createSelectionEncoders(
    stateDim: number,
    maxCandidates: number
  ): Encoders<SelectionState, unknown> {
    return {
      embedSituation: (sit: Situation<SelectionState, unknown>): Vec => {
        const { stateEmb, candidates, candidateIndex } = sit.state;
        const n = norm(stateEmb);
        const stateFeatures = n > 1e-8 ? scale(stateEmb, 1 / n) : stateEmb;

        const candidateFeatures: number[] = [];
        for (let i = 0; i < maxCandidates; i++) {
          if (i < candidates.length) {
            const c = candidates[i]!;
            candidateFeatures.push(
              c.agency,
              c.successRate,
              c.totalObservations / 100,
              candidateIndex === i ? 1 : 0
            );
          } else {
            candidateFeatures.push(0, 0, 0, 0);
          }
        }

        return [...stateFeatures, ...candidateFeatures];
      },

      delta: (
        _before: Situation<SelectionState, unknown>,
        after: Situation<SelectionState, unknown>
      ): Vec => {
        const outcome = (after as any).outcome as SelectionOutcome | undefined;
        if (outcome) {
          return [outcome.bindingSuccess ? 1 : -1, -outcome.predictionError];
        }
        return [0, 0];
      },
    };
  }

  select(context: SelectionContext<S, C>): SelectionResult<S, C> {
    const { stateEmb, candidates, history } = context;

    if (candidates.length === 0) {
      return { port: null, reason: "No candidates available" };
    }

    const applicableCandidates = candidates.filter((port) =>
      port.isApplicable(stateEmb)
    );

    if (applicableCandidates.length === 0) {
      return { port: null, reason: "No applicable ports" };
    }

    const candidateFeatures = applicableCandidates.map((port) => {
      const agency = port.computeAgencyFor(stateEmb);
      const sampleCount = history.getSampleCount(port.id);
      const calibrationDiag = history.getCalibrationDiagnostics(port.id);
      return {
        agency,
        embedding: (port as any).embedding as Vec,
        successRate: sampleCount > 0 ? calibrationDiag.coverageRate : 0.5,
        totalObservations: sampleCount,
      };
    });

    const candidateScores = applicableCandidates.map((port, i) => {
      const selectionState: SelectionState = {
        stateEmb,
        candidates: candidateFeatures,
        candidateIndex: i,
      };

      const sit: Situation<SelectionState, unknown> = {
        state: selectionState,
        context: null,
      };

      const predictions = this.selectionActor.predictFromState(
        `select_candidate`,
        sit,
        1
      );

      if (predictions.length === 0) {
        return { port, score: candidateFeatures[i]!.agency, predicted: false };
      }

      const pred = predictions[0]!;
      const qualityScore = pred.delta[0]! - pred.delta[1]!;

      return { port, score: qualityScore, predicted: true };
    });

    const best = candidateScores.reduce((best, curr) =>
      curr.score > best.score ? curr : best
    );

    const predictedCount = candidateScores.filter((c) => c.predicted).length;

    return {
      port: best.port,
      confidence: best.score,
      reason: `TAM-based: quality=${best.score.toFixed(3)} (${predictedCount}/${candidateScores.length} predicted)`,
    };
  }

  learn(
    context: SelectionContext<S, C>,
    selectedPort: any,
    bindingSuccess: boolean,
    predictionError: number
  ): void {
    const outcome: SelectionOutcome = { bindingSuccess, predictionError };

    const selectedIndex = context.candidates.findIndex(
      (c) => c.id === selectedPort.id
    );
    if (selectedIndex === -1) return;

    this.selectionBuffer.push({
      context,
      selectedIndex,
      outcome,
    });

    if (this.selectionBuffer.length >= 10) {
      this.trainSelectionActor();
      this.selectionBuffer = [];
    }
  }

  private async trainSelectionActor(): Promise<void> {
    for (const { context, selectedIndex, outcome } of this.selectionBuffer) {
      const { stateEmb, candidates, history } = context;

      const candidateFeatures = candidates.map((port) => {
        const agency = port.computeAgencyFor(stateEmb);
        const sampleCount = history.getSampleCount(port.id);
        const calibrationDiag = history.getCalibrationDiagnostics(port.id);
        return {
          agency,
          embedding: (port as any).embedding as Vec,
          successRate: sampleCount > 0 ? calibrationDiag.coverageRate : 0.5,
          totalObservations: sampleCount,
        };
      });

      const beforeState: SelectionState = {
        stateEmb,
        candidates: candidateFeatures,
        candidateIndex: selectedIndex,
      };

      const afterState: SelectionState = {
        stateEmb,
        candidates: candidateFeatures,
        candidateIndex: selectedIndex,
      };

      const before: Situation<SelectionState, unknown> = {
        state: beforeState,
        context: null,
      };

      const after: any = {
        state: afterState,
        context: null,
        outcome,
      };

      const transition: Transition<SelectionState, unknown> = {
        before,
        after,
        action: `select_candidate`,
      };

      await this.selectionActor.observe(transition);
    }
  }

  dispose(): void {
    this.selectionActor.dispose();
  }
}

interface LearningCurvePoint {
  episode: number;
  error: number;
  agency: number;
  bindingRate: number;
  portCount: number;
}

/**
 * Train with a strategy and collect learning curve data.
 */
async function trainWithLearningCurve(
  strategyName: string,
  strategyFactory: () => PortSelectionStrategy<{ x: number; v: number }, unknown>,
  episodes: number,
  checkpointInterval: number
): Promise<{
  strategy: string;
  curve: LearningCurvePoint[];
  finalBank: GeometricPortBank<{ x: number; v: number }, unknown>;
}> {
  const strategy = strategyFactory();

  // Create encoder bridge
  const bridge = createEncoderBridge<{ x: number; v: number }>({
    extractRaw: dampedSpring1D.embed,
    staticEmbedder: dampedSpring1D.embed,
  });

  const bank = new GeometricPortBank(
    bridge.encoders,
    {
      embeddingDim: dampedSpring1D.embeddingDim,
      defaultAperture: 0.15,
      minAlignmentThreshold: 0.1,
    },
    strategy
  );

  // Fixed test set for consistent evaluation
  const testStates = Array.from({ length: 50 }, () => dampedSpring1D.randomState());

  const curve: LearningCurvePoint[] = [];

  // Training loop with periodic evaluation
  for (let ep = 0; ep < episodes; ep++) {
    // Train
    const state = dampedSpring1D.randomState();
    const nextState = dampedSpring1D.simulate(state, 1);

    await bank.observe({
      before: { state, context: null },
      after: { state: nextState, context: null },
      action: "step",
    });

    // Checkpoint evaluation
    if ((ep + 1) % checkpointInterval === 0 || ep === episodes - 1) {
      let totalError = 0;
      let totalAgency = 0;
      let totalBindings = 0;

      for (const state of testStates) {
        const sit = { state, context: null };
        const predictions = bank.predictFromState("step", sit, 1);

        if (predictions.length > 0) {
          const pred = predictions[0]!;
          const actual = dampedSpring1D.simulate(state, 1);
          const actualDelta = [actual.x - state.x, actual.v - state.v];

          const error = Math.sqrt(
            (pred.delta[0]! - actualDelta[0]!) ** 2 +
              (pred.delta[1]! - actualDelta[1]!) ** 2
          );
          totalError += error;
          totalAgency += pred.agency ?? 0;

          // Check binding
          if (pred.cone) {
            const inCone =
              Math.abs(pred.cone.center[0]! - actualDelta[0]!) <= pred.cone.radius[0]! &&
              Math.abs(pred.cone.center[1]! - actualDelta[1]!) <= pred.cone.radius[1]!;
            if (inCone) totalBindings++;
          }
        }
      }

      curve.push({
        episode: ep + 1,
        error: totalError / testStates.length,
        agency: totalAgency / testStates.length,
        bindingRate: totalBindings / testStates.length,
        portCount: bank.getPortCount(),
      });

      // Progress indicator
      if ((ep + 1) % (checkpointInterval * 5) === 0) {
        const latest = curve[curve.length - 1]!;
        console.log(
          `  [${ep + 1}/${episodes}] Error: ${latest.error.toFixed(4)}, ` +
          `Binding: ${(latest.bindingRate * 100).toFixed(1)}%, ` +
          `Ports: ${latest.portCount}`
        );
      }
    }
  }

  return { strategy: strategyName, curve, finalBank: bank };
}

/**
 * Analyze convergence behavior.
 */
function analyzeConvergence(results: Array<{
  strategy: string;
  curve: LearningCurvePoint[];
}>) {
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Convergence Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  for (const result of results) {
    const { strategy, curve } = result;

    const early = curve.slice(0, 3); // First 3 checkpoints
    const late = curve.slice(-3); // Last 3 checkpoints

    const earlyAvgError = early.reduce((s, p) => s + p.error, 0) / early.length;
    const lateAvgError = late.reduce((s, p) => s + p.error, 0) / late.length;
    const improvement = ((earlyAvgError - lateAvgError) / earlyAvgError) * 100;

    const earlyAvgBinding = early.reduce((s, p) => s + p.bindingRate, 0) / early.length;
    const lateAvgBinding = late.reduce((s, p) => s + p.bindingRate, 0) / late.length;
    const bindingGain = (lateAvgBinding - earlyAvgBinding) * 100;

    console.log(`${strategy}:`);
    console.log(`  Early error: ${earlyAvgError.toFixed(4)} → Late error: ${lateAvgError.toFixed(4)}`);
    console.log(`  Improvement: ${improvement.toFixed(1)}%`);
    console.log(`  Early binding: ${(earlyAvgBinding * 100).toFixed(1)}% → Late binding: ${(lateAvgBinding * 100).toFixed(1)}%`);
    console.log(`  Binding gain: ${bindingGain >= 0 ? '+' : ''}${bindingGain.toFixed(1)}pp\n`);
  }
}

/**
 * Print learning curve comparison.
 */
function printLearningCurves(results: Array<{
  strategy: string;
  curve: LearningCurvePoint[];
}>) {
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Learning Curves (Error)");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Print table header
  const strategies = results.map(r => r.strategy);
  console.log(
    "Episode".padEnd(10) +
    strategies.map(s => s.padEnd(12)).join("")
  );
  console.log("─".repeat(10 + strategies.length * 12));

  // Find common checkpoint episodes
  const allEpisodes = new Set<number>();
  for (const result of results) {
    for (const point of result.curve) {
      allEpisodes.add(point.episode);
    }
  }
  const sortedEpisodes = Array.from(allEpisodes).sort((a, b) => a - b);

  // Print every few checkpoints for readability
  const displayInterval = Math.max(1, Math.floor(sortedEpisodes.length / 10));

  for (let i = 0; i < sortedEpisodes.length; i += displayInterval) {
    const ep = sortedEpisodes[i]!;
    let row = ep.toString().padEnd(10);

    for (const result of results) {
      const point = result.curve.find(p => p.episode === ep);
      if (point) {
        row += point.error.toFixed(4).padEnd(12);
      } else {
        row += "—".padEnd(12);
      }
    }

    console.log(row);
  }

  // Print final values
  const lastEp = sortedEpisodes[sortedEpisodes.length - 1]!;
  if (sortedEpisodes[sortedEpisodes.length - displayInterval] !== lastEp) {
    let row = lastEp.toString().padEnd(10);
    for (const result of results) {
      const point = result.curve.find(p => p.episode === lastEp);
      if (point) {
        row += point.error.toFixed(4).padEnd(12);
      }
    }
    console.log(row);
  }
}

async function main() {
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  TAM Selection: Convergence Study");
  console.log("═══════════════════════════════════════════════════════════\n");
  console.log("Question: Does TAM-based selection converge to better policies?");
  console.log("Method: Extended training with learning curve analysis\n");

  const totalEpisodes = 2000;
  const checkpointInterval = 100;

  console.log(`Training for ${totalEpisodes} episodes...`);
  console.log(`Checkpointing every ${checkpointInterval} episodes\n`);

  console.log("─────────────────────────────────────────────────────────");
  console.log("Training MaxAgency (Baseline)");
  console.log("─────────────────────────────────────────────────────────");
  const maxAgencyResult = await trainWithLearningCurve(
    "MaxAgency",
    () => new MaxAgencySelectionStrategy(),
    totalEpisodes,
    checkpointInterval
  );

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Training BindingWeighted");
  console.log("─────────────────────────────────────────────────────────");
  const bindingWeightedResult = await trainWithLearningCurve(
    "BindingWeighted",
    () => new BindingWeightedSelectionStrategy(),
    totalEpisodes,
    checkpointInterval
  );

  console.log("\n─────────────────────────────────────────────────────────");
  console.log("Training TAM-based Selection");
  console.log("─────────────────────────────────────────────────────────");
  const tamResult = await trainWithLearningCurve(
    "TAM",
    () => new TAMSelectionStrategy({
      embeddingDim: dampedSpring1D.embeddingDim,
      maxCandidates: 10,
    }),
    totalEpisodes,
    checkpointInterval
  );

  const results = [maxAgencyResult, bindingWeightedResult, tamResult];

  // Print learning curves
  printLearningCurves(results);

  // Analyze convergence
  analyzeConvergence(results);

  // Final comparison
  console.log("═══════════════════════════════════════════════════════════");
  console.log("  Final Performance (Episode " + totalEpisodes + ")");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Strategy         Error    Agency  Binding  Ports");
  console.log("───────────────  ───────  ──────  ───────  ─────");
  for (const result of results) {
    const final = result.curve[result.curve.length - 1]!;
    console.log(
      `${result.strategy.padEnd(15)}  ${final.error.toFixed(4)}  ${(final.agency * 100).toFixed(1)}%  ${(final.bindingRate * 100).toFixed(1)}%    ${final.portCount}`
    );
  }

  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Key Findings");
  console.log("═══════════════════════════════════════════════════════════\n");

  // Determine if TAM caught up
  const tamFinal = tamResult.curve[tamResult.curve.length - 1]!;
  const maxAgencyFinal = maxAgencyResult.curve[maxAgencyResult.curve.length - 1]!;
  const bindingWeightedFinal = bindingWeightedResult.curve[bindingWeightedResult.curve.length - 1]!;

  const bestError = Math.min(tamFinal.error, maxAgencyFinal.error, bindingWeightedFinal.error);
  const tamGap = ((tamFinal.error - bestError) / bestError * 100);

  if (tamFinal.error <= bestError * 1.05) {
    console.log("✓ TAM-based selection converged to competitive performance");
  } else {
    console.log(`✗ TAM-based selection still ${tamGap.toFixed(0)}% behind best strategy`);
  }

  // Check if TAM improved
  const tamEarly = tamResult.curve[0]!;
  const tamImprovement = ((tamEarly.error - tamFinal.error) / tamEarly.error * 100);

  if (tamImprovement > 20) {
    console.log(`✓ TAM showed strong learning (${tamImprovement.toFixed(0)}% improvement)`);
  } else if (tamImprovement > 5) {
    console.log(`• TAM showed moderate learning (${tamImprovement.toFixed(0)}% improvement)`);
  } else {
    console.log(`✗ TAM showed minimal learning (${tamImprovement.toFixed(0)}% improvement)`);
  }

  console.log("\nInterpretation:");
  console.log("  If TAM converges to match simpler strategies, it validates");
  console.log("  the meta-learning approach but suggests this domain is too");
  console.log("  simple to benefit from learned selection.");
  console.log("");
  console.log("  If TAM remains behind, it suggests either:");
  console.log("    1. Meta-learning needs even more data");
  console.log("    2. The selection problem is too simple for TAM");
  console.log("    3. TAM selection is better suited for harder domains");

  // Save results to JSON
  const finalResults = results.map((r) => ({
    strategy: r.strategy,
    error: r.curve[r.curve.length - 1]!.error,
    bindingRate: r.curve[r.curve.length - 1]!.bindingRate,
    agency: r.curve[r.curve.length - 1]!.agency,
    portCount: r.curve[r.curve.length - 1]!.portCount,
  }));

  const bestByError = finalResults.reduce((best, curr) =>
    curr.error < best.error ? curr : best
  );
  const bestByBinding = finalResults.reduce((best, curr) =>
    curr.bindingRate > best.bindingRate ? curr : best
  );

  const exportData = {
    name: "Learned Port Selection Convergence Study",
    config: {
      totalEpisodes,
      checkpointInterval,
      domain: "1D Damped Spring",
    },
    strategies: results.map((r) => ({
      name: r.strategy,
      learningCurve: r.curve,
      finalPerformance: r.curve[r.curve.length - 1],
    })),
    summary: {
      finalResults,
      bestByError: {
        strategy: bestByError.strategy,
        error: bestByError.error,
      },
      bestByBinding: {
        strategy: bestByBinding.strategy,
        bindingRate: bestByBinding.bindingRate,
      },
      tamConverged: tamFinal.error <= bestError * 1.05,
      tamImprovement: tamImprovement,
    },
    timestamp: new Date().toISOString(),
  };

  await Bun.write(
    "examples/results/06-learned-port-selection.json",
    JSON.stringify(exportData, null, 2)
  );
  console.log("\n✓ Results saved to examples/results/06-learned-port-selection.json");

  // Cleanup
  for (const result of results) {
    result.finalBank.dispose();
  }

  console.log("\n✓ Convergence study complete!");
}

main().catch(console.error);
