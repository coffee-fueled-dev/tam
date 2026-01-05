/**
 * Out-of-Distribution Detection Analysis
 *
 * Analyzes how well TAM's agency signal can detect OOD data.
 *
 * Key hypothesis: Agency should be HIGH on in-distribution data (model confident)
 * and LOW on out-of-distribution data (model uncertain, wide cones).
 *
 * Metrics:
 * - AUROC: Area under ROC curve (1.0 = perfect separation, 0.5 = random)
 * - Histograms: Distribution of agency for in-dist vs out-dist
 * - Optimal threshold: Agency cutoff that best separates distributions
 */

import type { Actor } from "../actor";

export interface OODSample {
  agency: number;
  error: number;
  avgConeRadius: number;
  bound: boolean; // Did prediction fall within cone?
  isInDist: boolean;
}

export interface OODAnalysisResult {
  // AUROC metrics (using different signals)
  aurocAgency: number; // Using agency as OOD detector
  aurocCalibration: number; // Using calibration gap as OOD detector

  // Separation quality
  agencyGap: number; // Mean in-dist - mean out-dist
  calibrationGap: number; // Calibration gap difference

  // Optimal threshold (maximizes TPR - FPR)
  optimalThreshold: number;
  optimalTPR: number; // True positive rate at threshold
  optimalFPR: number; // False positive rate at threshold

  // Distribution statistics
  inDistAgency: { mean: number; std: number };
  outDistAgency: { mean: number; std: number };
  inDistBindingRate: number;
  outDistBindingRate: number;
  inDistCalibration: number; // |agency - binding_rate|
  outDistCalibration: number;

  // ROC curve points for plotting
  rocCurve: Array<{ fpr: number; tpr: number; threshold: number }>;

  // Sample data
  samples: OODSample[];
}

/**
 * Compute AUROC from samples.
 * Uses trapezoidal rule to integrate under ROC curve.
 */
function computeAUROC(samples: OODSample[]): { auroc: number; rocCurve: Array<{ fpr: number; tpr: number; threshold: number }> } {
  // Sort by agency (descending) - higher agency = more confident = predicted in-dist
  const sorted = [...samples].sort((a, b) => b.agency - a.agency);

  const totalPositive = samples.filter(s => s.isInDist).length;
  const totalNegative = samples.filter(s => !s.isInDist).length;

  if (totalPositive === 0 || totalNegative === 0) {
    return { auroc: 0.5, rocCurve: [] };
  }

  const rocCurve: Array<{ fpr: number; tpr: number; threshold: number }> = [];
  let truePositives = 0;
  let falsePositives = 0;
  let auc = 0;
  let prevFPR = 0;

  // Add point at (0, 0) - threshold = infinity (reject all)
  rocCurve.push({ fpr: 0, tpr: 0, threshold: Infinity });

  for (let i = 0; i < sorted.length; i++) {
    const sample = sorted[i]!;

    // Count this sample as predicted in-dist (agency >= threshold)
    if (sample.isInDist) {
      truePositives++;
    } else {
      falsePositives++;
    }

    const tpr = truePositives / totalPositive;
    const fpr = falsePositives / totalNegative;

    // Trapezoidal rule: area of trapezoid
    auc += (fpr - prevFPR) * (tpr + (i > 0 ? truePositives - (sample.isInDist ? 1 : 0) : 0) / totalPositive) / 2;

    rocCurve.push({ fpr, tpr, threshold: sample.agency });
    prevFPR = fpr;
  }

  // Add point at (1, 1) - threshold = -infinity (accept all)
  rocCurve.push({ fpr: 1, tpr: 1, threshold: -Infinity });

  return { auroc: auc, rocCurve };
}

/**
 * Find optimal threshold (maximizes Youden's J = TPR - FPR)
 */
function findOptimalThreshold(rocCurve: Array<{ fpr: number; tpr: number; threshold: number }>) {
  let maxJ = -Infinity;
  let optimal = { threshold: 0.5, tpr: 0, fpr: 0 };

  for (const point of rocCurve) {
    const j = point.tpr - point.fpr; // Youden's J statistic
    if (j > maxJ) {
      maxJ = j;
      optimal = { threshold: point.threshold, tpr: point.tpr, fpr: point.fpr };
    }
  }

  return optimal;
}

/**
 * Compute mean and standard deviation
 */
function stats(values: number[]): { mean: number; std: number } {
  if (values.length === 0) return { mean: 0, std: 0 };

  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);

  return { mean, std };
}

/**
 * Check if trajectory falls within cone (binding).
 */
function checkBinding(actualDelta: number[], cone: { center: number[]; radius: number[] }): boolean {
  let sumSq = 0;
  for (let i = 0; i < actualDelta.length; i++) {
    const residual = actualDelta[i]! - cone.center[i]!;
    const normalized = cone.radius[i]! > 0 ? residual / cone.radius[i]! : residual;
    sumSq += normalized * normalized;
  }
  const normalizedDistance = Math.sqrt(sumSq);
  return normalizedDistance <= 1.0;
}

/**
 * Analyze OOD detection capability.
 *
 * Evaluates actor on both in-distribution and out-of-distribution test sets,
 * measuring agency, binding rate, and calibration quality.
 *
 * Key insight: OOD should show MISCALIBRATION (high agency, low binding).
 */
export function analyzeOODDetection<S>(
  actor: Actor<S>,
  inDistStates: S[],
  outDistStates: S[],
  embedState: (s: S) => number[],
  stepFn: (s: S) => S
): OODAnalysisResult {
  const samples: OODSample[] = [];

  // Collect in-distribution samples
  for (const state of inDistStates) {
    const pred = actor.predict(state);
    const actual = stepFn(state);
    const actualEmb = embedState(actual);
    const beforeEmb = embedState(state);
    const actualDelta = actualEmb.map((v, i) => v - beforeEmb[i]!);

    const error = Math.sqrt(
      actualDelta.reduce((sum, d, i) => sum + (d - pred.delta[i]!) ** 2, 0) / actualDelta.length
    );

    const avgConeRadius = pred.cone.radius.reduce((sum, r) => sum + r, 0) / pred.cone.radius.length;
    const bound = checkBinding(actualDelta, pred.cone);

    samples.push({
      agency: pred.agency,
      error,
      avgConeRadius,
      bound,
      isInDist: true,
    });
  }

  // Collect out-of-distribution samples
  for (const state of outDistStates) {
    const pred = actor.predict(state);
    const actual = stepFn(state);
    const actualEmb = embedState(actual);
    const beforeEmb = embedState(state);
    const actualDelta = actualEmb.map((v, i) => v - beforeEmb[i]!);

    const error = Math.sqrt(
      actualDelta.reduce((sum, d, i) => sum + (d - pred.delta[i]!) ** 2, 0) / actualDelta.length
    );

    const avgConeRadius = pred.cone.radius.reduce((sum, r) => sum + r, 0) / pred.cone.radius.length;
    const bound = checkBinding(actualDelta, pred.cone);

    samples.push({
      agency: pred.agency,
      error,
      avgConeRadius,
      bound,
      isInDist: false,
    });
  }

  // Compute AUROC using agency
  const { auroc: aurocAgency, rocCurve } = computeAUROC(samples);

  // Compute AUROC using calibration gap as OOD signal
  // Higher calibration gap = more likely OOD
  const samplesWithCalibrationScore = samples.map(s => ({
    ...s,
    // Invert agency for calibration: if bound=false but agency=high, score is high (miscalibration)
    agency: s.bound ? (1 - s.agency) : s.agency, // Higher score = more likely OOD
  }));
  const { auroc: aurocCalibration } = computeAUROC(samplesWithCalibrationScore);

  // Find optimal threshold
  const optimal = findOptimalThreshold(rocCurve);

  // Distribution statistics
  const inDistSamples = samples.filter(s => s.isInDist);
  const outDistSamples = samples.filter(s => !s.isInDist);

  const inDistAgencies = inDistSamples.map(s => s.agency);
  const outDistAgencies = outDistSamples.map(s => s.agency);

  const inDistAgency = stats(inDistAgencies);
  const outDistAgency = stats(outDistAgencies);
  const agencyGap = inDistAgency.mean - outDistAgency.mean;

  // Binding rates
  const inDistBindingRate = inDistSamples.filter(s => s.bound).length / inDistSamples.length;
  const outDistBindingRate = outDistSamples.filter(s => s.bound).length / outDistSamples.length;

  // Calibration: |agency - binding_rate|
  const inDistCalibration = Math.abs(inDistAgency.mean - inDistBindingRate);
  const outDistCalibration = Math.abs(outDistAgency.mean - outDistBindingRate);
  const calibrationGap = outDistCalibration - inDistCalibration;

  return {
    aurocAgency,
    aurocCalibration,
    agencyGap,
    calibrationGap,
    optimalThreshold: optimal.threshold,
    optimalTPR: optimal.tpr,
    optimalFPR: optimal.fpr,
    inDistAgency,
    outDistAgency,
    inDistBindingRate,
    outDistBindingRate,
    inDistCalibration,
    outDistCalibration,
    rocCurve,
    samples,
  };
}

/**
 * Print OOD analysis results to console
 */
export function printOODAnalysis(result: OODAnalysisResult) {
  console.log("\n═══════════════════════════════════════════════════════════");
  console.log("  Out-of-Distribution Detection Analysis");
  console.log("═══════════════════════════════════════════════════════════\n");

  console.log("Agency Distribution:");
  console.log(`  In-distribution:     ${(result.inDistAgency.mean * 100).toFixed(1)}% ± ${(result.inDistAgency.std * 100).toFixed(1)}%`);
  console.log(`  Out-of-distribution: ${(result.outDistAgency.mean * 100).toFixed(1)}% ± ${(result.outDistAgency.std * 100).toFixed(1)}%`);
  console.log(`  Agency gap:          ${(result.agencyGap * 100).toFixed(1)}%`);

  console.log(`\nBinding Rate (Actual Performance):`);
  console.log(`  In-distribution:     ${(result.inDistBindingRate * 100).toFixed(1)}%`);
  console.log(`  Out-of-distribution: ${(result.outDistBindingRate * 100).toFixed(1)}%`);
  console.log(`  Binding gap:         ${((result.inDistBindingRate - result.outDistBindingRate) * 100).toFixed(1)}%`);

  console.log(`\nCalibration Quality (|Agency - Binding|):`);
  console.log(`  In-distribution:     ${(result.inDistCalibration * 100).toFixed(1)}%`);
  console.log(`  Out-of-distribution: ${(result.outDistCalibration * 100).toFixed(1)}%`);
  console.log(`  Calibration gap:     ${(result.calibrationGap * 100).toFixed(1)}%`);

  console.log(`\nOOD Detection via Miscalibration:`);
  if (result.calibrationGap > 0.2) {
    console.log(`  ✓ STRONG OOD SIGNAL: Major miscalibration on OOD data`);
    console.log(`    Model is overconfident (high agency, low binding)`);
    console.log(`    This is the classic signature of out-of-distribution data`);
  } else if (result.calibrationGap > 0.1) {
    console.log(`  ✓ Moderate OOD signal: Calibration degrades on OOD`);
    console.log(`    Model somewhat overconfident on unfamiliar data`);
  } else if (result.calibrationGap > 0) {
    console.log(`  ~ Weak OOD signal: Slight miscalibration on OOD`);
  } else {
    console.log(`  ✗ No OOD signal: Model equally calibrated on both`);
    console.log(`    Either OOD is not truly different, or model extrapolates well`);
  }

  console.log(`\nAUROC Metrics:`);
  console.log(`  Using agency:        ${result.aurocAgency.toFixed(3)} ${result.aurocAgency > 0.7 ? "✓" : "⚠"}`);
  console.log(`  Using calibration:   ${result.aurocCalibration.toFixed(3)} ${result.aurocCalibration > 0.7 ? "✓" : "⚠"}`);
  console.log(`    (1.0 = perfect separation, 0.5 = random)`);
}
