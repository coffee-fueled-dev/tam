/**
 * Port Functor Discovery: Intra-domain functor learning for port proliferation.
 *
 * Key insight: Ports within the same domain share a learned manifold (via shared
 * CausalNet/CommitmentNet). When a port has structured failures (bimodal), we can
 * discover systematic transformations to create specialist ports.
 *
 * Instead of random perturbation, we learn:
 *   F: P → P such that the new port F(p_parent) handles failure cases better.
 *
 * Geometric interpretation:
 * - Ports are windows in port space P
 * - Functors are learned transformations in P
 * - A composed port is F(port) where F: P → P
 * - The fibration π: P × X → T applies to composed ports too
 *
 * Algorithm:
 * 1. Cluster failure trajectories (detect bimodal pattern)
 * 2. Compute target embedding that would predict the failure cluster well
 * 3. Learn F to map parent_embedding → target_embedding
 * 4. Return F as a reusable transformation (enables composition)
 */

import * as tf from "@tensorflow/tfjs";
import type { Vec } from "../vec";
import { sub, add, scale as vecScale, norm, mean } from "../vec";
import type { Situation } from "../types";
import { CausalNet } from "./causal";

/**
 * Simple port functor using affine transformation: F(p) = A*p + b
 * This is sufficient for many structured transformations (scaling, translation, rotation).
 */
export class PortFunctor {
  private model: tf.LayersModel;
  private optimizer: tf.Optimizer;
  public readonly dim: number;

  constructor(dim: number, learningRate = 0.01) {
    this.dim = dim;

    // Single dense layer with bias = affine transformation
    const input = tf.input({ shape: [dim] });
    const output = tf.layers
      .dense({
        units: dim,
        useBias: true,
        kernelInitializer: "identity", // Start close to identity
        biasInitializer: "zeros"
      })
      .apply(input) as tf.SymbolicTensor;

    this.model = tf.model({ inputs: input, outputs: output });
    this.optimizer = tf.train.adam(learningRate);
  }

  /**
   * Apply functor to transform a port embedding.
   */
  apply(portEmbedding: Vec): Vec {
    return tf.tidy(() => {
      const input = tf.tensor2d([portEmbedding]);
      const output = this.model.predict(input) as tf.Tensor;
      return Array.from(output.dataSync());
    });
  }

  /**
   * Train the functor to map source → target embedding.
   */
  trainStep(sourceEmbedding: Vec, targetEmbedding: Vec): number {
    let lossValue = 0;

    tf.tidy(() => {
      const sourceT = tf.tensor2d([sourceEmbedding]);
      const targetT = tf.tensor2d([targetEmbedding]);

      this.optimizer.minimize(() => {
        const predicted = this.model.predict(sourceT) as tf.Tensor;
        const mse = tf.losses.meanSquaredError(targetT, predicted);
        lossValue = (mse as tf.Scalar).dataSync()[0]!;
        return mse as tf.Scalar;
      });
    });

    return lossValue;
  }

  /**
   * Get the error when applying functor.
   */
  getError(sourceEmbedding: Vec, targetEmbedding: Vec): number {
    const predicted = this.apply(sourceEmbedding);
    return Math.sqrt(
      predicted.reduce((s, p, i) => s + (p - targetEmbedding[i]!) ** 2, 0) /
        predicted.length
    );
  }

  dispose(): void {
    this.model.dispose();
    this.optimizer.dispose();
  }
}

/**
 * Configuration for port functor discovery.
 */
export interface PortFunctorDiscoveryConfig {
  /** Maximum training epochs */
  maxEpochs: number;
  /** Learning rate */
  learningRate: number;
  /** Error tolerance for accepting functor */
  tolerance: number;
  /** Minimum failure samples needed */
  minSamples: number;
}

export const defaultPortFunctorConfig: PortFunctorDiscoveryConfig = {
  maxEpochs: 50,
  learningRate: 0.01,
  tolerance: 0.3,
  minSamples: 5,
};

/**
 * Cluster failure trajectories into two groups using k-means.
 * Returns null if trajectories don't form clear clusters.
 */
function clusterTrajectories(
  trajectories: Vec[]
): { c1: Vec[]; c2: Vec[]; center1: Vec; center2: Vec } | null {
  if (trajectories.length < 2) return null;

  const n = trajectories.length;
  const dim = trajectories[0]!.length;

  // Initialize: first and last trajectory
  let center1 = [...trajectories[0]!];
  let center2 = [...trajectories[n - 1]!];

  // k-means with k=2
  for (let iter = 0; iter < 10; iter++) {
    const cluster1: Vec[] = [];
    const cluster2: Vec[] = [];

    // Assign to nearest cluster
    for (const t of trajectories) {
      const d1 = norm(sub(t, center1));
      const d2 = norm(sub(t, center2));
      if (d1 < d2) {
        cluster1.push(t);
      } else {
        cluster2.push(t);
      }
    }

    if (cluster1.length === 0 || cluster2.length === 0) {
      return null; // Degenerate case
    }

    // Update centers
    center1 = mean(cluster1);
    center2 = mean(cluster2);
  }

  // Assign final clusters
  const c1: Vec[] = [];
  const c2: Vec[] = [];
  for (const t of trajectories) {
    const d1 = norm(sub(t, center1));
    const d2 = norm(sub(t, center2));
    if (d1 < d2) {
      c1.push(t);
    } else {
      c2.push(t);
    }
  }

  // Check if clusters are well-separated
  const separation = norm(sub(center1, center2));
  const avgWithinDist =
    (c1.reduce((s, t) => s + norm(sub(t, center1)), 0) / c1.length +
      c2.reduce((s, t) => s + norm(sub(t, center2)), 0) / c2.length) /
    2;

  if (separation < avgWithinDist * 1.5) {
    return null; // Not well-separated
  }

  return { c1, c2, center1, center2 };
}

/**
 * Compute target embedding that would predict a given trajectory cluster well.
 * Uses simple gradient-free optimization: perturb parent embedding to move
 * predictions toward cluster center.
 */
function computeTargetEmbedding(
  parentEmbedding: Vec,
  clusterCenter: Vec,
  situations: Vec[], // Situation embeddings where failures occurred
  causalNet: CausalNet
): Vec {
  // Start from parent embedding
  let bestEmbedding = [...parentEmbedding];
  let bestError = Infinity;

  // Try multiple random perturbations
  for (let trial = 0; trial < 10; trial++) {
    const candidate = parentEmbedding.map(
      (v) => v + (Math.random() - 0.5) * 0.5
    );

    // Evaluate: how well does this embedding predict the cluster center?
    let totalError = 0;
    for (const sitEmb of situations) {
      const predicted = causalNet.predict(sitEmb, candidate);
      const error = norm(sub(predicted, clusterCenter));
      totalError += error;
    }
    const avgError = totalError / situations.length;

    if (avgError < bestError) {
      bestError = avgError;
      bestEmbedding = candidate;
    }
  }

  return bestEmbedding;
}

/**
 * Discover a functor between parent port and failure cluster.
 *
 * Strategy:
 * 1. Cluster failure trajectories (detect systematic pattern)
 * 2. Compute target embedding that would predict the minority cluster well
 * 3. Learn affine transform F: parent_emb → target_emb
 * 4. Return F if error < tolerance, else null
 *
 * @param parentEmbedding - The parent port's embedding
 * @param failures - Array of failure cases with situations and trajectories
 * @param causalNet - Shared CausalNet for evaluation
 * @param config - Discovery configuration
 * @returns Functor if successful, null otherwise
 */
export async function discoverPortFunctor(
  parentEmbedding: Vec,
  failures: Array<{ situationEmb: Vec; trajectory: Vec }>,
  causalNet: CausalNet,
  config: Partial<PortFunctorDiscoveryConfig> = {}
): Promise<PortFunctor | null> {
  const cfg = { ...defaultPortFunctorConfig, ...config };

  if (failures.length < cfg.minSamples) {
    return null; // Not enough data
  }

  // 1. Cluster failure trajectories
  const trajectories = failures.map((f) => f.trajectory);
  const clusters = clusterTrajectories(trajectories);

  if (!clusters) {
    return null; // No clear bimodal pattern
  }

  // 2. Choose the cluster to specialize for (use smaller one)
  const targetCluster =
    clusters.c1.length < clusters.c2.length
      ? { trajs: clusters.c1, center: clusters.center1 }
      : { trajs: clusters.c2, center: clusters.center2 };

  // Get situations for this cluster
  const targetSituations: Vec[] = [];
  for (let i = 0; i < failures.length; i++) {
    const traj = trajectories[i]!;
    const sitEmb = failures[i]!.situationEmb;
    // Check if this trajectory is in target cluster
    const d1 = norm(sub(traj, clusters.center1));
    const d2 = norm(sub(traj, clusters.center2));
    const inTarget =
      targetCluster.center === clusters.center1 ? d1 < d2 : d2 < d1;
    if (inTarget) {
      targetSituations.push(sitEmb);
    }
  }

  if (targetSituations.length === 0) {
    return null;
  }

  // 3. Compute target embedding for the specialist port
  const targetEmbedding = computeTargetEmbedding(
    parentEmbedding,
    targetCluster.center,
    targetSituations,
    causalNet
  );

  // 4. Learn functor: F(parent) → target
  const functor = new PortFunctor(parentEmbedding.length, cfg.learningRate);

  let bestError = Infinity;
  let epochsSinceImprovement = 0;

  for (let epoch = 0; epoch < cfg.maxEpochs; epoch++) {
    functor.trainStep(parentEmbedding, targetEmbedding);

    const error = functor.getError(parentEmbedding, targetEmbedding);

    if (error < bestError) {
      bestError = error;
      epochsSinceImprovement = 0;
    } else {
      epochsSinceImprovement++;
    }

    // Early stopping
    if (error < cfg.tolerance * 0.5) {
      break;
    }

    if (epochsSinceImprovement >= 10) {
      break;
    }
  }

  // 5. Accept functor if error is reasonable
  if (bestError < cfg.tolerance) {
    return functor;
  }

  functor.dispose();
  return null;
}

/**
 * Simple API: Generate a new port embedding for proliferation.
 * Uses hybrid approach: try functor discovery, fallback to random perturbation.
 *
 * @param parentEmbedding - Parent port's embedding
 * @param failures - Failure cases (situation embeddings + trajectories)
 * @param causalNet - Shared CausalNet
 * @param enableFunctors - Whether to try functor discovery
 * @param config - Discovery config
 * @returns New embedding for specialist port
 */
export async function generateSpecialistEmbedding(
  parentEmbedding: Vec,
  failures: Array<{ situationEmb: Vec; trajectory: Vec }>,
  causalNet: CausalNet,
  enableFunctors: boolean = false,
  config?: Partial<PortFunctorDiscoveryConfig>
): Promise<{ embedding: Vec; usedFunctor: boolean }> {
  if (!enableFunctors || failures.length < 5) {
    // Fallback: random perturbation
    const embedding = parentEmbedding.map((v) => v + (Math.random() - 0.5) * 0.2);
    return { embedding, usedFunctor: false };
  }

  // Try functor discovery
  const functor = await discoverPortFunctor(
    parentEmbedding,
    failures,
    causalNet,
    config
  );

  if (functor) {
    const embedding = functor.apply(parentEmbedding);
    functor.dispose();
    return { embedding, usedFunctor: true };
  }

  // Fallback: random perturbation
  const embedding = parentEmbedding.map((v) => v + (Math.random() - 0.5) * 0.2);
  return { embedding, usedFunctor: false };
}
