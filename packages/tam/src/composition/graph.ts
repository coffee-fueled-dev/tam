/**
 * CompositionGraph: Finds paths through the port network.
 *
 * Handles undecidability by:
 * - Trying direct paths first
 * - Falling back to multi-hop if direct fails
 * - Returning null if no path found within budget
 *
 * Path quality is evaluated by:
 * - totalBindingRate: Product of binding rates along the path
 * - commutativityScore: Agreement between alternative paths (if multiple exist)
 *
 * Commutativity is a strong signal of Logical Locality:
 * If Scale → Shift produces the same result as Shift → Scale,
 * the model has captured true compositional structure.
 */

import type { PortRegistry } from "./registry";
import type { FunctorCache } from "./cache";
import type { CompositionPath, CompositionStep, FunctorResult, FunctorNetwork } from "./types";
import type { Vec } from "../vec";

/**
 * Implementation of CompositionPath with commutativity tracking.
 */
class CompositionPathImpl implements CompositionPath {
  steps: CompositionStep[];
  totalBindingRate: number;
  commutativityScore?: number; // 0-1, how well this path agrees with alternatives

  constructor(steps: CompositionStep[], commutativityScore?: number) {
    this.steps = steps;
    this.totalBindingRate = steps.reduce(
      (acc, step) => acc * step.bindingRate,
      1
    );
    this.commutativityScore = commutativityScore;
  }

  describe(): string {
    if (this.steps.length === 0) return "(empty path)";

    const parts = [this.steps[0]!.from];
    for (const step of this.steps) {
      parts.push(step.to);
    }
    let desc = parts.join(" → ");
    if (this.commutativityScore !== undefined) {
      desc += ` [comm: ${(this.commutativityScore * 100).toFixed(1)}%]`;
    }
    return desc;
  }

  /**
   * Quality score combining binding rate and commutativity.
   * Higher = more trustworthy composition.
   */
  getQualityScore(): number {
    const commWeight = this.commutativityScore ?? 0.5; // Default to uncertain
    // Geometric mean of binding rate and commutativity
    return Math.sqrt(this.totalBindingRate * commWeight);
  }
}

/**
 * CompositionGraph: Finds and manages composition paths.
 */
export class CompositionGraph {
  private registry: PortRegistry;
  private cache: FunctorCache;
  private failedAttempts = new Map<string, FunctorResult[]>();

  constructor(registry: PortRegistry, cache: FunctorCache) {
    this.registry = registry;
    this.cache = cache;
  }

  /**
   * Find a composition path between two domains.
   * May trigger functor discovery for untested pairs.
   *
   * @param source - Source domain name
   * @param target - Target domain name
   * @param maxHops - Maximum number of intermediate domains (default: 3)
   * @returns CompositionPath if found, null otherwise
   */
  async findPath(
    source: string,
    target: string,
    maxHops: number = 3
  ): Promise<CompositionPath | null> {
    // Validate domains exist
    if (!this.registry.has(source)) {
      throw new Error(`Source domain "${source}" not registered`);
    }
    if (!this.registry.has(target)) {
      throw new Error(`Target domain "${target}" not registered`);
    }

    // Same domain = empty path
    if (source === target) {
      return new CompositionPathImpl([]);
    }

    // Try direct path first
    const directResult = await this.cache.getFunctor(source, target);
    if (directResult.status === "found" && directResult.functor) {
      return new CompositionPathImpl([
        {
          from: source,
          to: target,
          functor: directResult.functor,
          bindingRate: directResult.bindingRate,
        },
      ]);
    }

    // Record failed attempt
    this.recordFailedAttempt(source, target, directResult);

    // If maxHops is 1, we can't try multi-hop
    if (maxHops <= 1) {
      return null;
    }

    // Try multi-hop via BFS
    const path = await this.findMultiHopPath(source, target, maxHops);
    return path;
  }

  /**
   * Find a multi-hop path using BFS.
   */
  private async findMultiHopPath(
    source: string,
    target: string,
    maxHops: number
  ): Promise<CompositionPath | null> {
    const allDomains = this.registry.list();

    // BFS state: [current domain, path so far]
    const queue: Array<{ domain: string; path: CompositionStep[] }> = [
      { domain: source, path: [] },
    ];
    const visited = new Set<string>([source]);

    while (queue.length > 0) {
      const current = queue.shift()!;

      // Don't exceed max hops
      if (current.path.length >= maxHops) {
        continue;
      }

      // Try each potential next domain
      for (const nextDomain of allDomains) {
        if (visited.has(nextDomain)) continue;

        // Try to find/discover functor
        const result = await this.cache.getFunctor(current.domain, nextDomain);

        if (result.status === "found" && result.functor) {
          const newPath = [
            ...current.path,
            {
              from: current.domain,
              to: nextDomain,
              functor: result.functor,
              bindingRate: result.bindingRate,
            },
          ];

          // Check if we reached target
          if (nextDomain === target) {
            return new CompositionPathImpl(newPath);
          }

          // Continue searching
          visited.add(nextDomain);
          queue.push({ domain: nextDomain, path: newPath });
        } else {
          // Record failed attempt
          this.recordFailedAttempt(current.domain, nextDomain, result);
        }
      }
    }

    // No path found
    return null;
  }

  /**
   * Record a failed functor discovery attempt.
   */
  private recordFailedAttempt(
    source: string,
    target: string,
    result: FunctorResult
  ): void {
    const key = `${source}::${target}`;
    const attempts = this.failedAttempts.get(key) ?? [];
    attempts.push(result);
    this.failedAttempts.set(key, attempts);
  }

  /**
   * Get diagnostic info about failed attempts between two domains.
   */
  getFailedAttempts(source: string, target: string): FunctorResult[] {
    const key = `${source}::${target}`;
    return this.failedAttempts.get(key) ?? [];
  }

  /**
   * Get all pairs that have been attempted.
   */
  getAttemptedPairs(): Array<{
    source: string;
    target: string;
    status: string;
  }> {
    const results: Array<{ source: string; target: string; status: string }> =
      [];

    for (const result of this.cache.getAllCached()) {
      results.push({
        source: result.source,
        target: result.target,
        status: result.status,
      });
    }

    return results;
  }

  /**
   * Get the composition graph as an adjacency list (successful functors only).
   */
  getSuccessfulGraph(): Map<string, string[]> {
    const graph = new Map<string, string[]>();

    for (const result of this.cache.getSuccessful()) {
      const targets = graph.get(result.source) ?? [];
      targets.push(result.target);
      graph.set(result.source, targets);
    }

    return graph;
  }

  /**
   * Check commutativity between two paths.
   *
   * Given paths P1 and P2 from source to target, tests whether they produce
   * approximately the same result for random inputs.
   *
   * Returns a score from 0-1:
   * - 1.0: Paths are perfectly commutative (identical outputs)
   * - 0.0: Paths produce completely different outputs
   *
   * This is a strong signal of Logical Locality: if the model has captured
   * true compositional structure, different paths to the same destination
   * should agree.
   */
  async checkCommutativity(
    path1: CompositionPath,
    path2: CompositionPath,
    numSamples: number = 20
  ): Promise<number> {
    // Paths must have same source and target
    if (path1.steps.length === 0 || path2.steps.length === 0) {
      return 1.0; // Empty paths are trivially commutative
    }

    const source1 = path1.steps[0]!.from;
    const target1 = path1.steps[path1.steps.length - 1]!.to;
    const source2 = path2.steps[0]!.from;
    const target2 = path2.steps[path2.steps.length - 1]!.to;

    if (source1 !== source2 || target1 !== target2) {
      return 0; // Different endpoints, not comparable
    }

    // Generate random inputs and compare outputs
    let totalAgreement = 0;

    for (let i = 0; i < numSamples; i++) {
      // Generate random input in source embedding space
      const input = this.generateRandomInput(8); // Assume 8-dim embedding

      // Apply path1
      const output1 = this.applyPath(input, path1);

      // Apply path2
      const output2 = this.applyPath(input, path2);

      // Compute agreement (1 - normalized distance)
      const agreement = this.computeAgreement(output1, output2);
      totalAgreement += agreement;
    }

    return totalAgreement / numSamples;
  }

  /**
   * Generate a random input vector.
   */
  private generateRandomInput(dim: number): Vec {
    return Array.from({ length: dim }, () => Math.random() * 2 - 1);
  }

  /**
   * Apply a composition path to an input.
   */
  private applyPath(input: Vec, path: CompositionPath): Vec {
    let current = input;
    for (const step of path.steps) {
      current = step.functor.apply(current);
    }
    return current;
  }

  /**
   * Compute agreement between two outputs (1 - normalized distance).
   */
  private computeAgreement(a: Vec, b: Vec): number {
    if (a.length !== b.length) return 0;

    let sumSq = 0;
    let maxMag = 0;
    for (let i = 0; i < a.length; i++) {
      sumSq += (a[i]! - b[i]!) ** 2;
      maxMag = Math.max(maxMag, Math.abs(a[i]!), Math.abs(b[i]!));
    }

    const distance = Math.sqrt(sumSq);
    const normalized = distance / (maxMag * Math.sqrt(a.length) + 1e-8);

    // Convert to agreement: 1 = identical, 0 = very different
    return Math.max(0, 1 - normalized);
  }

  /**
   * Find all paths between two domains and rank by commutativity.
   * More expensive but provides stronger confidence in composition quality.
   */
  async findAllPaths(
    source: string,
    target: string,
    maxHops: number = 3
  ): Promise<CompositionPath[]> {
    const paths: CompositionPath[] = [];

    // Find paths using DFS (allow multiple paths)
    await this.findPathsDFS(source, target, [], new Set(), maxHops, paths);

    // If we have multiple paths, compute pairwise commutativity
    if (paths.length >= 2) {
      const scores = new Map<CompositionPath, number>();

      for (const path of paths) {
        let totalComm = 0;
        let count = 0;

        for (const other of paths) {
          if (path !== other) {
            const comm = await this.checkCommutativity(path, other);
            totalComm += comm;
            count++;
          }
        }

        scores.set(path, count > 0 ? totalComm / count : 0.5);
      }

      // Create new paths with commutativity scores
      return paths.map(
        (p) =>
          new CompositionPathImpl(
            (p as CompositionPathImpl).steps,
            scores.get(p)
          )
      );
    }

    return paths;
  }

  /**
   * DFS to find all paths between two domains.
   */
  private async findPathsDFS(
    current: string,
    target: string,
    pathSoFar: CompositionStep[],
    visited: Set<string>,
    maxHops: number,
    results: CompositionPath[]
  ): Promise<void> {
    if (current === target) {
      results.push(new CompositionPathImpl([...pathSoFar]));
      return;
    }

    if (pathSoFar.length >= maxHops) {
      return;
    }

    visited.add(current);

    for (const nextDomain of this.registry.list()) {
      if (visited.has(nextDomain)) continue;

      const result = await this.cache.getFunctor(current, nextDomain);
      if (result.status === "found" && result.functor) {
        pathSoFar.push({
          from: current,
          to: nextDomain,
          functor: result.functor,
          bindingRate: result.bindingRate,
        });

        await this.findPathsDFS(nextDomain, target, pathSoFar, visited, maxHops, results);

        pathSoFar.pop();
      }
    }

    visited.delete(current);
  }
}
