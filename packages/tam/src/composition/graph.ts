/**
 * CompositionGraph: Finds paths through the port network.
 *
 * Handles undecidability by:
 * - Trying direct paths first
 * - Falling back to multi-hop if direct fails
 * - Returning null if no path found within budget
 */

import type { PortRegistry } from "./registry";
import type { FunctorCache } from "./cache";
import type { CompositionPath, CompositionStep, FunctorResult } from "./types";

/**
 * Implementation of CompositionPath.
 */
class CompositionPathImpl implements CompositionPath {
  steps: CompositionStep[];
  totalBindingRate: number;

  constructor(steps: CompositionStep[]) {
    this.steps = steps;
    this.totalBindingRate = steps.reduce(
      (acc, step) => acc * step.bindingRate,
      1
    );
  }

  describe(): string {
    if (this.steps.length === 0) return "(empty path)";

    const parts = [this.steps[0]!.from];
    for (const step of this.steps) {
      parts.push(step.to);
    }
    return parts.join(" → ");
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
}
