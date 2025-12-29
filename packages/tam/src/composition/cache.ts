/**
 * FunctorCache: Lazy-loaded cache of discovered functors.
 *
 * Caches functor discovery results (both successes and failures)
 * to avoid redundant computation.
 */

import type {
  FunctorDiscoveryConfig,
  FunctorResult,
  FunctorNetwork,
} from "./types";
import { defaultFunctorDiscoveryConfig } from "./types";

/**
 * Key for cache entries.
 */
function cacheKey(source: string, target: string): string {
  return `${source}::${target}`;
}

/**
 * Lazy-loaded cache of discovered functors.
 */
export class FunctorCache {
  private cache = new Map<string, FunctorResult>();
  private readonly defaultConfig: FunctorDiscoveryConfig;

  /** Function to perform functor discovery */
  private discoveryFn: (
    source: string,
    target: string,
    config: Partial<FunctorDiscoveryConfig>
  ) => Promise<FunctorResult>;

  constructor(
    discoveryFn: (
      source: string,
      target: string,
      config: Partial<FunctorDiscoveryConfig>
    ) => Promise<FunctorResult>,
    config: Partial<FunctorDiscoveryConfig> = {}
  ) {
    this.discoveryFn = discoveryFn;
    this.defaultConfig = { ...defaultFunctorDiscoveryConfig, ...config };
  }

  /**
   * Get a functor between two domains.
   * Attempts discovery on first call, caches result.
   */
  async getFunctor(source: string, target: string): Promise<FunctorResult> {
    const key = cacheKey(source, target);

    // Return cached result if available
    const cached = this.cache.get(key);
    if (cached) {
      return cached;
    }

    // Perform discovery
    const result = await this.discoveryFn(source, target, this.defaultConfig);

    // Cache the result
    this.cache.set(key, result);

    return result;
  }

  /**
   * Check if a functor result is cached without triggering discovery.
   */
  has(source: string, target: string): boolean {
    return this.cache.has(cacheKey(source, target));
  }

  /**
   * Get cached result (including failures) without triggering discovery.
   * Returns null if not cached.
   */
  getCached(source: string, target: string): FunctorResult | null {
    return this.cache.get(cacheKey(source, target)) ?? null;
  }

  /**
   * Retry discovery with different config.
   * Replaces cached result if successful.
   */
  async retry(
    source: string,
    target: string,
    config: Partial<FunctorDiscoveryConfig>
  ): Promise<FunctorResult> {
    const key = cacheKey(source, target);

    // Dispose old functor if exists
    const old = this.cache.get(key);
    if (old?.functor) {
      old.functor.dispose();
    }

    // Perform discovery with new config
    const mergedConfig = { ...this.defaultConfig, ...config };
    const result = await this.discoveryFn(source, target, mergedConfig);

    // Cache the new result
    this.cache.set(key, result);

    return result;
  }

  /**
   * Get all domains that have successful functors from a source.
   */
  functorsFrom(source: string): string[] {
    const targets: string[] = [];

    for (const [key, result] of this.cache.entries()) {
      if (result.source === source && result.status === "found") {
        targets.push(result.target);
      }
    }

    return targets;
  }

  /**
   * Get all domains that have successful functors to a target.
   */
  functorsTo(target: string): string[] {
    const sources: string[] = [];

    for (const [key, result] of this.cache.entries()) {
      if (result.target === target && result.status === "found") {
        sources.push(result.source);
      }
    }

    return sources;
  }

  /**
   * Get all cached results.
   */
  getAllCached(): FunctorResult[] {
    return Array.from(this.cache.values());
  }

  /**
   * Get all successful functors.
   */
  getSuccessful(): FunctorResult[] {
    return Array.from(this.cache.values()).filter((r) => r.status === "found");
  }

  /**
   * Clear the cache and dispose all functors.
   */
  clear(): void {
    for (const result of this.cache.values()) {
      if (result.functor) {
        result.functor.dispose();
      }
    }
    this.cache.clear();
  }

  /**
   * Get the number of cached entries.
   */
  get size(): number {
    return this.cache.size;
  }
}

