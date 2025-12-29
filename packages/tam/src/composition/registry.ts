/**
 * PortRegistry: Central registry of trained domain ports.
 *
 * Allows registration and lookup of GeometricPortBanks by name.
 */

import type { GeometricPortBank } from "../geometric";
import type { PortMetadata } from "./types";

/**
 * Registered port entry with metadata.
 */
interface RegistryEntry {
  port: GeometricPortBank<unknown, unknown>;
  metadata: PortMetadata;
  /** Function to get embedding dimension from a sample */
  getEmbeddingDim: () => number;
}

/**
 * Central registry of trained domain ports.
 */
export class PortRegistry {
  private ports = new Map<string, RegistryEntry>();

  /**
   * Register a port with a unique name.
   *
   * @param name - Unique identifier for the port
   * @param port - The trained GeometricPortBank
   * @param embeddingDim - Dimension of state embeddings for this domain
   */
  register<S, C = unknown>(
    name: string,
    port: GeometricPortBank<S, C>,
    embeddingDim: number
  ): void {
    if (this.ports.has(name)) {
      throw new Error(`Port "${name}" is already registered`);
    }

    this.ports.set(name, {
      port: port as GeometricPortBank<unknown, unknown>,
      metadata: {
        name,
        embeddingDim,
        registeredAt: new Date(),
      },
      getEmbeddingDim: () => embeddingDim,
    });
  }

  /**
   * Get a registered port by name.
   */
  get<S = unknown, C = unknown>(name: string): GeometricPortBank<S, C> {
    const entry = this.ports.get(name);
    if (!entry) {
      throw new Error(`Port "${name}" not found in registry`);
    }
    return entry.port as GeometricPortBank<S, C>;
  }

  /**
   * Get metadata for a registered port.
   */
  getMetadata(name: string): PortMetadata {
    const entry = this.ports.get(name);
    if (!entry) {
      throw new Error(`Port "${name}" not found in registry`);
    }
    return entry.metadata;
  }

  /**
   * Check if a port is registered.
   */
  has(name: string): boolean {
    return this.ports.has(name);
  }

  /**
   * List all registered port names.
   */
  list(): string[] {
    return Array.from(this.ports.keys());
  }

  /**
   * Get all registered ports with metadata.
   */
  listWithMetadata(): PortMetadata[] {
    return Array.from(this.ports.values()).map((entry) => entry.metadata);
  }

  /**
   * Unregister a port.
   */
  unregister(name: string): boolean {
    return this.ports.delete(name);
  }

  /**
   * Clear all registered ports.
   */
  clear(): void {
    this.ports.clear();
  }

  /**
   * Get the number of registered ports.
   */
  get size(): number {
    return this.ports.size;
  }
}

