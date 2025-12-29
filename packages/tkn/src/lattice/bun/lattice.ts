import { Database } from "bun:sqlite";
import type { ILattice } from "../lattice";
import { Trie } from "./trie";
import { Graph, DegreeScorer, type IBunHubScorer } from "./graph";

export interface BunLatticeConfig {
  filename?: string;
  scorer?: IBunHubScorer;
}

/**
 * Composes a Trie for token storage and a Graph for transitions.
 */
export class Lattice implements ILattice {
  private db: Database;
  private trie: Trie;
  private graph: Graph;

  constructor(config: BunLatticeConfig | string = {}) {
    // Support legacy string parameter for filename
    const { filename = ":memory:", scorer = new DegreeScorer() } =
      typeof config === "string" ? { filename: config } : config;

    this.db = new Database(filename, { create: true });
    this.db.run("PRAGMA journal_mode = WAL;");
    this.db.run("PRAGMA synchronous = OFF;");
    this.db.run("PRAGMA temp_store = MEMORY;");

    // Initialize both components with the same database
    this.graph = new Graph(this.db, scorer);
    this.trie = new Trie(this.db);
  }

  /**
   * Bulk transition insertion for large sequences (transactional).
   * @param pairs - Array of [from, to] token pairs
   */
  merge(pairs: [string, string][]): void {
    const tx = this.db.transaction(() => {
      for (let i = 0; i < pairs.length; i++) {
        const { from_id, to_id } = this.graph.merge(pairs[i][0], pairs[i][1]);
        this.trie.merge(pairs[i][0], from_id);
        this.trie.merge(pairs[i][1], to_id);
      }
    });
    tx();
  }

  /**
   * Retrieves all outgoing transitions for a token.
   * @param from - The source token
   * @returns Array of transitions with weights
   */
  getNext(from: string): { to: string; weight: number }[] {
    return this.graph.getNext(from);
  }

  /**
   * Gets immediate child characters of a prefix in the trie.
   * @param prefix - The prefix to search for
   * @returns Array of child characters
   */
  nextCharacters(prefix: string): string[] {
    return this.trie.nextCharacters(prefix);
  }

  /**
   * Returns top N tokens by hub score using the configured scoring algorithm.
   * @param limit - Number of tokens to return (default 10)
   * @returns Array of tokens with hub scores
   */
  getTopTokens(limit = 10): { pattern: string; confidence: number }[] {
    return this.graph.getTopTokens(limit);
  }

  /**
   * Pipes sequences from an async generator into the lattice.
   * - Trie: Stores individual sequence elements (characters + sentinels) linked to graph nodes
   * - Graph: Builds transitions between consecutive pattern keys
   *
   * Example: sequence ["t", "h", "e", "<0>"] with key "the<0>" creates:
   * - Graph node for pattern "the<0>"
   * - Trie nodes for "t", "h", "e", and "<0>" (sentinel stored as-is)
   * - Graph transitions between consecutive patterns
   *
   * @param source - AsyncGenerator that yields sequences with keys (e.g., from ISequencer.read())
   * @param batchSize - Number of pairs to batch before merging (default 1000)
   */
  async pipe(
    source: AsyncGenerator<{ key: string; sequence: string[] }, void, unknown>,
    batchSize = 1000
  ): Promise<void> {
    let batch: [string, string][] = [];
    let previousPattern: string | null = null;

    for await (const segment of source) {
      const currentPattern = segment.key;

      // Get or create graph node for current pattern
      const currentMarkovId = this.graph.getOrCreateNode(currentPattern);

      // Insert sequence elements into trie (handles sentinels as whole elements)
      for (const element of segment.sequence) {
        this.trie.merge(element, currentMarkovId);
      }

      // Build graph transition from previous pattern to current pattern
      if (previousPattern !== null) {
        batch.push([previousPattern, currentPattern]);

        if (batch.length >= batchSize) {
          this.merge(batch.splice(0, batchSize));
        }
      }

      previousPattern = currentPattern;
    }

    // Merge remaining pairs
    if (batch.length > 0) {
      this.merge(batch.splice(0));
    }
  }

  /**
   * Closes the database connection.
   */
  close(): void {
    this.db.close();
  }
}
