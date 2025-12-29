import type { ILattice } from "../lattice";
import { Graph } from "./graph";
import { Trie } from "./trie";
import { DegreeScorer, type IBrowserHubScorer } from "./scorers";

export interface BrowserLatticeConfig {
  scorer?: IBrowserHubScorer;
}

/**
 * In-memory implementation of a Lattice that composes a Trie and a Graph.
 */
export class Lattice implements ILattice {
  private graph: Graph;
  private trie: Trie;

  constructor(config: BrowserLatticeConfig = {}) {
    const { scorer = new DegreeScorer() } = config;
    this.graph = new Graph(scorer);
    this.trie = new Trie();
  }

  merge(pairs: [string, string][]): void {
    for (const [from, to] of pairs) {
      const { from_id, to_id } = this.graph.merge(from, to);
      this.trie.merge(from, from_id);
      this.trie.merge(to, to_id);
    }
  }

  getNext(from: string): { to: string; weight: number }[] {
    return this.graph.getNext(from);
  }

  nextCharacters(prefix: string): string[] {
    return this.trie.nextCharacters(prefix);
  }

  getTopTokens(limit = 10): { pattern: string; confidence: number }[] {
    return this.graph.getTopTokens(limit);
  }

  async pipe(
    source: AsyncGenerator<{ key: string; sequence: string[] }, void, unknown>,
    batchSize = 1000
  ): Promise<void> {
    let batch: [string, string][] = [];
    let previousPattern: string | null = null;
    let previousMarkovId: number | null = null;

    for await (const segment of source) {
      const currentPattern = segment.key;

      // Get or create graph node for current pattern
      const currentMarkovId = this.graph.getOrCreateNode(currentPattern);

      // Insert sequence elements into trie (handles sentinels as whole elements)
      for (const element of segment.sequence) {
        this.trie.merge(element, currentMarkovId);
      }

      // Build graph transition from previous pattern to current pattern
      if (previousPattern !== null && previousMarkovId !== null) {
        this.graph.merge(previousPattern, currentPattern);

        batch.push([previousPattern, currentPattern]);

        if (batch.length >= batchSize) {
          batch.splice(0, batchSize);
        }
      }

      previousPattern = currentPattern;
      previousMarkovId = currentMarkovId;
    }
  }

  close(): void {
    // No-op for in-memory implementation
  }
}
