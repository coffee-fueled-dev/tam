import type { IGraph } from "../graph";
import { DegreeScorer, type IBrowserHubScorer } from "./scorers";

interface Node {
  id: number;
  token: string;
  hubScore: number;
}

interface Edge {
  from_id: number;
  to_id: number;
  weight: number;
}

/**
 * In-memory implementation of a directed graph for token transitions.
 */
export class Graph implements IGraph {
  private nodes: Map<string, Node> = new Map();
  private nodeIdCounter = 0;
  private edges: Map<string, Edge> = new Map();
  private adjacencyList: Map<number, Map<number, number>> = new Map();
  private scorer: IBrowserHubScorer;

  constructor(scorer: IBrowserHubScorer = new DegreeScorer()) {
    this.scorer = scorer;
  }

  /**
   * Gets or creates a node for a token.
   */
  getOrCreateNode(token: string): number {
    let node = this.nodes.get(token);
    if (!node) {
      node = {
        id: ++this.nodeIdCounter,
        token,
        hubScore: 0,
      };
      this.nodes.set(token, node);
      this.adjacencyList.set(node.id, new Map());
    }
    return node.id;
  }

  merge(from: string, to: string): { from_id: number; to_id: number } {
    const from_id = this.getOrCreateNode(from);
    const to_id = this.getOrCreateNode(to);

    const key = `${from_id}:${to_id}`;
    const edge = this.edges.get(key);

    if (edge) {
      edge.weight++;
    } else {
      this.edges.set(key, { from_id, to_id, weight: 1 });
    }

    const fromAdj = this.adjacencyList.get(from_id)!;
    fromAdj.set(to_id, (fromAdj.get(to_id) || 0) + 1);

    return { from_id, to_id };
  }

  mergeBatch(pairs: [string, string][]): { from_id: number; to_id: number }[] {
    return pairs.map(([from, to]) => this.merge(from, to));
  }

  getNext(from: string): { to: string; weight: number }[] {
    const node = this.nodes.get(from);
    if (!node) return [];

    const adj = this.adjacencyList.get(node.id);
    if (!adj) return [];

    const results: { to: string; weight: number }[] = [];
    for (const [to_id, weight] of adj.entries()) {
      const toNode = Array.from(this.nodes.values()).find(
        (n) => n.id === to_id
      );
      if (toNode) {
        results.push({ to: toNode.token, weight });
      }
    }

    return results;
  }

  getTopTokens(limit = 10): { pattern: string; confidence: number }[] {
    this.scorer.compute({
      nodes: this.nodes,
      adjacencyList: this.adjacencyList,
    });

    return Array.from(this.nodes.values())
      .sort((a, b) => b.hubScore - a.hubScore)
      .slice(0, limit)
      .map((node) => ({ pattern: node.token, confidence: node.hubScore }));
  }
}
