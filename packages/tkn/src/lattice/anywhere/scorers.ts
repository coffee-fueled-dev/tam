import type { IHubScorer } from "../scorer";

interface Node {
  id: number;
  token: string;
  hubScore: number;
}

export interface BrowserGraphContext {
  nodes: Map<string, Node>;
  adjacencyList: Map<number, Map<number, number>>;
}

/**
 * Browser-specific hub scorer interface.
 */
export interface IBrowserHubScorer extends IHubScorer<BrowserGraphContext> {}

/**
 * Computes local hub scores for nodes based on weighted out-degree.
 *
 * The degree score represents each node's local connectivity strength,
 * calculated as the sum of the weights of its outgoing transitions:
 *
 *    degree(node) = Σ[ w(node → to) ]
 *
 * The resulting hub score is stored as:
 *
 *    hubScore = log1p(degree)
 *
 * where `log1p` provides a smooth logarithmic scaling to prevent
 * extremely high-degree nodes from dominating.
 *
 * This measure is a fast, local approximation of node importance,
 * suitable for online updates or incremental scoring where full
 * PageRank computation would be too expensive.
 */
export class DegreeScorer implements IBrowserHubScorer {
  compute(context: BrowserGraphContext): void {
    for (const node of context.nodes.values()) {
      const adjacency = context.adjacencyList.get(node.id);

      // Sum all outgoing edge weights
      let degree = 0;
      if (adjacency) {
        for (const weight of adjacency.values()) {
          degree += weight;
        }
      }

      // Apply logarithmic scaling
      node.hubScore = Math.log1p(degree);
    }
  }
}
