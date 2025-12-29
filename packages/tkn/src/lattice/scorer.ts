/**
 * Generic interface for hub scoring algorithms.
 * @template TContext - The context type the scorer operates on (e.g., Database, in-memory structures)
 */
export interface IHubScorer<TContext> {
  /**
   * Computes hub scores for all nodes in the graph.
   * @param context - The context containing the graph data
   */
  compute(context: TContext): void;
}
