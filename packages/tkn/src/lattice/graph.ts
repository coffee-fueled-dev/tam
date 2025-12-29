/**
 * Interface for a directed graph that stores token transitions (Markov chain).
 */
export interface IGraph {
  /**
   * Adds a transition between two tokens (creates nodes if needed).
   * @param from - Source token
   * @param to - Destination token
   * @returns Object containing the from and to node IDs
   */
  merge(from: string, to: string): { from_id: number; to_id: number };

  /**
   * Bulk transition insertion for large sequences.
   * @param pairs - Array of [from, to] token pairs
   * @returns Array of objects containing from and to node IDs
   */
  mergeBatch(pairs: [string, string][]): { from_id: number; to_id: number }[];

  /**
   * Retrieves all outgoing transitions for a token.
   * @param from - The source token
   * @returns Array of transitions with weights
   */
  getNext(from: string): { to: string; weight: number }[];

  /**
   * Returns top N tokens by hub score using the configured scoring algorithm.
   * @param limit - Number of tokens to return (default 10)
   * @returns Array of tokens with confidences
   */
  getTopTokens(limit?: number): { pattern: string; confidence: number }[];
}
