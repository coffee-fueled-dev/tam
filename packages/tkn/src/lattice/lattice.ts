/**
 * Interface for a Lattice that composes a Trie for token storage and a Graph for transitions.
 */
export interface ILattice {
  /**
   * Merge sequences of tokens into the Lattice
   * @param pairs - Array of [from, to] token pairs
   */
  merge(pairs: [string, string][]): void;

  /**
   * Retrieves all outgoing transitions for a token.
   * @param from - The source token
   * @returns Array of transitions with weights
   */
  getNext(from: string): { to: string; weight: number }[];

  /**
   * Gets immediate child characters of a prefix in the trie.
   * @param prefix - The prefix to search for
   * @returns Array of child characters
   */
  nextCharacters(prefix: string): string[];

  /**
   * Returns top N tokens by hub score using the configured scoring algorithm.
   * @param limit - Number of tokens to return (default 10)
   * @returns Array of tokens with confidences
   */
  getTopTokens(limit?: number): { pattern: string; confidence: number }[];

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
  pipe(
    source: AsyncGenerator<{ key: string; sequence: string[] }, void, unknown>,
    batchSize?: number
  ): Promise<void>;

  /**
   * Closes the underlying storage/database connection.
   */
  close(): void;
}
