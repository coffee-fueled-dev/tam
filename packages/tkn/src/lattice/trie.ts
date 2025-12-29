/**
 * Interface for a prefix trie that stores tokens character by character.
 */
export interface ITrie {
  /**
   * Inserts a token into the trie, creating nodes character by character.
   * @param token - The token to insert
   * @param markov_id - The markov node id to associate with this token
   * @returns The terminal node's trie id
   */
  merge(token: string, markov_id: number): number;

  /**
   * Gets immediate child characters of a prefix in the trie.
   * @param prefix - The prefix to search for
   * @returns Array of child characters
   */
  nextCharacters(prefix: string): string[];
}
