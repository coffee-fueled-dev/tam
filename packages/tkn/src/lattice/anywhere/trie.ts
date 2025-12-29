import type { ITrie } from "../trie";

interface TrieNode {
  id: number;
  parent_id: number | null;
  char: string;
  terminal: boolean;
  token?: string;
  markov_id?: number;
  children: Map<string, TrieNode>;
}

/**
 * In-memory implementation of a prefix trie for token storage.
 */
export class Trie implements ITrie {
  private root: TrieNode;
  private nodeIdCounter = 0;

  constructor() {
    this.root = {
      id: ++this.nodeIdCounter,
      parent_id: null,
      char: "",
      terminal: false,
      children: new Map(),
    };
  }

  merge(token: string, markov_id: number): number {
    let current = this.root;

    for (let i = 0; i < token.length; i++) {
      const char = token[i];
      let child = current.children.get(char);

      if (!child) {
        child = {
          id: ++this.nodeIdCounter,
          parent_id: current.id,
          char,
          terminal: false,
          children: new Map(),
        };
        current.children.set(char, child);
      }

      current = child;
    }

    // Mark terminal node
    current.terminal = true;
    current.token = token;
    current.markov_id = markov_id;

    return current.id;
  }

  nextCharacters(prefix: string): string[] {
    let current = this.root;

    for (const char of prefix) {
      const child = current.children.get(char);
      if (!child) return [];
      current = child;
    }

    return Array.from(current.children.keys());
  }
}
