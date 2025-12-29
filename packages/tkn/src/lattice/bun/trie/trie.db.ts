import { Database, Statement } from "bun:sqlite";
import type { TrieNode, TrieNodeInsert, TrieNodeUpdate } from "./trie.model";

// Statement types
export type InsertTrieNodeStmt = Statement<
  Pick<TrieNode, "id">,
  [TrieNodeInsert]
>;
export type SelectTrieNodeStmt = Statement<
  Pick<TrieNode, "id">,
  [Pick<TrieNode, "parent_id" | "char">]
>;
export type UpdateTrieTerminalStmt = Statement<void, [TrieNodeUpdate]>;
export type SelectTrieChildrenStmt = Statement<
  Pick<TrieNode, "char">,
  [{ parent_id: number | null }]
>;

export const createTrieTable = (database: Database) =>
  database.run(`
    CREATE TABLE IF NOT EXISTS trie_nodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      parent_id INTEGER,
      char TEXT NOT NULL,
      pattern TEXT,
      terminal INTEGER DEFAULT 0,
      markov_id INTEGER,
      FOREIGN KEY (parent_id) REFERENCES trie_nodes(id),
      FOREIGN KEY (markov_id) REFERENCES nodes(id)
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_trie_parent_char ON trie_nodes(parent_id, char);
    CREATE INDEX IF NOT EXISTS idx_trie_pattern ON trie_nodes(pattern);
  `);

export const createTrieStatements = (database: Database) => {
  const insertTrieNode: InsertTrieNodeStmt = database.query(`
    INSERT INTO trie_nodes (parent_id, char, terminal)
    VALUES ($parent_id, $char, $terminal)
    ON CONFLICT(parent_id, char)
    DO UPDATE SET terminal = terminal OR $terminal
    RETURNING id;
  `);

  const selectTrieNode: SelectTrieNodeStmt = database.query(`
    SELECT id FROM trie_nodes WHERE parent_id IS $parent_id AND char = $char;
  `);

  const updateTrieTerminal: UpdateTrieTerminalStmt = database.query(`
    UPDATE trie_nodes
    SET terminal = 1, pattern = $pattern, markov_id = $markov_id
    WHERE id = $id;
  `);

  const selectTrieChildren: SelectTrieChildrenStmt = database.query(`
    SELECT char FROM trie_nodes WHERE parent_id = $parent_id
  `);

  return {
    insertTrieNode,
    selectTrieNode,
    updateTrieTerminal,
    selectTrieChildren,
  };
};
