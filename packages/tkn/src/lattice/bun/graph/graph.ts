import { Database } from "bun:sqlite";
import type { IGraph } from "../../graph";
import {
  createGraphTables,
  createGraphStatements,
  type InsertNodeStmt,
  type SelectNodeIdStmt,
  type InsertEdgeStmt,
  type SelectTransitionsStmt,
  type SelectTopTokensStmt,
} from "./graph.db";
import { DegreeScorer, type IBunHubScorer } from "./scorers";

export class Graph implements IGraph {
  private db: Database;
  private scorer: IBunHubScorer;

  // Prepared statements
  private insertNode!: InsertNodeStmt;
  private selectNodeId!: SelectNodeIdStmt;
  private insertEdge!: InsertEdgeStmt;
  private selectTransitions!: SelectTransitionsStmt;
  private selectTopTokens!: SelectTopTokensStmt;

  constructor(database: Database, scorer: IBunHubScorer = new DegreeScorer()) {
    this.db = database;
    this.scorer = scorer;
    this.initSchema();
    this.prepareStatements();
  }

  private initSchema() {
    createGraphTables(this.db);
  }

  private prepareStatements() {
    const {
      insertNode,
      selectNodeId,
      insertEdge,
      selectTransitions,
      selectTopTokens,
    } = createGraphStatements(this.db);

    this.insertNode = insertNode;
    this.selectNodeId = selectNodeId;
    this.insertEdge = insertEdge;
    this.selectTransitions = selectTransitions;
    this.selectTopTokens = selectTopTokens;
  }

  /**
   * Gets or creates a Markov node for a token.
   * @param token - The token string
   * @returns The node id
   */
  getOrCreateNode(pattern: string): number {
    this.insertNode.run({ pattern });
    const row = this.selectNodeId.get({ pattern });
    if (!row)
      throw new Error(`Failed to get/create node for pattern: ${pattern}`);
    return row.id;
  }

  /**
   * Adds a transition between two tokens (creates nodes if needed).
   * @param from - Source token
   * @param to - Destination token
   */
  merge(from: string, to: string): { from_id: number; to_id: number } {
    const from_id = this.getOrCreateNode(from);
    const to_id = this.getOrCreateNode(to);
    this.insertEdge.run({ from_id, to_id });
    return { from_id, to_id };
  }

  /**
   * Bulk transition insertion for large sequences (transactional).
   * @param pairs - Array of [from, to] token pairs
   */
  mergeBatch(pairs: [string, string][]): { from_id: number; to_id: number }[] {
    const insertions: { from_id: number; to_id: number }[] = [];
    const tx = this.db.transaction(() => {
      for (let i = 0; i < pairs.length; i++)
        insertions.push(this.merge(pairs[i][0], pairs[i][1]));
    });
    tx();

    return insertions;
  }

  /**
   * Retrieves all outgoing transitions for a token.
   * @param from - The source token
   * @returns Array of transitions with weights
   */
  getNext(from: string): { to: string; weight: number }[] {
    return this.selectTransitions.all({ from });
  }

  /**
   * Returns top N tokens by hub score using the configured scoring algorithm.
   * @param limit - Number of tokens to return (default 10)
   * @returns Array of tokens with confidences
   */
  getTopTokens(limit = 10): { pattern: string; confidence: number }[] {
    this.scorer.compute(this.db);
    return this.selectTopTokens.all({ limit });
  }
}
