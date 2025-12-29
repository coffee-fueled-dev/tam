import { Database, Statement } from "bun:sqlite";
import type {
  GraphNode,
  GraphNodeInsert,
  GraphEdgeInsert,
} from "./graph.model";

// Statement types
export type InsertNodeStmt = Statement<void, [GraphNodeInsert]>;
export type SelectNodeIdStmt = Statement<
  Pick<GraphNode, "id">,
  [Pick<GraphNode, "pattern">]
>;
export type InsertEdgeStmt = Statement<void, [GraphEdgeInsert]>;
export type SelectTransitionsStmt = Statement<
  { to: string; weight: number },
  [{ from: string }]
>;
export type SelectNodeCountStmt = Statement<{ count: number }, []>;
export type ComputeDegreeStmt = Statement<void, []>;
export type InitPageRankStmt = Statement<void, []>;
export type PageRankIterationStmt = Statement<
  void,
  [{ base: number; alpha: number }]
>;
export type FinalizePageRankStmt = Statement<void, []>;
export type SelectTopTokensStmt = Statement<
  { pattern: string; confidence: number },
  [{ limit: number }]
>;

export const createGraphTables = (database: Database) =>
  database.run(`
    CREATE TABLE IF NOT EXISTS nodes (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      token TEXT UNIQUE NOT NULL,
      hub_score REAL DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS edges (
      from_id INTEGER NOT NULL,
      to_id INTEGER NOT NULL,
      weight REAL DEFAULT 1,
      PRIMARY KEY (from_id, to_id)
    );
  `);

export const createGraphStatements = (database: Database) => {
  // Node and edge operations
  const insertNode: InsertNodeStmt = database.query(`
    INSERT INTO nodes (token)
    VALUES ($token)
    ON CONFLICT(token) DO NOTHING;
  `);

  const selectNodeId: SelectNodeIdStmt = database.query(
    `SELECT id FROM nodes WHERE token = $token;`
  );

  const insertEdge: InsertEdgeStmt = database.query(`
    INSERT INTO edges (from_id, to_id, weight)
    VALUES ($from_id, $to_id, 1)
    ON CONFLICT(from_id, to_id)
    DO UPDATE SET weight = weight + 1;
  `);

  const selectTransitions: SelectTransitionsStmt = database.query(`
    SELECT n.token AS to, e.weight
    FROM edges e
    JOIN nodes n ON n.id = e.to_id
    WHERE e.from_id = (SELECT id FROM nodes WHERE token = $from)
  `);

  const selectNodeCount: SelectNodeCountStmt = database.query(`
    SELECT COUNT(*) as count FROM nodes
  `);

  const selectTopTokens: SelectTopTokensStmt = database.query(`
    SELECT token AS pattern, hub_score AS confidence
    FROM nodes
    ORDER BY hub_score DESC
    LIMIT $limit
  `);

  // Hub score computation
  const computeDegree: ComputeDegreeStmt = database.query(`
    UPDATE nodes
    SET hub_score = log(1 + COALESCE((
      SELECT SUM(weight) FROM edges WHERE edges.from_id = nodes.id
    ), 0));
  `);

  const initPageRank: InitPageRankStmt = database.query(`
    DROP TABLE IF EXISTS pr;
    CREATE TEMP TABLE pr AS
    SELECT id, 1.0 / (SELECT COUNT(*) FROM nodes) AS score FROM nodes;
    DROP TABLE IF EXISTS pr_next;
    CREATE TEMP TABLE pr_next (id INTEGER PRIMARY KEY, score REAL);
  `);

  const pageRankIteration: PageRankIterationStmt = database.query(`
    DELETE FROM pr_next;
    INSERT INTO pr_next
    SELECT n.id,
           $base + $alpha * COALESCE((
               SELECT SUM(p.score * e.weight / (
                 SELECT SUM(weight) FROM edges WHERE from_id = e.from_id
               ))
               FROM edges e
               JOIN pr p ON p.id = e.from_id
               WHERE e.to_id = n.id
             ), 0)
    FROM nodes n;
    DELETE FROM pr;
    INSERT INTO pr SELECT * FROM pr_next;
  `);

  const finalizePageRank: FinalizePageRankStmt = database.query(`
    UPDATE nodes
    SET hub_score = (
      SELECT score / (SELECT MAX(score) FROM pr)
      FROM pr WHERE pr.id = nodes.id
    );
  `);

  return {
    insertNode,
    selectNodeId,
    insertEdge,
    selectTransitions,
    selectNodeCount,
    selectTopTokens,
    computeDegree,
    initPageRank,
    pageRankIteration,
    finalizePageRank,
  };
};
