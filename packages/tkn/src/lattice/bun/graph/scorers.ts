import type { Database } from "bun:sqlite";
import type { IHubScorer } from "../../scorer";

/**
 * Bun-specific hub scorer interface.
 */
export interface IBunHubScorer extends IHubScorer<Database> {}

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
export class DegreeScorer implements IBunHubScorer {
  private stmt?: ReturnType<Database["prepare"]>;

  compute(db: Database): void {
    if (!this.stmt) {
      this.stmt = db.prepare(`
        UPDATE nodes
        SET hub_score = log(1 + COALESCE((
          SELECT SUM(weight) FROM edges WHERE edges.from_id = nodes.id
        ), 0))
      `);
    }
    this.stmt.run();
  }
}

/**
 * Computes weighted PageRank scores for nodes.
 *
 * PageRank assigns each node a score representing its global importance
 * within the graph based on the weight and structure of incoming links.
 *
 * This implementation uses a weighted transition model:
 *
 *    PR(to) = (1 - α)/N + α * Σ[ PR(from) * (w(from→to) / Σ(w(from→*)) ) ]
 *
 * where:
 *   - α (alpha) is the damping factor (typically 0.85),
 *   - N is the total number of nodes,
 *   - w(from→to) is the weight of the transition from `from` to `to`,
 *   - Σ(w(from→*)) is the total outgoing weight from node `from`.
 *
 * The algorithm iteratively updates PageRank values over `iters` steps.
 * After convergence, the scores are normalized so the highest-ranked node
 * has a hubScore of 1.
 *
 * @param iters - Number of iterations (default 15)
 * @param alpha - Damping factor (default 0.85)
 */
export class PageRankScorer implements IBunHubScorer {
  constructor(private iters = 15, private alpha = 0.85) {}

  compute(db: Database): void {
    // Initialize temp tables
    db.run("DROP TABLE IF EXISTS pr");
    db.run(`
      CREATE TEMP TABLE pr AS
      SELECT id, 1.0 / (SELECT COUNT(*) FROM nodes) AS score FROM nodes
    `);
    db.run("DROP TABLE IF EXISTS pr_next");
    db.run("CREATE TEMP TABLE pr_next (id INTEGER PRIMARY KEY, score REAL)");

    const countStmt = db.prepare("SELECT COUNT(*) as count FROM nodes");
    const result = countStmt.get() as { count: number } | undefined;
    const N = result?.count ?? 0;
    const base = (1 - this.alpha) / N;

    // Prepare iteration statements
    const deleteNextStmt = db.prepare("DELETE FROM pr_next");
    const insertNextStmt = db.prepare(`
      INSERT INTO pr_next
      SELECT n.id,
             :base + :alpha * COALESCE((
                 SELECT SUM(p.score * e.weight / (
                   SELECT SUM(weight) FROM edges WHERE from_id = e.from_id
                 ))
                 FROM edges e
                 JOIN pr p ON p.id = e.from_id
                 WHERE e.to_id = n.id
               ), 0)
      FROM nodes n
    `);
    const deletePrStmt = db.prepare("DELETE FROM pr");
    const insertPrStmt = db.prepare("INSERT INTO pr SELECT * FROM pr_next");

    // Run iterations
    for (let i = 0; i < this.iters; i++) {
      deleteNextStmt.run();
      insertNextStmt.run({ base, alpha: this.alpha });
      deletePrStmt.run();
      insertPrStmt.run();
    }

    // Finalize by normalizing scores
    db.run(`
      UPDATE nodes
      SET hub_score = (
        SELECT score / (SELECT MAX(score) FROM pr)
        FROM pr WHERE pr.id = nodes.id
      )
    `);
  }
}
