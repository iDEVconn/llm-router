import type { Pool } from "pg";
import type { VectorStore, VectorStoreEntry, VectorStoreMatch } from "../vector-store";

export interface PgVectorStoreOptions {
  /** A `pg` `Pool` (or any object exposing a compatible `.query()`). */
  pool: Pool;
  /** Table name — trusted deploy-time config, not end-user input. */
  tableName?: string;
}

const IDENTIFIER_PATTERN = /^[a-zA-Z_][a-zA-Z0-9_]*$/;

function assertValidIdentifier(name: string, kind: string): void {
  if (!IDENTIFIER_PATTERN.test(name)) {
    throw new Error(`Invalid ${kind} "${name}": expected a simple identifier.`);
  }
}

/**
 * `VectorStore` backed by Postgres + pgvector. Covers Postgres directly and
 * Supabase in one adapter, since Supabase is Postgres with pgvector built
 * in. Requires the `pg` peer dependency and a table with
 * `id text primary key`, `embedding vector(n)`, and `metadata jsonb`
 * columns (create it and its pgvector index yourself — this adapter never
 * runs DDL).
 */
export class PgVectorStore implements VectorStore {
  private readonly pool: Pool;
  private readonly table: string;

  constructor(opts: PgVectorStoreOptions) {
    this.pool = opts.pool;
    this.table = opts.tableName ?? "llm_router_embeddings";
    assertValidIdentifier(this.table, "table name");
  }

  async upsert(entries: VectorStoreEntry[]): Promise<void> {
    for (const entry of entries) {
      await this.pool.query(
        `INSERT INTO ${this.table} (id, embedding, metadata) VALUES ($1, $2, $3)
         ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, metadata = EXCLUDED.metadata`,
        [entry.id, JSON.stringify(entry.vector), entry.metadata ?? {}],
      );
    }
  }

  async query(
    vector: number[],
    topK: number,
    filter?: Record<string, unknown>,
  ): Promise<VectorStoreMatch[]> {
    const params: unknown[] = [JSON.stringify(vector), topK];
    const clauses: string[] = [];

    for (const [key, value] of Object.entries(filter ?? {})) {
      assertValidIdentifier(key, "filter key");
      params.push(value);
      clauses.push(`metadata ->> '${key}' = $${params.length}`);
    }

    const where = clauses.length > 0 ? `WHERE ${clauses.join(" AND ")}` : "";
    const result = await this.pool.query(
      `SELECT id, metadata, 1 - (embedding <=> $1) AS score FROM ${this.table} ${where} ORDER BY embedding <=> $1 LIMIT $2`,
      params,
    );

    return result.rows.map((row: { id: string; score: number; metadata?: Record<string, unknown> }) => ({
      id: row.id,
      score: row.score,
      metadata: row.metadata,
    }));
  }

  async delete(ids: string[]): Promise<void> {
    await this.pool.query(`DELETE FROM ${this.table} WHERE id = ANY($1)`, [ids]);
  }
}
