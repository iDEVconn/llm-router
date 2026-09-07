import { describe, expect, it, vi } from "vitest";
import { PgVectorStore } from "../embeddings/pgvector/index";

function makePool(queryImpl: (sql: string, params?: unknown[]) => unknown) {
  return { query: vi.fn().mockImplementation(queryImpl) };
}

describe("PgVectorStore", () => {
  it("upserts each entry with an INSERT ... ON CONFLICT statement", async () => {
    const query = vi.fn().mockResolvedValue({ rows: [] });
    const pool = { query };
    const store = new PgVectorStore({ pool: pool as never });

    await store.upsert([{ id: "a", vector: [1, 2, 3], metadata: { text: "hi" } }]);

    expect(query).toHaveBeenCalledTimes(1);
    const [sql, params] = query.mock.calls[0]!;
    expect(sql).toMatch(/INSERT INTO/i);
    expect(sql).toMatch(/ON CONFLICT/i);
    expect(params).toEqual(["a", "[1,2,3]", { text: "hi" }]);
  });

  it("queries by cosine distance and maps rows to VectorStoreMatch", async () => {
    const query = vi.fn().mockResolvedValue({
      rows: [{ id: "doc-1", score: 0.92, metadata: { text: "chunk" } }],
    });
    const pool = { query };
    const store = new PgVectorStore({ pool: pool as never });

    const results = await store.query([1, 0, 0], 5);

    expect(query).toHaveBeenCalledTimes(1);
    const [sql, params] = query.mock.calls[0]!;
    expect(sql).toMatch(/ORDER BY/i);
    expect(params).toEqual(["[1,0,0]", 5]);
    expect(results).toEqual([{ id: "doc-1", score: 0.92, metadata: { text: "chunk" } }]);
  });

  it("applies a metadata filter using parametrized values", async () => {
    const query = vi.fn().mockResolvedValue({ rows: [] });
    const pool = { query };
    const store = new PgVectorStore({ pool: pool as never });

    await store.query([1, 0], 3, { tenant: "acme" });

    const [sql, params] = query.mock.calls[0]!;
    expect(sql).toMatch(/WHERE/i);
    expect(params).toEqual(["[1,0]", 3, "acme"]);
  });

  it("rejects filter keys that are not simple identifiers", async () => {
    const pool = makePool(() => ({ rows: [] }));
    const store = new PgVectorStore({ pool: pool as never });

    await expect(store.query([1, 0], 3, { "tenant'; DROP TABLE x; --": "acme" })).rejects.toThrow(
      /invalid filter key/i,
    );
  });

  it("deletes by id using = ANY($1)", async () => {
    const query = vi.fn().mockResolvedValue({ rows: [] });
    const pool = { query };
    const store = new PgVectorStore({ pool: pool as never });

    await store.delete(["a", "b"]);

    expect(query).toHaveBeenCalledTimes(1);
    const [sql, params] = query.mock.calls[0]!;
    expect(sql).toMatch(/DELETE FROM/i);
    expect(sql).toMatch(/ANY/i);
    expect(params).toEqual([["a", "b"]]);
  });
});
