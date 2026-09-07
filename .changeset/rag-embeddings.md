---
"@idevconn/llm-router": minor
---

Add RAG support. `EmbeddingStrategy` (`src/embeddings/types.ts`) mirrors `LlmStrategy` for embeddings providers, and `VectorStore` (`src/embeddings/vector-store.ts`) is a provider-neutral upsert/query/delete interface. `Retriever` (`src/rag.ts`) ties them together: `retrieve(query, opts?)` embeds the query, queries the store, and returns `{ chunks, sources }` — every chunk is run through `sanitizeUntrustedContent` before it's returned, since retrieved content is a classic prompt-injection vector and this is not optional.

Ships one concrete adapter, `PgVectorStore`, via the new `@idevconn/llm-router/embeddings/pgvector` subpath export (same optional-peer-dependency pattern as the LLM adapters — declare `pg` yourself). It covers Postgres directly and Supabase in one shot, since Supabase is Postgres with pgvector built in; Mongo Atlas Vector Search and Vertex AI-backed stores need their own adapters later. It expects a table with `id text primary key`, `embedding vector(n)`, and `metadata jsonb` columns and never runs DDL. Filter keys are validated against a simple-identifier pattern before being interpolated into SQL (values are always parametrized) to prevent injection through metadata filter keys.
