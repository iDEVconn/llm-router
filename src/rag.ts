import { sanitizeUntrustedContent } from "./injection-defense";
import type { EmbeddingStrategy } from "./embeddings/types";
import type { VectorStore, VectorStoreMatch } from "./embeddings/vector-store";

export interface RetrieverOptions {
  embeddingStrategy: EmbeddingStrategy;
  vectorStore: VectorStore;
}

export interface RetrieveOptions {
  topK?: number;
  filter?: Record<string, unknown>;
}

export interface RetrieveResult {
  chunks: string[];
  sources: VectorStoreMatch[];
}

const DEFAULT_TOP_K = 5;

/**
 * Retrieved content is a classic prompt-injection vector, so every chunk
 * is run through `sanitizeUntrustedContent` before it's handed back —
 * not optional, not a caller-provided flag.
 */
export class Retriever {
  constructor(private readonly opts: RetrieverOptions) {}

  async retrieve(query: string, opts: RetrieveOptions = {}): Promise<RetrieveResult> {
    const [queryVector] = await this.opts.embeddingStrategy.embed([query]);
    const sources = await this.opts.vectorStore.query(
      queryVector as number[],
      opts.topK ?? DEFAULT_TOP_K,
      opts.filter,
    );

    const chunks = sources.map((source) =>
      sanitizeUntrustedContent(String(source.metadata?.text ?? ""), {
        label: `source:${source.id}`,
      }),
    );

    return { chunks, sources };
  }
}
