export interface VectorStoreEntry {
  id: string;
  vector: number[];
  metadata?: Record<string, unknown>;
}

export interface VectorStoreMatch {
  id: string;
  score: number;
  metadata?: Record<string, unknown>;
}

export interface VectorStore {
  upsert(entries: VectorStoreEntry[]): Promise<void>;
  query(
    vector: number[],
    topK: number,
    filter?: Record<string, unknown>,
  ): Promise<VectorStoreMatch[]>;
  delete(ids: string[]): Promise<void>;
}
