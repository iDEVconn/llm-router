/**
 * Mirrors `LlmStrategy` in spirit — one implementation per embeddings
 * provider, kept out of core so heavy SDKs stay opt-in.
 */
export interface EmbeddingStrategy {
  readonly providerName: string;
  readonly dimensions: number;
  embed(texts: string[]): Promise<number[][]>;
}
