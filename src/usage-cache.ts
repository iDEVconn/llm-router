import type { LlmUsage } from "./types";

/** A usable token count: a finite number >= 0, else undefined (null, NaN, negative, non-number). */
export function toTokenCount(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
}

/** Raw cache counters as a provider reported them (any of them may be missing or junk). */
export interface ReportedCacheUsage {
  cacheCreationInputTokens?: unknown;
  cacheReadInputTokens?: unknown;
  /** Whether the provider's input count already contains the cache counts. */
  inputTokensIncludeCache?: boolean;
}

/**
 * The optional cache part of an `LlmUsage`: only the counters that are
 * usable numbers, plus `inputTokensIncludeCache: true` when asked for and
 * at least one counter is present. Spread it into the usage object so
 * absent counters stay absent (not `undefined`-valued keys).
 */
export function cacheUsageFields(
  reported: ReportedCacheUsage,
): Pick<LlmUsage, "cacheCreationInputTokens" | "cacheReadInputTokens" | "inputTokensIncludeCache"> {
  const creation = toTokenCount(reported.cacheCreationInputTokens);
  const read = toTokenCount(reported.cacheReadInputTokens);
  const any = creation !== undefined || read !== undefined;
  return {
    ...(creation !== undefined ? { cacheCreationInputTokens: creation } : {}),
    ...(read !== undefined ? { cacheReadInputTokens: read } : {}),
    ...(any && reported.inputTokensIncludeCache ? { inputTokensIncludeCache: true } : {}),
  };
}

/** OpenAI-compatible usage shape (ChatGPT, Grok). */
export interface OpenAiCompatibleUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  prompt_tokens_details?: { cached_tokens?: number | null } | null;
}

/**
 * OpenAI Chat Completions `usage.prompt_tokens_details.cached_tokens`
 * (openai SDK `CompletionUsage.PromptTokensDetails`); xAI documents the
 * same field. `prompt_tokens` already contains these tokens.
 */
export function openAiCacheFields(usage: OpenAiCompatibleUsage | undefined): {
  cacheReadInputTokens?: number | null;
  inputTokensIncludeCache: true;
} {
  return {
    cacheReadInputTokens: usage?.prompt_tokens_details?.cached_tokens,
    inputTokensIncludeCache: true,
  };
}
