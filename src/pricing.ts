import type { LlmUsage } from "./types";
import { toTokenCount } from "./usage-cache";

/**
 * Per-model rates in currency units per 1M tokens.
 *
 * `cacheWritePer1M` / `cacheReadPer1M` are optional: when a usage carries
 * cache counters and the rate is missing, those tokens are priced at
 * `inputPer1M`, which over-prices cache reads (a conservative estimate).
 * For Anthropic set both explicitly, per model
 * (https://platform.claude.com/docs/en/about-claude/pricing): 5-minute
 * cache write is 1.25x input; cache read is 0.1x input by default, 0.05x on
 * Opus 5.5 / Sonnet 5.5 and 0.025x on Fable / Mythos 5.1 and 5.2. For
 * OpenAI-compatible providers, DeepSeek and Gemini set `cacheReadPer1M` to
 * the provider's cached-input price.
 */
export interface ModelPricing {
  inputPer1M: number;
  outputPer1M: number;
  cacheWritePer1M?: number;
  cacheReadPer1M?: number;
}

/**
 * Prices are not baked in as a constant — they drift independently of
 * this package's release cycle, so the caller owns and passes the table.
 */
export type PricingTable = Record<
  string /* provider */,
  Record<string /* model */, ModelPricing>
>;

/** Disjoint input-side token counts of one usage. */
export interface InputTokenSplit {
  /** Input tokens billed at the normal input rate. */
  uncachedInputTokens: number;
  /** Input tokens written to the prompt cache. */
  cacheCreationInputTokens: number;
  /** Input tokens read from the prompt cache. */
  cacheReadInputTokens: number;
  /** Sum of the three above: the whole prompt as the model saw it. */
  totalInputTokens: number;
}

/**
 * Splits a usage's input side into disjoint uncached / cache-write /
 * cache-read counts, whatever the provider's convention
 * (`inputTokensIncludeCache`). Missing, NaN or negative cache counters
 * count as 0; with `inputTokensIncludeCache` the uncached part never goes
 * below 0.
 */
export function splitInputTokens(usage: LlmUsage): InputTokenSplit {
  const creation = toTokenCount(usage.cacheCreationInputTokens) ?? 0;
  const read = toTokenCount(usage.cacheReadInputTokens) ?? 0;
  if (usage.inputTokensIncludeCache) {
    const input = toTokenCount(usage.inputTokens) ?? 0;
    const uncached = Math.max(0, input - creation - read);
    return {
      uncachedInputTokens: uncached,
      cacheCreationInputTokens: creation,
      cacheReadInputTokens: read,
      totalInputTokens: uncached + creation + read,
    };
  }
  // `inputTokens` is used as given here, so a usage without cache counters
  // prices exactly as it did before cache counters existed.
  return {
    uncachedInputTokens: usage.inputTokens,
    cacheCreationInputTokens: creation,
    cacheReadInputTokens: read,
    totalInputTokens: usage.inputTokens + creation + read,
  };
}

/**
 * Cost of one call. A usage without cache counters is priced exactly as
 * before (input + output). Cache counters are priced at
 * `cacheWritePer1M` / `cacheReadPer1M` (falling back to `inputPer1M`), and
 * tokens already inside `inputTokens` (`inputTokensIncludeCache`) are not
 * counted twice.
 */
export function calculateCost(
  usage: LlmUsage,
  provider: string,
  model: string,
  pricing: PricingTable,
): number {
  const rates = pricing[provider]?.[model];
  if (!rates) {
    throw new Error(`No pricing entry for provider "${provider}", model "${model}".`);
  }
  const split = splitInputTokens(usage);
  return (
    (split.uncachedInputTokens / 1_000_000) * rates.inputPer1M +
    (split.cacheCreationInputTokens / 1_000_000) * (rates.cacheWritePer1M ?? rates.inputPer1M) +
    (split.cacheReadInputTokens / 1_000_000) * (rates.cacheReadPer1M ?? rates.inputPer1M) +
    (usage.outputTokens / 1_000_000) * rates.outputPer1M
  );
}
