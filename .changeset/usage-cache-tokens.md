---
"@idevconn/llm-router": minor
---

`LlmUsage` now reports prompt-cache tokens. New optional fields: `cacheCreationInputTokens`, `cacheReadInputTokens` and `inputTokensIncludeCache`. They are filled on normal, streaming and aborted-stream (`LlmAbortedError.usage`) results. Claude: `cache_creation_input_tokens` / `cache_read_input_tokens`, reported in addition to `inputTokens` (which stays Anthropic's uncached `input_tokens`). ChatGPT and Grok: `prompt_tokens_details.cached_tokens`. DeepSeek: `prompt_cache_hit_tokens`. Gemini: `cachedContentTokenCount`. For these four, `inputTokens` already includes the cached tokens, so `inputTokensIncludeCache: true` is set. The fields are omitted when the provider reports nothing. `inputTokens` keeps its old meaning for every provider.

New `splitInputTokens(usage)` returns disjoint uncached / cache-write / cache-read / total input counts for any provider. `PricingTable` entries accept optional `cacheWritePer1M` / `cacheReadPer1M`, and `calculateCost` / `withBudget` price the cache tokens at those rates, falling back to `inputPer1M`. A usage without cache fields costs exactly the same as before. Because Claude results now carry their cache tokens, Claude costs from `calculateCost` / `withBudget` now include tokens that were left out before; add the cache rates to your Claude pricing entries to price them exactly.

New optional `requireApiKey` on `LlmGenerateOptions`: when true, a missing or blank `apiKey` throws `InvalidGenerateOptionsError` before any network call instead of silently falling back to the platform key (or Vertex ADC). The default behaviour is unchanged.
