/**
 * A binary attachment to send alongside the prompt — typically an image
 * or PDF for vision-capable models. `data` may be a base64 string or a
 * raw Buffer; strategies handle both.
 */
export interface LlmAttachment {
  data: string | Buffer;
  mimetype: string;
}

/** One turn of multi-turn conversation history. See `LlmGenerateOptions.messages`. */
export interface LlmMessage {
  role: "user" | "assistant";
  content: string;
}

/** Per-call options. */
export interface LlmGenerateOptions {
  /**
   * Single-turn prompt text. Mutually exclusive with `messages` — for
   * multi-turn conversation history, use `messages` instead. Exactly one
   * of `prompt` / `messages` must be set; both or neither throws
   * `InvalidGenerateOptionsError`.
   */
  prompt?: string;
  /**
   * Full multi-turn conversation history. When provided, this replaces
   * `prompt` as the content sent to the provider — exactly one of
   * `prompt` / `messages` must be set, never both, never neither
   * (`InvalidGenerateOptionsError` otherwise). Roles must alternate
   * starting with "user" (provider-enforced, not validated here).
   * `attachments` apply only to the final turn.
   */
  messages?: LlmMessage[];
  /**
   * Stable instructions shared across many calls (e.g. a system/role
   * prompt). Kept separate from `prompt`/`messages` so strategies that
   * support prompt caching (currently Claude, via `cache_control`) can
   * cache it independently of the per-call dynamic content, cutting the
   * cost of repeated calls that reuse the same instructions. Strategies
   * without caching support still use it correctly (as a system-role
   * message or `systemInstruction`) — they just don't get the cost
   * benefit.
   */
  systemPrompt?: string;
  attachments?: LlmAttachment[];
  /** Override the strategy's `defaultModel`. Blank/whitespace = use default. */
  model?: string;
  /**
   * BYOK — bypass the strategy's platform key for this single call. An
   * empty or undefined value falls back to the platform key (unless
   * `requireApiKey` is true).
   */
  apiKey?: string;
  /**
   * When true, `generate()` throws `InvalidGenerateOptionsError` before any
   * network call if `apiKey` is missing or blank, instead of silently using
   * the strategy's platform key (or Vertex ADC). Use it on calls that must
   * be billed to the caller's own key. Default false (unchanged behaviour).
   */
  requireApiKey?: boolean;
  /** Maximum output tokens. Provider-specific defaults apply when omitted. */
  maxTokens?: number;
  /**
   * Called with each incremental text delta as it arrives. When provided,
   * a strategy that supports streaming uses the provider's streaming
   * endpoint internally instead of a single blocking call, but still
   * resolves the same `Promise<LlmResponse>` once the stream ends — this
   * is an additive side-channel, not an alternate return type. A strategy
   * MAY silently never invoke this (no streaming support) — callers must
   * never assume `onToken` fires at all; only the resolved `LlmResponse`
   * is guaranteed.
   */
  onToken?: (delta: string) => void;
  /**
   * Provider-agnostic reasoning-depth request. Not every provider/model
   * supports every variant — a strategy that receives a variant it cannot
   * honor MUST throw `UnsupportedThinkingModeError` (same philosophy as
   * `UnsupportedAttachmentError`: fail loudly up front, never silently
   * downgrade a caller's explicit request). Use `capabilities`
   * (`'thinking'` tag) to check support before requesting it, or catch
   * the typed error.
   *
   *  - `adaptive`: let the provider pick reasoning depth automatically
   *    (Anthropic `thinking.type=adaptive`; Gemini dynamic thinking
   *    budget `-1`).
   *  - `budget`: explicit token reservation for reasoning (Anthropic
   *    `thinking.type=enabled` + `budget_tokens`; Gemini
   *    `thinkingConfig.thinkingBudget`).
   *  - `effort`: coarse effort level, for providers/models that expose
   *    reasoning as an enum rather than a token count (OpenAI
   *    reasoning-capable models via `reasoning_effort`).
   */
  thinking?:
    | { type: "adaptive" }
    | { type: "budget"; tokens: number }
    | { type: "effort"; level: "low" | "medium" | "high" };
  /**
   * Aborts the call (and the underlying stream, if one is open). Without
   * this, a caller that disconnects leaks an open provider connection and
   * keeps burning tokens server-side with nowhere for the output to go.
   */
  signal?: AbortSignal;
}

/**
 * Token counts for one call.
 *
 * `inputTokens` keeps each provider's own meaning (unchanged since the
 * first release):
 *  - Claude: Anthropic's `input_tokens`, i.e. the UNCACHED input only.
 *    Cache writes and cache reads are NOT in it; they are reported in
 *    `cacheCreationInputTokens` / `cacheReadInputTokens` on top of it.
 *  - ChatGPT, Grok, DeepSeek, Gemini: the provider's full prompt count
 *    (`prompt_tokens` / `promptTokenCount`), which ALREADY INCLUDES the
 *    cached tokens; the adapters then set `inputTokensIncludeCache: true`.
 *
 * The cache fields are optional and additive: undefined when the provider
 * reported nothing, `0` when it reported zero, never NaN or negative.
 * `splitInputTokens()` turns any usage into disjoint
 * uncached / cache-write / cache-read counts, and `calculateCost()` prices
 * them, so callers do not need per-provider rules. By hand:
 *
 *  - `inputTokensIncludeCache` not true (Claude):
 *    cost = inputTokens*pIn + cacheCreationInputTokens*pWrite
 *         + cacheReadInputTokens*pRead + outputTokens*pOut
 *    (Anthropic: pWrite = 1.25*pIn for the 5-minute TTL, pRead = 0.1*pIn).
 *  - `inputTokensIncludeCache: true` (OpenAI-compatible, DeepSeek, Gemini):
 *    uncached = inputTokens - cacheReadInputTokens - (cacheCreationInputTokens ?? 0)
 *    cost = uncached*pIn + cacheReadInputTokens*pRead + outputTokens*pOut
 */
export interface LlmUsage {
  /** Provider-reported input tokens; see the interface doc for what each provider includes. */
  inputTokens: number;
  outputTokens: number;
  /**
   * Input tokens written to the provider's prompt cache on this call
   * (Claude: `cache_creation_input_tokens`, billed above the input rate).
   * Undefined when the provider reported none.
   */
  cacheCreationInputTokens?: number;
  /**
   * Input tokens served from the provider's prompt cache on this call
   * (Claude: `cache_read_input_tokens`; OpenAI/Grok:
   * `prompt_tokens_details.cached_tokens`; DeepSeek:
   * `prompt_cache_hit_tokens`; Gemini: `cachedContentTokenCount`).
   * Undefined when the provider reported none.
   */
  cacheReadInputTokens?: number;
  /**
   * True when `inputTokens` already contains the cache counts above
   * (OpenAI-compatible providers, DeepSeek, Gemini). Absent/false means the
   * cache counts are in addition to `inputTokens` (Claude). Only set when a
   * cache field is set.
   */
  inputTokensIncludeCache?: boolean;
}

export interface LlmResponse {
  /** Concatenated text body — JSON parsing / extraction is the caller's job. */
  text: string;
  /** The model the provider actually used (may differ from the request hint). */
  model: string;
  usage: LlmUsage;
  /**
   * True when the provider stopped generating because it hit the output
   * token limit, not because it finished naturally — `text` is very
   * likely incomplete (e.g. truncated JSON that will fail to parse).
   * Each strategy reads its own provider's exact signal for this
   * (Anthropic `stop_reason`, Gemini `finishReason`, OpenAI/xAI
   * `finish_reason`) rather than guessing from token counts.
   */
  truncated: boolean;
  /**
   * Reasoning/thinking trace text, if the provider returned one and
   * `thinking` was requested (or the model always includes one, e.g.
   * DeepSeek's `deepseek-reasoner` `reasoning_content`). Undefined when
   * not applicable — never an empty string standing in for "none."
   * SECURITY: treat this exactly like `text` — untrusted model output.
   * Never re-feed it into another prompt without running it through
   * `sanitizeUntrustedContent`/`detectPromptInjection` first.
   */
  thinking?: string;
}

/**
 * Strategy contract — one implementation per provider. The router stays
 * framework-agnostic, so strategies are plain classes / objects, not
 * NestJS providers. Wrap with `@Injectable()` at the consumer if needed.
 */
export interface LlmStrategy {
  /** Canonical provider id (e.g. `"gemini"`). Used in registry lookups. */
  readonly providerName: string;
  /** Model name used when neither env nor caller specifies one. */
  readonly defaultModel: string;

  /** Generate a completion. Returns text + token usage. */
  generate(opts: LlmGenerateOptions): Promise<LlmResponse>;

  /**
   * Live-check an API key with the cheapest call the provider supports.
   * Throws when the key is rejected. `model` lets the caller verify
   * access to a specific model; falls back to `defaultModel` when omitted.
   */
  validateKey(apiKey: string, model?: string): Promise<void>;

  /**
   * True when this strategy has a usable platform-level key configured
   * (constructor-supplied, not per-call BYOK). Optional for backward
   * compatibility — a strategy that omits this is treated by TaskRouter
   * as "unavailable," never assumed usable.
   */
  hasPlatformKey?(): boolean;

  /**
   * Free-text capability tags used by TaskRouter's rule-matching stage
   * (see `KNOWN_CAPABILITY_TAGS` in `task-router.ts`). Optional; a
   * strategy without tags never wins the rule stage and routes through
   * the LLM-fallback stage instead.
   */
  readonly capabilities?: readonly string[];
}

/** Curated metadata for a single model. Renderable in BYOK UI dropdowns. */
export interface LlmModelInfo {
  id: string;
  description: string;
}

/** Curated metadata for a single provider. */
export interface LlmProviderInfo<TName extends string = string> {
  name: TName;
  /** Where the user generates a BYOK key. Surface as a link in BYOK UI. */
  keyUrl: string;
  /** Ordered list of models — cheapest / fastest first so it doubles as the default. */
  models: readonly LlmModelInfo[];
  /** True when the provider's vision endpoint accepts PDF inputs directly. */
  acceptsPdf: boolean;
}
