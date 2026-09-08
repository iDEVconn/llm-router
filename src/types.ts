/**
 * A binary attachment to send alongside the prompt — typically an image
 * or PDF for vision-capable models. `data` may be a base64 string or a
 * raw Buffer; strategies handle both.
 */
export interface LlmAttachment {
  data: string | Buffer;
  mimetype: string;
}

/** Per-call options. */
export interface LlmGenerateOptions {
  prompt: string;
  /**
   * Stable instructions shared across many calls (e.g. a system/role
   * prompt). Kept separate from `prompt` so strategies that support
   * prompt caching (currently Claude, via `cache_control`) can cache it
   * independently of the per-call dynamic content, cutting the cost of
   * repeated calls that reuse the same instructions. Strategies without
   * caching support still use it correctly (as a system-role message or
   * `systemInstruction`) — they just don't get the cost benefit.
   */
  systemPrompt?: string;
  attachments?: LlmAttachment[];
  /** Override the strategy's `defaultModel`. Blank/whitespace = use default. */
  model?: string;
  /** BYOK — bypass the strategy's platform key for this single call. */
  apiKey?: string;
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

export interface LlmUsage {
  inputTokens: number;
  outputTokens: number;
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
