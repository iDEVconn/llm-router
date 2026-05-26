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
  attachments?: LlmAttachment[];
  /** Override the strategy's `defaultModel`. Blank/whitespace = use default. */
  model?: string;
  /** BYOK — bypass the strategy's platform key for this single call. */
  apiKey?: string;
  /** Maximum output tokens. Provider-specific defaults apply when omitted. */
  maxTokens?: number;
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
