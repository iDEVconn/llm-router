import OpenAI from "openai";
import {
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const DEFAULT_BASE_URL = "https://api.x.ai/v1";
const FALLBACK_DEFAULT_MODEL = "grok-4.3";

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

export interface GrokStrategyOptions {
  apiKey?: string;
  defaultModel?: string;
  /** Override the xAI API base URL. Default `https://api.x.ai/v1`. */
  baseURL?: string;
}

function toBase64(data: string | Buffer): string {
  if (typeof data === "string") return data;
  return data.toString("base64");
}

/**
 * xAI Grok adapter. xAI's REST API is OpenAI-compatible, so this reuses
 * the `openai` SDK with `baseURL` pointed at xAI. Grok's vision endpoint
 * does NOT accept PDF — callers must convert PDFs client-side. Hitting
 * the adapter with a non-image MIME yields `UnsupportedAttachmentError`
 * up front rather than an opaque 4xx mid-stream.
 */
export class GrokStrategy implements LlmStrategy {
  readonly providerName = "grok";
  readonly capabilities = ["vision", "cheap", "streaming"] as const;
  readonly defaultModel: string;
  private platformClient: OpenAI | null = null;
  private readonly platformApiKey: string | undefined;
  private readonly baseURL: string;

  constructor(opts: GrokStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
    this.baseURL = opts.baseURL?.trim() || DEFAULT_BASE_URL;
  }

  private getPlatformClient(): OpenAI {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "Grok platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new OpenAI({
        apiKey: this.platformApiKey,
        baseURL: this.baseURL,
      });
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    for (const attachment of opts.attachments ?? []) {
      if (!SUPPORTED_IMAGE_TYPES.has(attachment.mimetype)) {
        throw new UnsupportedAttachmentError(
          this.providerName,
          attachment.mimetype,
          "Grok vision only accepts image inputs. Convert the file to PNG or JPEG, or switch to a provider with PDF support.",
        );
      }
    }

    // xAI's reasoning-effort parameter contract could not be verified
    // against primary docs (docs.x.ai) — do not wire this up without
    // confirming the real parameter name/shape there first.
    if (opts.thinking) {
      throw new UnsupportedThinkingModeError(this.providerName, opts.thinking.type, []);
    }

    opts.signal?.throwIfAborted();

    const client = opts.apiKey
      ? new OpenAI({ apiKey: opts.apiKey, baseURL: this.baseURL })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;

    const messageContent: Array<
      | { type: "text"; text: string }
      | { type: "image_url"; image_url: { url: string; detail: "high" | "low" | "auto" } }
    > = [];

    for (const attachment of opts.attachments ?? []) {
      const data = toBase64(attachment.data);
      messageContent.push({
        type: "image_url",
        image_url: { url: `data:${attachment.mimetype};base64,${data}`, detail: "high" },
      });
    }
    messageContent.push({ type: "text", text: opts.prompt });

    // A leading system-role message is the correct OpenAI-wire-format shape
    // for stable instructions. It also positions the request to benefit
    // from any automatic prefix-based caching xAI's backend may apply
    // (OpenAI-compatible APIs commonly cache repeated prompt prefixes
    // transparently) — no explicit cache API is documented for xAI, so
    // this is a structural best-effort, not a guaranteed cost saving.
    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: messageContent },
        ]
      : [{ role: "user" as const, content: messageContent }];

    const body = {
      model: modelName,
      // Cast through unknown to avoid a hard dep on OpenAI's deep message types.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: messages as any,
      ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
    };

    if (opts.onToken) {
      return this.generateStreaming(client, body, opts.onToken, opts.signal);
    }

    const response = await client.chat.completions.create(body, { signal: opts.signal });
    return this.shapeResponse(response);
  }

  private async generateStreaming(
    client: OpenAI,
    body: Record<string, unknown>,
    onToken: (delta: string) => void,
    signal: AbortSignal | undefined,
  ): Promise<LlmResponse> {
    const stream = await client.chat.completions.create(
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      { ...body, stream: true, stream_options: { include_usage: true } } as any,
      { signal },
    );

    let text = "";
    let model: string | undefined;
    let finishReason: string | null | undefined;
    let usage: { prompt_tokens?: number; completion_tokens?: number } | undefined;

    for await (const chunk of stream as unknown as AsyncIterable<{
      choices?: Array<{ delta?: { content?: string | null }; finish_reason?: string | null }>;
      model?: string;
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    }>) {
      if (chunk.model) model = chunk.model;
      if (chunk.usage) usage = chunk.usage;
      const choice = chunk.choices?.[0];
      if (choice?.finish_reason) finishReason = choice.finish_reason;
      const delta = choice?.delta?.content;
      if (delta) {
        text += delta;
        try {
          onToken(delta);
        } catch (err) {
          console.warn("grok onToken callback threw; ignoring", err);
        }
      }
    }

    return this.shapeResponse({
      choices: [{ message: { content: text }, finish_reason: finishReason }],
      model: model ?? "",
      usage,
    });
  }

  private shapeResponse(raw: {
    choices?: Array<{
      message?: { content?: string | null };
      finish_reason?: string | null;
    }>;
    model: string;
    usage?: { prompt_tokens?: number; completion_tokens?: number };
  }): LlmResponse {
    const rawContent = raw.choices?.[0]?.message?.content ?? "";
    const text = typeof rawContent === "string" ? rawContent : "";

    return {
      text,
      model: raw.model,
      usage: {
        inputTokens: raw.usage?.prompt_tokens ?? 0,
        outputTokens: raw.usage?.completion_tokens ?? 0,
      },
      truncated: raw.choices?.[0]?.finish_reason === "length",
    };
  }

  /**
   * Cheapest auth-checked call against xAI. `models.list` is free and
   * provider-wide, which is exactly what's needed for key validation —
   * no token spend, no quota impact. The `model` parameter is accepted
   * to satisfy `LlmStrategy.validateKey` but ignored, because xAI has no
   * per-model auth gate beyond what `models.list` already verifies.
   */
  async validateKey(apiKey: string, _model?: string): Promise<void> {
    const client = new OpenAI({ apiKey, baseURL: this.baseURL });
    try {
      await client.models.list();
    } catch (cause) {
      throw new LlmKeyValidationError(this.providerName, cause);
    }
  }

  hasPlatformKey(): boolean {
    return this.platformApiKey !== undefined;
  }
}
