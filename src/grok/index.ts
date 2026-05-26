import OpenAI from "openai";
import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
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

    const response = await client.chat.completions.create({
      model: modelName,
      // Cast through unknown to avoid a hard dep on OpenAI's deep message types.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: [{ role: "user", content: messageContent as any }],
      ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
    });

    const raw = response.choices[0]?.message?.content ?? "";
    const text = typeof raw === "string" ? raw : "";

    return {
      text,
      model: response.model,
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
      },
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
