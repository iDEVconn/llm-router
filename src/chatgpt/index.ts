import OpenAI from "openai";
import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const DEFAULT_BASE_URL = "https://api.openai.com/v1";
const FALLBACK_DEFAULT_MODEL = "gpt-4.1-mini";

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

export interface ChatGptStrategyOptions {
  apiKey?: string;
  defaultModel?: string;
  /** Override the OpenAI API base URL. Default `https://api.openai.com/v1`. */
  baseURL?: string;
}

function toBase64(data: string | Buffer): string {
  if (typeof data === "string") return data;
  return data.toString("base64");
}

/**
 * OpenAI ChatGPT adapter. Chat Completions' vision input only accepts
 * images, not PDFs — callers must convert PDFs client-side. Hitting the
 * adapter with a non-image MIME yields `UnsupportedAttachmentError` up
 * front rather than an opaque 4xx mid-stream.
 */
export class ChatGptStrategy implements LlmStrategy {
  readonly providerName = "chatgpt";
  readonly capabilities = ["code", "reasoning", "vision", "multilingual"] as const;
  readonly defaultModel: string;
  private platformClient: OpenAI | null = null;
  private readonly platformApiKey: string | undefined;
  private readonly baseURL: string;

  constructor(opts: ChatGptStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
    this.baseURL = opts.baseURL?.trim() || DEFAULT_BASE_URL;
  }

  private getPlatformClient(): OpenAI {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "ChatGPT platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new OpenAI({ apiKey: this.platformApiKey, baseURL: this.baseURL });
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    for (const attachment of opts.attachments ?? []) {
      if (!SUPPORTED_IMAGE_TYPES.has(attachment.mimetype)) {
        throw new UnsupportedAttachmentError(
          this.providerName,
          attachment.mimetype,
          "OpenAI Chat Completions vision only accepts image inputs. Convert the file to PNG or JPEG, or switch to a provider with PDF support.",
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

    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: messageContent },
        ]
      : [{ role: "user" as const, content: messageContent }];

    const response = await client.chat.completions.create({
      model: modelName,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: messages as any,
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
      truncated: response.choices[0]?.finish_reason === "length",
    };
  }

  /**
   * Cheapest auth-checked call against OpenAI. `models.list` is free and
   * account-wide — no token spend. `model` is accepted to satisfy
   * `LlmStrategy.validateKey` but ignored, same as the Grok adapter.
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
