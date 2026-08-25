import OpenAI from "openai";
import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const DEFAULT_BASE_URL = "https://api.deepseek.com";
const FALLBACK_DEFAULT_MODEL = "deepseek-chat";

export interface DeepSeekStrategyOptions {
  apiKey?: string;
  defaultModel?: string;
  /** Override the DeepSeek API base URL. Default `https://api.deepseek.com`. */
  baseURL?: string;
}

/**
 * DeepSeek adapter. DeepSeek's API is OpenAI-compatible, so this reuses
 * the `openai` SDK with `baseURL` pointed at DeepSeek. DeepSeek has no
 * vision-capable endpoint — any attachment throws
 * `UnsupportedAttachmentError` up front rather than an opaque 4xx
 * mid-stream.
 */
export class DeepSeekStrategy implements LlmStrategy {
  readonly providerName = "deepseek";
  readonly capabilities = ["code", "reasoning", "cheap"] as const;
  readonly defaultModel: string;
  private platformClient: OpenAI | null = null;
  private readonly platformApiKey: string | undefined;
  private readonly baseURL: string;

  constructor(opts: DeepSeekStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
    this.baseURL = opts.baseURL?.trim() || DEFAULT_BASE_URL;
  }

  private getPlatformClient(): OpenAI {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "DeepSeek platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new OpenAI({ apiKey: this.platformApiKey, baseURL: this.baseURL });
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    const attachments = opts.attachments ?? [];
    if (attachments.length > 0) {
      throw new UnsupportedAttachmentError(
        this.providerName,
        attachments[0]!.mimetype,
        "DeepSeek has no vision-capable endpoint. Switch to a provider with vision support.",
      );
    }

    const client = opts.apiKey
      ? new OpenAI({ apiKey: opts.apiKey, baseURL: this.baseURL })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;

    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: opts.prompt },
        ]
      : [{ role: "user" as const, content: opts.prompt }];

    const response = await client.chat.completions.create({
      model: modelName,
      messages,
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
