import { GoogleGenerativeAI } from "@google/generative-ai";
import { LlmKeyValidationError } from "../errors";
import type {
  LlmAttachment,
  LlmGenerateOptions,
  LlmResponse,
  LlmStrategy,
} from "../types";

interface GeminiUsageMetadata {
  promptTokenCount?: number;
  candidatesTokenCount?: number;
}

export interface GeminiStrategyOptions {
  /** Platform API key (used when caller doesn't pass `apiKey` per-call). */
  apiKey?: string;
  /** Default model when caller doesn't pass `model` per-call. */
  defaultModel?: string;
}

const FALLBACK_DEFAULT_MODEL = "gemini-2.5-flash-lite";

function toBase64(data: string | Buffer): string {
  if (typeof data === "string") return data;
  return data.toString("base64");
}

/**
 * Google Gemini adapter. The platform key is optional so deployments may
 * run in BYOK-only mode — every call has to supply `apiKey` then. The
 * platform SDK client is lazy-instantiated on first use so a missing
 * platform key only fails calls that actually need it.
 */
export class GeminiStrategy implements LlmStrategy {
  readonly providerName = "gemini";
  readonly defaultModel: string;
  private platformClient: GoogleGenerativeAI | null = null;
  private readonly platformApiKey: string | undefined;

  constructor(opts: GeminiStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
  }

  private getPlatformClient(): GoogleGenerativeAI {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "Gemini platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new GoogleGenerativeAI(this.platformApiKey);
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    const client = opts.apiKey ? new GoogleGenerativeAI(opts.apiKey) : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;
    const model = client.getGenerativeModel({ model: modelName });

    const parts: Array<{ text: string } | { inlineData: { mimeType: string; data: string } }> = [
      { text: opts.prompt },
    ];
    for (const attachment of opts.attachments ?? []) {
      parts.push({
        inlineData: { mimeType: attachment.mimetype, data: toBase64(attachment.data) },
      });
    }

    const result = await model.generateContent(parts);

    // Token usage may live on `response.usageMetadata` or on the top-level
    // `result.usageMetadata` depending on the SDK version; check both.
    const usage =
      (result.response as { usageMetadata?: GeminiUsageMetadata }).usageMetadata ??
      (result as unknown as { usageMetadata?: GeminiUsageMetadata }).usageMetadata;

    return {
      text: result.response.text(),
      model: modelName,
      usage: {
        inputTokens: usage?.promptTokenCount ?? 0,
        outputTokens: usage?.candidatesTokenCount ?? 0,
      },
    };
  }

  /**
   * Lightweight key check. `countTokens` is one of the cheapest Gemini
   * calls — accepts the key + model and rejects fast on bad credentials,
   * without spending real generation budget.
   */
  async validateKey(apiKey: string, model?: string): Promise<void> {
    const client = new GoogleGenerativeAI(apiKey);
    try {
      await client
        .getGenerativeModel({ model: model?.trim() || this.defaultModel })
        .countTokens("validate");
    } catch (cause) {
      throw new LlmKeyValidationError(this.providerName, cause);
    }
  }

  // For tests + callers that want to know whether a platform key is wired.
  hasPlatformKey(): boolean {
    return this.platformApiKey !== undefined;
  }
}

export type { LlmAttachment };
