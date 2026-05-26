import Anthropic from "@anthropic-ai/sdk";
import { LlmKeyValidationError } from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

const FALLBACK_DEFAULT_MODEL = "claude-haiku-4-5";
const DEFAULT_MAX_TOKENS = 4096;
const VALIDATE_MAX_TOKENS = 1;

export interface ClaudeStrategyOptions {
  apiKey?: string;
  defaultModel?: string;
}

function toBase64(data: string | Buffer): string {
  if (typeof data === "string") return data;
  return data.toString("base64");
}

/**
 * Anthropic Claude adapter. Routes images vs PDFs to the correct content
 * block type (`image` for jpeg/png/gif/webp, `document` for PDF) because
 * Anthropic's Messages API requires the distinction up front. Anything
 * else is forwarded to Claude as an image block and Claude will 400 if it
 * doesn't recognize the media type — we mirror that surface rather than
 * second-guess the SDK.
 */
export class ClaudeStrategy implements LlmStrategy {
  readonly providerName = "claude";
  readonly defaultModel: string;
  private platformClient: Anthropic | null = null;
  private readonly platformApiKey: string | undefined;

  constructor(opts: ClaudeStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
  }

  private getPlatformClient(): Anthropic {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "Claude platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new Anthropic({ apiKey: this.platformApiKey });
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    const client = opts.apiKey
      ? new Anthropic({ apiKey: opts.apiKey })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;

    type ContentBlock =
      | { type: "text"; text: string }
      | {
          type: "image";
          source: { type: "base64"; media_type: string; data: string };
        }
      | {
          type: "document";
          source: { type: "base64"; media_type: "application/pdf"; data: string };
        };

    const content: ContentBlock[] = [];

    for (const attachment of opts.attachments ?? []) {
      const data = toBase64(attachment.data);
      if (SUPPORTED_IMAGE_TYPES.has(attachment.mimetype)) {
        content.push({
          type: "image",
          source: { type: "base64", media_type: attachment.mimetype, data },
        });
      } else {
        content.push({
          type: "document",
          source: { type: "base64", media_type: "application/pdf", data },
        });
      }
    }

    content.push({ type: "text", text: opts.prompt });

    const response = await client.messages.create({
      model: modelName,
      max_tokens: opts.maxTokens ?? DEFAULT_MAX_TOKENS,
      // Anthropic's SDK types accept the broader union; cast here so the
      // pkg compiles without pulling in the entire Anthropic.Messages
      // type surface as a public dep.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: [{ role: "user", content: content as any }],
    });

    const text = response.content
      .filter((block): block is Anthropic.TextBlock => block.type === "text")
      .map((block) => block.text)
      .join("");

    return {
      text,
      model: response.model,
      usage: {
        inputTokens: response.usage.input_tokens ?? 0,
        outputTokens: response.usage.output_tokens ?? 0,
      },
    };
  }

  /**
   * Live key check via a minimal 1-token `messages.create`. Anthropic
   * doesn't expose a free ping endpoint, so the canonical pattern is to
   * spend a single token. Cost ≈ $0.
   */
  async validateKey(apiKey: string, model?: string): Promise<void> {
    const client = new Anthropic({ apiKey });
    try {
      await client.messages.create({
        model: model?.trim() || this.defaultModel,
        max_tokens: VALIDATE_MAX_TOKENS,
        messages: [{ role: "user", content: "ok" }],
      });
    } catch (cause) {
      throw new LlmKeyValidationError(this.providerName, cause);
    }
  }

  hasPlatformKey(): boolean {
    return this.platformApiKey !== undefined;
  }
}
