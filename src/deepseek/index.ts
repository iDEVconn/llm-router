import OpenAI from "openai";
import {
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const DEFAULT_BASE_URL = "https://api.deepseek.com";
const FALLBACK_DEFAULT_MODEL = "deepseek-chat";

interface DeepSeekDelta {
  content?: string | null;
  reasoning_content?: string | null;
}

interface DeepSeekStreamChunk {
  choices?: Array<{ delta?: DeepSeekDelta; finish_reason?: string | null }>;
  model?: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

interface DeepSeekCompletion {
  choices: Array<{
    message?: { content?: string | null; reasoning_content?: string | null };
    finish_reason?: string | null;
  }>;
  model: string;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
}

function shapeResponse(
  text: string,
  model: string,
  inputTokens: number,
  outputTokens: number,
  truncated: boolean,
  thinking: string | undefined,
): LlmResponse {
  return {
    text,
    model,
    usage: { inputTokens, outputTokens },
    truncated,
    ...(thinking !== undefined ? { thinking } : {}),
  };
}

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
  readonly capabilities = ["code", "reasoning", "cheap", "streaming"] as const;
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

    if (opts.thinking) {
      // deepseek-reasoner's reasoning is a model choice, not a
      // request-time toggle, per current verified understanding — no
      // request-side `thinking` param is confirmed to exist.
      throw new UnsupportedThinkingModeError(this.providerName, opts.thinking.type, []);
    }

    if (opts.signal?.aborted) {
      throw (opts.signal.reason as Error | undefined) ?? new Error("Aborted");
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

    const requestOptions = opts.signal ? { signal: opts.signal } : undefined;

    if (opts.onToken) {
      const stream = await client.chat.completions.create(
        {
          model: modelName,
          messages,
          stream: true,
          stream_options: { include_usage: true },
          ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
        },
        requestOptions,
      );

      let text = "";
      let reasoning = "";
      let model = modelName;
      let inputTokens = 0;
      let outputTokens = 0;
      let truncated = false;

      for await (const chunk of stream as AsyncIterable<DeepSeekStreamChunk>) {
        const choice = chunk.choices?.[0];
        const delta = choice?.delta;
        if (delta?.content) {
          text += delta.content;
          try {
            opts.onToken(delta.content);
          } catch (err) {
            console.warn("llm-router: deepseek onToken callback threw", err);
          }
        }
        if (delta?.reasoning_content) reasoning += delta.reasoning_content;
        if (choice?.finish_reason === "length") truncated = true;
        if (chunk.model) model = chunk.model;
        if (chunk.usage) {
          inputTokens = chunk.usage.prompt_tokens ?? 0;
          outputTokens = chunk.usage.completion_tokens ?? 0;
        }
      }

      return shapeResponse(text, model, inputTokens, outputTokens, truncated, reasoning || undefined);
    }

    const response: DeepSeekCompletion = await client.chat.completions.create(
      {
        model: modelName,
        messages,
        ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
      },
      requestOptions,
    );

    const raw = response.choices[0]?.message?.content ?? "";
    const text = typeof raw === "string" ? raw : "";
    const reasoning = response.choices[0]?.message?.reasoning_content ?? undefined;

    return shapeResponse(
      text,
      response.model,
      response.usage?.prompt_tokens ?? 0,
      response.usage?.completion_tokens ?? 0,
      response.choices[0]?.finish_reason === "length",
      reasoning ?? undefined,
    );
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
