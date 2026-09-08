import { GoogleGenerativeAI } from "@google/generative-ai";
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
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

type GeminiPart = { text: string } | { inlineData: { mimeType: string; data: string } };

export type GeminiConnection = "gemini" | "vertex";

export interface GeminiStrategyOptions {
  /** Platform API key (used when caller doesn't pass `apiKey` per-call). */
  apiKey?: string;
  /** Default model when caller doesn't pass `model` per-call. */
  defaultModel?: string;
  /**
   * How platform-funded calls connect. `"vertex"` uses Vertex AI + ADC
   * (`GoogleGenAI({ vertexai: true })`) and does not require `apiKey`.
   * Per-call BYOK (`opts.apiKey`) always uses the direct Gemini API.
   */
  connection?: GeminiConnection;
}

const FALLBACK_DEFAULT_MODEL = "gemini-2.5-flash-lite";

// `@google/generative-ai` (the direct-API SDK) has no thinking support at
// all — no `thinkingConfig`/`thinkingBudget` anywhere in its types. Only
// the Vertex path (`@google/genai`) exposes it.
const BASE_CAPABILITIES = ["vision", "long-context", "multilingual", "cheap"] as const;

function toBase64(data: string | Buffer): string {
  if (typeof data === "string") return data;
  return data.toString("base64");
}

function buildParts(opts: LlmGenerateOptions): GeminiPart[] {
  const parts: GeminiPart[] = [{ text: opts.prompt }];
  for (const attachment of opts.attachments ?? []) {
    parts.push({
      inlineData: { mimeType: attachment.mimetype, data: toBase64(attachment.data) },
    });
  }
  return parts;
}

function forwardDelta(onToken: ((delta: string) => void) | undefined, delta: string): void {
  if (!onToken || !delta) return;
  try {
    onToken(delta);
  } catch (err) {
    console.warn("[gemini] onToken callback threw; ignoring", err);
  }
}

/** Direct-API-shaped response (blocking `result.response`, or the awaited streaming final response). */
function buildDirectResponse(
  response: {
    text: () => string;
    usageMetadata?: GeminiUsageMetadata;
    candidates?: Array<{ finishReason?: string }>;
  },
  modelName: string,
  topLevelUsage?: GeminiUsageMetadata,
): LlmResponse {
  const usage = response.usageMetadata ?? topLevelUsage;
  const finishReason = response.candidates?.[0]?.finishReason;
  return {
    text: response.text(),
    model: modelName,
    usage: {
      inputTokens: usage?.promptTokenCount ?? 0,
      outputTokens: usage?.candidatesTokenCount ?? 0,
    },
    truncated: finishReason === "MAX_TOKENS",
  };
}

/** `{type:'effort'}` isn't representable as a Gemini thinkingBudget — Vertex only supports adaptive/budget. */
function buildVertexThinkingConfig(
  thinking: NonNullable<LlmGenerateOptions["thinking"]>,
): { thinkingBudget: number; includeThoughts: true } {
  if (thinking.type === "adaptive") {
    return { thinkingBudget: -1, includeThoughts: true };
  }
  if (thinking.type === "budget") {
    if (thinking.tokens <= 0) {
      throw new InvalidThinkingConfigError("tokens must be > 0");
    }
    return { thinkingBudget: thinking.tokens, includeThoughts: true };
  }
  throw new UnsupportedThinkingModeError("gemini", thinking.type, ["adaptive", "budget"]);
}

interface VertexAccumulator {
  text: string;
  thinking: string;
  usage?: GeminiUsageMetadata;
  finishReason?: string;
}

function accumulateVertexChunk(
  acc: VertexAccumulator,
  chunk: {
    text?: string;
    candidates?: Array<{
      content?: { parts?: Array<{ text?: string; thought?: boolean }> };
      finishReason?: string;
    }>;
    usageMetadata?: GeminiUsageMetadata;
  },
  onToken: ((delta: string) => void) | undefined,
): void {
  const parts = chunk.candidates?.[0]?.content?.parts ?? [];
  if (parts.length > 0) {
    for (const part of parts) {
      if (!part.text) continue;
      if (part.thought) {
        acc.thinking += part.text;
      } else {
        acc.text += part.text;
        forwardDelta(onToken, part.text);
      }
    }
  } else if (typeof chunk.text === "string" && chunk.text) {
    // Some non-streaming responses expose a top-level `.text` string
    // instead of a `candidates[].content.parts` shape.
    acc.text += chunk.text;
    forwardDelta(onToken, chunk.text);
  }

  if (chunk.usageMetadata) acc.usage = chunk.usageMetadata;
  const finishReason = chunk.candidates?.[0]?.finishReason;
  if (finishReason) acc.finishReason = finishReason;
}

function finishVertexResult(acc: VertexAccumulator, modelName: string): LlmResponse {
  return {
    text: acc.text,
    model: modelName,
    usage: {
      inputTokens: acc.usage?.promptTokenCount ?? 0,
      outputTokens: acc.usage?.candidatesTokenCount ?? 0,
    },
    truncated: acc.finishReason === "MAX_TOKENS",
    ...(acc.thinking ? { thinking: acc.thinking } : {}),
  };
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
  private readonly connection: GeminiConnection;

  constructor(opts: GeminiStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
    this.connection = opts.connection === "vertex" ? "vertex" : "gemini";
  }

  /**
   * The direct-API SDK (`@google/generative-ai`) has no thinking support at
   * all, so only a `connection: "vertex"` instance advertises `'thinking'`.
   * Both connection modes support streaming.
   */
  get capabilities(): readonly string[] {
    return this.connection === "vertex"
      ? [...BASE_CAPABILITIES, "streaming", "thinking"]
      : [...BASE_CAPABILITIES, "streaming"];
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
    if (opts.signal?.aborted) {
      throw opts.signal.reason ?? new Error("gemini request aborted before it started");
    }

    const modelName = opts.model?.trim() || this.defaultModel;
    if (!opts.apiKey && this.connection === "vertex") {
      return this.generateViaVertex(opts, modelName);
    }
    return this.generateViaDirectApi(opts, modelName);
  }

  private async generateViaDirectApi(
    opts: LlmGenerateOptions,
    modelName: string,
  ): Promise<LlmResponse> {
    if (opts.thinking) {
      throw new UnsupportedThinkingModeError("gemini", opts.thinking.type, []);
    }

    const client = opts.apiKey ? new GoogleGenerativeAI(opts.apiKey) : this.getPlatformClient();
    // `systemInstruction` correctly separates stable instructions from the
    // per-call prompt — note this is NOT Gemini's Cached Content API (which
    // needs a separate create/lifecycle/TTL flow and a much higher minimum
    // token count than most system prompts reach), so it doesn't reduce
    // cost, only keeps the request shape correct.
    const model = client.getGenerativeModel({
      model: modelName,
      ...(opts.systemPrompt ? { systemInstruction: opts.systemPrompt } : {}),
    });

    const parts = buildParts(opts);
    const requestOptions = opts.signal ? { signal: opts.signal } : undefined;

    if (opts.onToken) {
      const result = await model.generateContentStream(parts, requestOptions);
      for await (const chunk of result.stream) {
        forwardDelta(opts.onToken, chunk.text());
      }
      const finalResponse = await result.response;
      return buildDirectResponse(finalResponse, modelName);
    }

    const result = await model.generateContent(parts, requestOptions);

    // Token usage may live on `response.usageMetadata` or on the top-level
    // `result.usageMetadata` depending on the SDK version; check both.
    const topLevelUsage = (result as unknown as { usageMetadata?: GeminiUsageMetadata })
      .usageMetadata;
    return buildDirectResponse(result.response, modelName, topLevelUsage);
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

  private async generateViaVertex(
    opts: LlmGenerateOptions,
    modelName: string,
  ): Promise<LlmResponse> {
    const { GoogleGenAI } = await import("@google/genai");
    const project = process.env.GOOGLE_CLOUD_PROJECT?.trim() || undefined;
    const location = process.env.GOOGLE_CLOUD_LOCATION?.trim() || undefined;
    const client = new GoogleGenAI({
      vertexai: true,
      ...(project ? { project } : {}),
      ...(location ? { location } : {}),
    });

    const parts = buildParts(opts);
    const contents = [{ role: "user", parts }];

    // Validate/build thinking config before any network call — a rejected
    // config must never reach the SDK.
    const thinkingConfig = opts.thinking ? buildVertexThinkingConfig(opts.thinking) : undefined;

    const config: Record<string, unknown> = {};
    if (opts.systemPrompt) config.systemInstruction = opts.systemPrompt;
    if (opts.signal) config.abortSignal = opts.signal;
    if (thinkingConfig) config.thinkingConfig = thinkingConfig;
    const hasConfig = Object.keys(config).length > 0;

    if (opts.onToken) {
      const stream = await client.models.generateContentStream({
        model: modelName,
        contents,
        ...(hasConfig ? { config } : {}),
      });
      const acc: VertexAccumulator = { text: "", thinking: "" };
      for await (const chunk of stream) {
        accumulateVertexChunk(acc, chunk, opts.onToken);
      }
      return finishVertexResult(acc, modelName);
    }

    const result = await client.models.generateContent({
      model: modelName,
      contents,
      ...(hasConfig ? { config } : {}),
    });
    const acc: VertexAccumulator = { text: "", thinking: "" };
    accumulateVertexChunk(acc, result, undefined);
    return finishVertexResult(acc, modelName);
  }

  // Vertex authenticates via ADC, so a missing Gemini API key is not "unwired".
  hasPlatformKey(): boolean {
    return this.connection === "vertex" || this.platformApiKey !== undefined;
  }
}

export type { LlmAttachment };
