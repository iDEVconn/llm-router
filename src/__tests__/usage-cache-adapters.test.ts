import { beforeEach, describe, expect, it, vi } from "vitest";

const mockMessagesCreate = vi.fn();
const mockMessagesStream = vi.fn();
const mockChatCompletionsCreate = vi.fn();
const mockGenerateContent = vi.fn();
const mockGenerateContentStream = vi.fn();
const mockVertexGenerateContent = vi.fn();
const mockVertexGenerateContentStream = vi.fn();

vi.mock("@anthropic-ai/sdk", () => {
  class Anthropic {
    public readonly messages = { create: mockMessagesCreate, stream: mockMessagesStream };
    constructor(public readonly opts: { apiKey: string }) {}
  }
  return { default: Anthropic };
});

vi.mock("openai", () => {
  class OpenAI {
    public readonly chat = { completions: { create: mockChatCompletionsCreate } };
    public readonly models = { list: vi.fn() };
    constructor(public readonly opts: { apiKey: string; baseURL?: string }) {}
  }
  return { default: OpenAI };
});

vi.mock("@google/generative-ai", () => {
  class GoogleGenerativeAI {
    constructor(public readonly key: string) {}
    getGenerativeModel() {
      return { generateContent: mockGenerateContent, generateContentStream: mockGenerateContentStream };
    }
  }
  return { GoogleGenerativeAI };
});

vi.mock("@google/genai", () => {
  class GoogleGenAI {
    models = {
      generateContent: mockVertexGenerateContent,
      generateContentStream: mockVertexGenerateContentStream,
    };
  }
  return { GoogleGenAI };
});

import { InvalidGenerateOptionsError, LlmAbortedError } from "../errors";
import { ChatGptStrategy } from "../chatgpt/index";
import { ClaudeStrategy } from "../claude/index";
import { DeepSeekStrategy } from "../deepseek/index";
import { GeminiStrategy } from "../gemini/index";
import { GrokStrategy } from "../grok/index";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function* asyncOf(items: any[]) {
  for (const item of items) yield item;
}

beforeEach(() => {
  vi.clearAllMocks();
});

describe("Claude cache usage", () => {
  const message = (usage: Record<string, unknown>) => ({
    content: [{ type: "text", text: "ok" }],
    model: "claude-haiku-4-5",
    stop_reason: "end_turn",
    usage,
  });

  it("non-stream: reports cache creation/read separately and keeps inputTokens uncached", async () => {
    mockMessagesCreate.mockResolvedValueOnce(
      message({ input_tokens: 12, output_tokens: 7, cache_creation_input_tokens: 2048, cache_read_input_tokens: 4096 }),
    );
    const result = await new ClaudeStrategy({ apiKey: "k" }).generate({ prompt: "p", systemPrompt: "s" });
    expect(result.usage).toEqual({
      inputTokens: 12,
      outputTokens: 7,
      cacheCreationInputTokens: 2048,
      cacheReadInputTokens: 4096,
    });
  });

  it("non-stream: reports zero cache counters as 0", async () => {
    mockMessagesCreate.mockResolvedValueOnce(
      message({ input_tokens: 12, output_tokens: 7, cache_creation_input_tokens: 0, cache_read_input_tokens: 0 }),
    );
    const result = await new ClaudeStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({
      inputTokens: 12,
      outputTokens: 7,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
    });
  });

  it("non-stream: omits the cache fields when they are null or absent", async () => {
    mockMessagesCreate.mockResolvedValueOnce(
      message({ input_tokens: 12, output_tokens: 7, cache_creation_input_tokens: null }),
    );
    const result = await new ClaudeStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({ inputTokens: 12, outputTokens: 7 });
    expect("cacheCreationInputTokens" in result.usage).toBe(false);
    expect("cacheReadInputTokens" in result.usage).toBe(false);
  });

  it("stream: final usage carries the cache counters", async () => {
    mockMessagesStream.mockReturnValueOnce({
      [Symbol.asyncIterator]: () =>
        asyncOf([{ type: "content_block_delta", delta: { type: "text_delta", text: "ok" } }]),
      finalMessage: () =>
        Promise.resolve(
          message({ input_tokens: 3, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 900 }),
        ),
    });
    const result = await new ClaudeStrategy({ apiKey: "k" }).generate({ prompt: "p", onToken: () => {} });
    expect(result.usage).toEqual({
      inputTokens: 3,
      outputTokens: 1,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 900,
    });
  });

  async function runAborted(events: unknown[]) {
    const controller = new AbortController();
    mockMessagesStream.mockReturnValueOnce({
      [Symbol.asyncIterator]: async function* () {
        for (const event of events) yield event;
        controller.abort();
        throw new Error("Request was aborted.");
      },
      finalMessage: () => new Promise(() => {}),
    });
    const err = await new ClaudeStrategy({ apiKey: "k" })
      .generate({ prompt: "12345678", onToken: () => {}, signal: controller.signal })
      .catch((e: unknown) => e);
    return err as LlmAbortedError;
  }

  it("aborted stream: carries the cache counters from message_start", async () => {
    const err = await runAborted([
      {
        type: "message_start",
        message: {
          usage: { input_tokens: 5, output_tokens: 1, cache_creation_input_tokens: 1500, cache_read_input_tokens: 0 },
        },
      },
      { type: "content_block_delta", delta: { type: "text_delta", text: "Hi" } },
      { type: "message_delta", delta: { stop_reason: null }, usage: { output_tokens: 4 } },
    ]);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err.usage).toEqual({
      inputTokens: 5,
      outputTokens: 4,
      cacheCreationInputTokens: 1500,
      cacheReadInputTokens: 0,
    });
    expect(err.usageEstimated).toBe(false);
  });

  it("aborted stream: a cumulative message_delta cache count overrides message_start", async () => {
    const err = await runAborted([
      {
        type: "message_start",
        message: { usage: { input_tokens: 5, output_tokens: 1, cache_read_input_tokens: 10 } },
      },
      {
        type: "message_delta",
        delta: { stop_reason: null },
        usage: { output_tokens: 4, cache_read_input_tokens: 20, cache_creation_input_tokens: null },
      },
    ]);
    expect(err.usage).toEqual({ inputTokens: 5, outputTokens: 4, cacheReadInputTokens: 20 });
  });

  it("aborted stream: no cache fields when message_start never arrived", async () => {
    const err = await runAborted([{ type: "content_block_delta", delta: { type: "text_delta", text: "abcd" } }]);
    expect(err.usage).toEqual({ inputTokens: 2, outputTokens: 1 });
    expect(err.usageEstimated).toBe(true);
  });
});

const openAiUsage = (cached?: number) => ({
  prompt_tokens: 5000,
  completion_tokens: 20,
  ...(cached === undefined ? {} : { prompt_tokens_details: { cached_tokens: cached } }),
});

describe.each([
  ["chatgpt", () => new ChatGptStrategy({ apiKey: "k" })],
  ["grok", () => new GrokStrategy({ apiKey: "k" })],
])("%s cache usage (OpenAI-compatible)", (_name, make) => {
  const completion = (usage: unknown) => ({
    choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    model: "m",
    usage,
  });

  it("non-stream: maps prompt_tokens_details.cached_tokens and flags it as included", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(completion(openAiUsage(4096)));
    const result = await make().generate({ prompt: "p" });
    expect(result.usage).toEqual({
      inputTokens: 5000,
      outputTokens: 20,
      cacheReadInputTokens: 4096,
      inputTokensIncludeCache: true,
    });
  });

  it("non-stream: reports cached_tokens 0 as 0", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(completion(openAiUsage(0)));
    const result = await make().generate({ prompt: "p" });
    expect(result.usage).toEqual({
      inputTokens: 5000,
      outputTokens: 20,
      cacheReadInputTokens: 0,
      inputTokensIncludeCache: true,
    });
  });

  it("non-stream: omits cache fields when no details are reported", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(completion(openAiUsage()));
    const result = await make().generate({ prompt: "p" });
    expect(result.usage).toEqual({ inputTokens: 5000, outputTokens: 20 });
  });

  it("stream: maps cached_tokens from the final usage chunk", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      asyncOf([
        { choices: [{ delta: { content: "ok" }, finish_reason: "stop" }], model: "m" },
        { choices: [], usage: openAiUsage(1024) },
      ]),
    );
    const result = await make().generate({ prompt: "p", onToken: () => {} });
    expect(result.usage).toEqual({
      inputTokens: 5000,
      outputTokens: 20,
      cacheReadInputTokens: 1024,
      inputTokensIncludeCache: true,
    });
  });

  it("aborted stream: carries cached_tokens when the usage chunk already arrived", async () => {
    const controller = new AbortController();
    mockChatCompletionsCreate.mockResolvedValueOnce(
      (async function* () {
        yield { choices: [{ delta: { content: "Hi" } }], model: "m" };
        yield { choices: [], usage: openAiUsage(64) };
        controller.abort();
      })(),
    );
    const err = (await make()
      .generate({ prompt: "p", onToken: () => {}, signal: controller.signal })
      .catch((e: unknown) => e)) as LlmAbortedError;
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err.usage).toEqual({
      inputTokens: 5000,
      outputTokens: 20,
      cacheReadInputTokens: 64,
      inputTokensIncludeCache: true,
    });
    expect(err.usageEstimated).toBe(false);
  });
});

describe("DeepSeek cache usage", () => {
  const completion = (usage: unknown) => ({
    choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
    model: "deepseek-chat",
    usage,
  });

  it("non-stream: maps prompt_cache_hit_tokens to cacheReadInputTokens (included in inputTokens)", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      completion({ prompt_tokens: 100, completion_tokens: 5, prompt_cache_hit_tokens: 64, prompt_cache_miss_tokens: 36 }),
    );
    const result = await new DeepSeekStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({
      inputTokens: 100,
      outputTokens: 5,
      cacheReadInputTokens: 64,
      inputTokensIncludeCache: true,
    });
  });

  it("non-stream: falls back to prompt_tokens_details.cached_tokens", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      completion({ prompt_tokens: 100, completion_tokens: 5, prompt_tokens_details: { cached_tokens: 0 } }),
    );
    const result = await new DeepSeekStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({
      inputTokens: 100,
      outputTokens: 5,
      cacheReadInputTokens: 0,
      inputTokensIncludeCache: true,
    });
  });

  it("non-stream: omits cache fields when none are reported", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(completion({ prompt_tokens: 100, completion_tokens: 5 }));
    const result = await new DeepSeekStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({ inputTokens: 100, outputTokens: 5 });
  });

  it("stream: maps prompt_cache_hit_tokens from the final usage chunk", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      asyncOf([
        { choices: [{ delta: { content: "ok" }, finish_reason: "stop" }], model: "deepseek-chat" },
        { choices: [], usage: { prompt_tokens: 100, completion_tokens: 5, prompt_cache_hit_tokens: 80 } },
      ]),
    );
    const result = await new DeepSeekStrategy({ apiKey: "k" }).generate({ prompt: "p", onToken: () => {} });
    expect(result.usage).toEqual({
      inputTokens: 100,
      outputTokens: 5,
      cacheReadInputTokens: 80,
      inputTokensIncludeCache: true,
    });
  });

  it("aborted stream: carries the cache hit count when the usage chunk already arrived", async () => {
    const controller = new AbortController();
    mockChatCompletionsCreate.mockResolvedValueOnce(
      (async function* () {
        yield { choices: [{ delta: { content: "Hi" } }] };
        yield { choices: [], usage: { prompt_tokens: 30, completion_tokens: 9, prompt_cache_hit_tokens: 16 } };
        controller.abort();
      })(),
    );
    const err = (await new DeepSeekStrategy({ apiKey: "k" })
      .generate({ prompt: "p", onToken: () => {}, signal: controller.signal })
      .catch((e: unknown) => e)) as LlmAbortedError;
    expect(err.usage).toEqual({
      inputTokens: 30,
      outputTokens: 9,
      cacheReadInputTokens: 16,
      inputTokensIncludeCache: true,
    });
  });
});

describe("Gemini cache usage", () => {
  it("direct non-stream: maps cachedContentTokenCount (included in promptTokenCount)", async () => {
    mockGenerateContent.mockResolvedValueOnce({
      response: {
        text: () => "ok",
        usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 5, cachedContentTokenCount: 32 },
      },
    });
    const result = await new GeminiStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({
      inputTokens: 50,
      outputTokens: 5,
      cacheReadInputTokens: 32,
      inputTokensIncludeCache: true,
    });
  });

  it("direct non-stream: omits cache fields when cachedContentTokenCount is absent", async () => {
    mockGenerateContent.mockResolvedValueOnce({
      response: { text: () => "ok", usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 5 } },
    });
    const result = await new GeminiStrategy({ apiKey: "k" }).generate({ prompt: "p" });
    expect(result.usage).toEqual({ inputTokens: 50, outputTokens: 5 });
  });

  it("direct stream: maps cachedContentTokenCount 0 as 0", async () => {
    mockGenerateContentStream.mockResolvedValueOnce({
      stream: asyncOf([{ text: () => "ok" }]),
      response: Promise.resolve({
        text: () => "ok",
        usageMetadata: { promptTokenCount: 50, candidatesTokenCount: 5, cachedContentTokenCount: 0 },
      }),
    });
    const result = await new GeminiStrategy({ apiKey: "k" }).generate({ prompt: "p", onToken: () => {} });
    expect(result.usage).toEqual({
      inputTokens: 50,
      outputTokens: 5,
      cacheReadInputTokens: 0,
      inputTokensIncludeCache: true,
    });
  });

  it("vertex non-stream and stream: map cachedContentTokenCount", async () => {
    const usageMetadata = { promptTokenCount: 50, candidatesTokenCount: 5, cachedContentTokenCount: 40 };
    mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok", usageMetadata });
    mockVertexGenerateContentStream.mockResolvedValueOnce(asyncOf([{ text: "ok", usageMetadata }]));
    const strategy = new GeminiStrategy({ connection: "vertex" });
    const expected = { inputTokens: 50, outputTokens: 5, cacheReadInputTokens: 40, inputTokensIncludeCache: true };
    expect((await strategy.generate({ prompt: "p" })).usage).toEqual(expected);
    expect((await strategy.generate({ prompt: "p", onToken: () => {} })).usage).toEqual(expected);
  });

  it("aborted direct stream: carries cachedContentTokenCount from the last chunk", async () => {
    const controller = new AbortController();
    mockGenerateContentStream.mockResolvedValueOnce({
      stream: (async function* () {
        yield {
          text: () => "Hi",
          usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 1, cachedContentTokenCount: 8 },
        };
        controller.abort();
        throw controller.signal.reason;
      })(),
      response: new Promise(() => {}),
    });
    const err = (await new GeminiStrategy({ apiKey: "k" })
      .generate({ prompt: "p", onToken: () => {}, signal: controller.signal })
      .catch((e: unknown) => e)) as LlmAbortedError;
    expect(err.usage).toEqual({
      inputTokens: 20,
      outputTokens: 1,
      cacheReadInputTokens: 8,
      inputTokensIncludeCache: true,
    });
  });
});

describe("requireApiKey", () => {
  it.each([
    ["claude", () => new ClaudeStrategy({ apiKey: "platform" })],
    ["chatgpt", () => new ChatGptStrategy({ apiKey: "platform" })],
    ["grok", () => new GrokStrategy({ apiKey: "platform" })],
    ["deepseek", () => new DeepSeekStrategy({ apiKey: "platform" })],
    ["gemini", () => new GeminiStrategy({ apiKey: "platform" })],
    ["gemini vertex", () => new GeminiStrategy({ connection: "vertex" })],
  ])("%s: refuses to fall back to the platform key, before any network call", async (_name, make) => {
    await expect(make().generate({ prompt: "p", apiKey: "", requireApiKey: true })).rejects.toBeInstanceOf(
      InvalidGenerateOptionsError,
    );
    expect(mockMessagesCreate).not.toHaveBeenCalled();
    expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    expect(mockGenerateContent).not.toHaveBeenCalled();
    expect(mockVertexGenerateContent).not.toHaveBeenCalled();
  });

  it("without requireApiKey an empty apiKey still falls back to the platform key (unchanged)", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "ok" }],
      model: "m",
      usage: { input_tokens: 1, output_tokens: 1 },
    });
    await expect(new ClaudeStrategy({ apiKey: "platform" }).generate({ prompt: "p", apiKey: "" })).resolves.toBeDefined();
  });
});
