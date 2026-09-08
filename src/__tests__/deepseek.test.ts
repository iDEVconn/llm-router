import { beforeEach, describe, expect, it, vi } from "vitest";

const mockChatCompletionsCreate = vi.fn();
const mockModelsList = vi.fn();

vi.mock("openai", () => {
  class OpenAI {
    public readonly chat: { completions: { create: typeof mockChatCompletionsCreate } };
    public readonly models: { list: typeof mockModelsList };
    constructor(public readonly opts: { apiKey: string; baseURL?: string }) {
      this.chat = { completions: { create: mockChatCompletionsCreate } };
      this.models = { list: mockModelsList };
    }
  }
  return { default: OpenAI };
});

import { LlmKeyValidationError, UnsupportedAttachmentError, UnsupportedThinkingModeError } from "../errors";
import { DeepSeekStrategy } from "../deepseek/index";

function asyncIterableFrom<T>(items: T[]): AsyncIterable<T> {
  return {
    [Symbol.asyncIterator]() {
      let i = 0;
      return {
        next(): Promise<IteratorResult<T>> {
          if (i < items.length) return Promise.resolve({ done: false, value: items[i++]! });
          return Promise.resolve({ done: true, value: undefined as unknown as T });
        },
      };
    },
  };
}

describe("DeepSeekStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("declares its capability tags", () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["code", "reasoning", "cheap", "streaming"]);
  });

  it("sends a plain-text user message", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "answer" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "platform-key" });

    const result = await strategy.generate({ prompt: "write a function" });

    expect(result.text).toBe("answer");
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 2 });
    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.messages).toEqual([{ role: "user", content: "write a function" }]);
  });

  it("sends systemPrompt as a leading system message when provided", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "ok" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    await strategy.generate({ prompt: "p", systemPrompt: "Be concise." });

    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.messages).toEqual([
      { role: "system", content: "Be concise." },
      { role: "user", content: "p" },
    ]);
  });

  it("throws UnsupportedAttachmentError for ANY attachment (no vision endpoint)", async () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    await expect(
      strategy.generate({
        prompt: "p",
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      }),
    ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
  });

  it("reports truncated=true when finish_reason is length", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "cut" }, finish_reason: "length" }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 4, completion_tokens: 4096 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    const result = await strategy.generate({ prompt: "p" });
    expect(result.truncated).toBe(true);
  });

  it("uses the custom baseURL when supplied", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 0, completion_tokens: 0 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k", baseURL: "https://internal.deepseek.example" });
    await strategy.generate({ prompt: "p" });
    expect(mockChatCompletionsCreate).toHaveBeenCalledOnce();
  });

  it("validateKey calls models.list without spending tokens", async () => {
    mockModelsList.mockResolvedValueOnce({ data: [] });
    const strategy = new DeepSeekStrategy({});
    await strategy.validateKey("u-key");
    expect(mockModelsList).toHaveBeenCalledOnce();
  });

  it("validateKey wraps SDK rejections as LlmKeyValidationError", async () => {
    mockModelsList.mockRejectedValueOnce(new Error("unauthorized"));
    const strategy = new DeepSeekStrategy({});
    await expect(strategy.validateKey("bad")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new DeepSeekStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });

  it("hasPlatformKey reflects whether a constructor apiKey was given", () => {
    expect(new DeepSeekStrategy({ apiKey: "k" }).hasPlatformKey()).toBe(true);
    expect(new DeepSeekStrategy({}).hasPlatformKey()).toBe(false);
  });

  it("declares 'streaming' but not 'thinking' in capabilities", () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toContain("streaming");
    expect(strategy.capabilities).not.toContain("thinking");
  });

  it("streams deltas via onToken and resolves the final response", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      asyncIterableFrom([
        { choices: [{ delta: { content: "Hel" } }] },
        { choices: [{ delta: { content: "lo" } }] },
        {
          choices: [{ delta: {}, finish_reason: "stop" }],
          model: "deepseek-chat",
          usage: { prompt_tokens: 3, completion_tokens: 2 },
        },
      ]),
    );
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    const deltas: string[] = [];

    const result = await strategy.generate({ prompt: "p", onToken: (d) => deltas.push(d) });

    expect(deltas).toEqual(["Hel", "lo"]);
    expect(result.text).toBe("Hello");
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 2 });
    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.stream).toBe(true);
    expect(call.stream_options).toEqual({ include_usage: true });
  });

  it("swallows onToken callback errors without failing the call", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      asyncIterableFrom([
        { choices: [{ delta: { content: "hi" } }] },
        { choices: [{ delta: {}, finish_reason: "stop" }], model: "deepseek-chat", usage: { prompt_tokens: 1, completion_tokens: 1 } },
      ]),
    );
    const strategy = new DeepSeekStrategy({ apiKey: "k" });

    const result = await strategy.generate({
      prompt: "p",
      onToken: () => {
        throw new Error("boom");
      },
    });

    expect(result.text).toBe("hi");
  });

  it.each([
    { type: "adaptive" as const },
    { type: "budget" as const, tokens: 2048 },
    { type: "effort" as const, level: "high" as const },
  ])("throws UnsupportedThinkingModeError for thinking=%o, no network call made", async (thinking) => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    await expect(strategy.generate({ prompt: "p", thinking })).rejects.toBeInstanceOf(
      UnsupportedThinkingModeError,
    );
    expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
  });

  it("surfaces reasoning_content on LlmResponse.thinking even when opts.thinking was not set", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "answer", reasoning_content: "because..." } }],
      model: "deepseek-reasoner",
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "p" });

    expect(result.thinking).toBe("because...");
  });

  it("leaves LlmResponse.thinking undefined when reasoning_content is absent", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "answer" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "p" });

    expect(result.thinking).toBeUndefined();
  });

  it("accumulates streamed reasoning_content deltas into the final thinking field", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      asyncIterableFrom([
        { choices: [{ delta: { reasoning_content: "step1 " } }] },
        { choices: [{ delta: { content: "ans", reasoning_content: "step2" } }] },
        {
          choices: [{ delta: {}, finish_reason: "stop" }],
          model: "deepseek-reasoner",
          usage: { prompt_tokens: 3, completion_tokens: 2 },
        },
      ]),
    );
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    const deltas: string[] = [];

    const result = await strategy.generate({ prompt: "p", onToken: (d) => deltas.push(d) });

    expect(deltas).toEqual(["ans"]);
    expect(result.thinking).toBe("step1 step2");
    expect(result.text).toBe("ans");
  });

  it("rejects promptly on a pre-aborted signal without making a network call", async () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    const controller = new AbortController();
    controller.abort();

    await expect(strategy.generate({ prompt: "p", signal: controller.signal })).rejects.toThrow();
    expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
  });
});
