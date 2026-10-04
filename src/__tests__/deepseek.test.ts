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

import {
  InvalidGenerateOptionsError,
  LlmAbortedError,
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
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

  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "deepseek-chat",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new DeepSeekStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "user", content: "first" },
        { role: "assistant", content: "second" },
        { role: "user", content: "third" },
      ]);
    });

    it("prepends the systemPrompt as a leading system message in multi-turn mode too", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "deepseek-chat",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new DeepSeekStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [{ role: "user", content: "first" }],
        systemPrompt: "Be concise.",
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "system", content: "Be concise." },
        { role: "user", content: "first" },
      ]);
    });

    it("still rejects attachments in multi-turn mode (no vision endpoint)", async () => {
      const strategy = new DeepSeekStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({
          messages: [{ role: "user", content: "first" }],
          attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
        }),
      ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new DeepSeekStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new DeepSeekStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });
  });
});

describe("DeepSeekStrategy streaming abort", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const chunk = (content: string) => ({ choices: [{ delta: { content } }] });

  /**
   * Behaves like the real openai SDK `Stream`: on abort it swallows the
   * error (core/streaming.js returns on `isAbortError(e) || signal.aborted`)
   * and the iteration just ends — no throw. `mode: "throw"` instead throws
   * `abortErr` after aborting (defensive path / other SDK versions).
   */
  async function* abortingChunks(
    chunks: unknown[],
    controller: AbortController,
    abortErr: unknown,
    mode: "silent" | "throw" = "silent",
  ) {
    for (const c of chunks) yield c;
    controller.abort();
    if (mode === "throw") throw abortErr;
  }

  async function runAborted(
    chunks: unknown[],
    extra: Partial<Parameters<DeepSeekStrategy["generate"]>[0]> = {},
    mode: "silent" | "throw" = "silent",
  ) {
    const controller = new AbortController();
    // Only used in "throw" mode; the openai SDK's APIUserAbortError is a plain Error (name "Error").
    const abortErr = new Error("Request was aborted.");
    mockChatCompletionsCreate.mockResolvedValueOnce(
      abortingChunks(chunks, controller, abortErr, mode),
    );
    const deltas: string[] = [];
    const err = await new DeepSeekStrategy({ apiKey: "k" })
      .generate({
        prompt: "12345678",
        systemPrompt: "abcd",
        onToken: (d) => deltas.push(d),
        signal: controller.signal,
        ...extra,
      })
      .catch((e: unknown) => e);
    return { err: err as LlmAbortedError, deltas, abortErr, reason: controller.signal.reason };
  }

  it("a silently-ended aborted stream (real SDK behaviour) rejects with LlmAbortedError: delivered deltas, estimated usage, signal.reason as cause", async () => {
    const { err, deltas, reason } = await runAborted([chunk("Hel"), chunk("lo")]);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("AbortError");
    expect(err.cause).toBe(reason);
    expect(err.partialText).toBe("Hello");
    expect(err.partialText).toBe(deltas.join(""));
    // Usage only arrives in the final chunk → always estimated mid-stream:
    // input (8 + 4 chars) / 4 = 3, output 5 chars / 4 → 2.
    expect(err.usage).toEqual({ inputTokens: 3, outputTokens: 2 });
    expect(err.usageEstimated).toBe(true);
    expect(err.message).toBe("deepseek request aborted (5 chars received)");
  });

  it("uses provider-reported usage when the final usage chunk already arrived", async () => {
    const { err } = await runAborted([
      chunk("Hi"),
      { choices: [], usage: { prompt_tokens: 30, completion_tokens: 9 } },
    ]);
    expect(err.usage).toEqual({ inputTokens: 30, outputTokens: 9 });
    expect(err.usageEstimated).toBe(false);
  });

  it("an abort that the SDK surfaces by throwing mid-stream is wrapped with that error as cause", async () => {
    const { err, abortErr } = await runAborted([chunk("Hel")], {}, "throw");
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err.cause).toBe(abortErr);
    expect(err.partialText).toBe("Hel");
  });

  it("an abort after finish_reason arrived resolves normally (the answer was complete)", async () => {
    const controller = new AbortController();
    mockChatCompletionsCreate.mockResolvedValueOnce(
      abortingChunks(
        [chunk("Hel"), { choices: [{ delta: { content: "lo" }, finish_reason: "stop" }] }],
        controller,
        undefined,
      ),
    );
    const result = await new DeepSeekStrategy({ apiKey: "k" }).generate({
      prompt: "p",
      onToken: () => {},
      signal: controller.signal,
    });
    expect(controller.signal.aborted).toBe(true);
    expect(result.text).toBe("Hello");
    expect(result.truncated).toBe(false);
  });

  it("a stream that ends without finish_reason and without an abort still resolves (unchanged)", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce(
      (async function* () {
        yield chunk("Hel");
      })(),
    );
    const result = await new DeepSeekStrategy({ apiKey: "k" }).generate({
      prompt: "p",
      onToken: () => {},
      signal: new AbortController().signal,
    });
    expect(result.text).toBe("Hel");
  });

  it("an abort while opening the stream gives partialText '' and usage estimated from the prompt", async () => {
    const controller = new AbortController();
    const abortErr = new Error("Request was aborted.");
    mockChatCompletionsCreate.mockImplementationOnce(() => {
      controller.abort();
      return Promise.reject(abortErr);
    });
    const err = await new DeepSeekStrategy({ apiKey: "k" })
      .generate({ prompt: "12345678", onToken: () => {}, signal: controller.signal })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect((err as LlmAbortedError).cause).toBe(abortErr);
    expect((err as LlmAbortedError).partialText).toBe("");
    expect((err as LlmAbortedError).usage).toEqual({ inputTokens: 2, outputTokens: 0 });
    expect((err as LlmAbortedError).usageEstimated).toBe(true);
  });

  it("keeps a delta whose onToken callback threw in partialText", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { err } = await runAborted([chunk("A"), chunk("B")], {
      onToken: (d) => {
        if (d === "B") throw new Error("consumer bug");
      },
    });
    expect(err.partialText).toBe("AB");
    warn.mockRestore();
  });

  it("does not wrap a non-abort mid-stream error", async () => {
    const boom = new Error("socket hang up");
    mockChatCompletionsCreate.mockResolvedValueOnce(
      (async function* () {
        yield chunk("Hel");
        throw boom;
      })(),
    );
    await expect(
      new DeepSeekStrategy({ apiKey: "k" }).generate({
        prompt: "p",
        onToken: () => {},
        signal: new AbortController().signal,
      }),
    ).rejects.toBe(boom);
  });

  it("leaves a non-streaming abort unchanged", async () => {
    const controller = new AbortController();
    const abortErr = new Error("Request was aborted.");
    mockChatCompletionsCreate.mockImplementationOnce(() => {
      controller.abort();
      return Promise.reject(abortErr);
    });
    await expect(
      new DeepSeekStrategy({ apiKey: "k" }).generate({ prompt: "p", signal: controller.signal }),
    ).rejects.toBe(abortErr);
  });

  it("leaves the pre-start abort check unchanged for a streaming call (plain reason, no SDK call)", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      new DeepSeekStrategy({ apiKey: "k" }).generate({
        prompt: "p",
        onToken: () => {},
        signal: controller.signal,
      }),
    ).rejects.toBe(controller.signal.reason);
    expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
  });

  it("carries streamed reasoning_content as partialThinking and counts it toward the output estimate", async () => {
    const { err } = await runAborted([
      { choices: [{ delta: { reasoning_content: "thinking" } }] },
      chunk("Hi"),
    ]);
    expect(err.partialText).toBe("Hi");
    expect(err.partialThinking).toBe("thinking");
    // output: ("Hi" + "thinking") = 10 chars / 4 → 3
    expect(err.usage).toEqual({ inputTokens: 3, outputTokens: 3 });
    expect(err.usageEstimated).toBe(true);
  });
});
