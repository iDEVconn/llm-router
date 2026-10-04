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
  InvalidThinkingConfigError,
  LlmAbortedError,
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import { ChatGptStrategy } from "../chatgpt/index";

async function* asyncChunks(chunks: unknown[]) {
  for (const chunk of chunks) yield chunk;
}

describe("ChatGptStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("declares its capability tags", () => {
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual([
      "code",
      "reasoning",
      "vision",
      "multilingual",
      "streaming",
      "thinking",
    ]);
  });

  it("sends an image_url message + prompt for image attachments", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "extracted" } }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "platform-key" });

    const result = await strategy.generate({
      prompt: "describe",
      attachments: [{ data: Buffer.from("img"), mimetype: "image/jpeg" }],
    });

    expect(result.text).toBe("extracted");
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 5 });

    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    const content = call.messages[0].content;
    expect(content[0].type).toBe("image_url");
    expect(content[0].image_url.url).toMatch(/^data:image\/jpeg;base64,/);
    expect(content[1].type).toBe("text");
    expect(content[1].text).toBe("describe");
  });

  it("throws UnsupportedAttachmentError on PDF inputs", async () => {
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    await expect(
      strategy.generate({
        prompt: "p",
        attachments: [{ data: Buffer.from("pdf"), mimetype: "application/pdf" }],
      }),
    ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
  });

  it("reports truncated=true when finish_reason is length", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "cut off" }, finish_reason: "length" }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 4, completion_tokens: 4096 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    const result = await strategy.generate({ prompt: "p" });
    expect(result.truncated).toBe(true);
  });

  it("sends systemPrompt as a leading system message when provided", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "ok" } }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    await strategy.generate({ prompt: "p", systemPrompt: "Be concise." });

    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.messages[0]).toEqual({ role: "system", content: "Be concise." });
    expect(call.messages[1].role).toBe("user");
  });

  it("uses the custom baseURL when supplied", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "" } }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 0, completion_tokens: 0 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "k", baseURL: "https://internal.openai.example/v1" });
    await strategy.generate({ prompt: "p" });
    expect(mockChatCompletionsCreate).toHaveBeenCalledOnce();
  });

  it("validateKey calls models.list without spending tokens", async () => {
    mockModelsList.mockResolvedValueOnce({ data: [] });
    const strategy = new ChatGptStrategy({});
    await strategy.validateKey("u-key");
    expect(mockModelsList).toHaveBeenCalledOnce();
  });

  it("validateKey wraps SDK rejections as LlmKeyValidationError", async () => {
    mockModelsList.mockRejectedValueOnce(new Error("unauthorized"));
    const strategy = new ChatGptStrategy({});
    await expect(strategy.validateKey("bad")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new ChatGptStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });

  it("hasPlatformKey reflects whether a constructor apiKey was given", () => {
    expect(new ChatGptStrategy({ apiKey: "k" }).hasPlatformKey()).toBe(true);
    expect(new ChatGptStrategy({}).hasPlatformKey()).toBe(false);
  });

  describe("streaming", () => {
    it("uses the streaming SDK method and forwards deltas in order when onToken is set", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce(
        asyncChunks([
          { choices: [{ delta: { content: "Hel" } }], model: "gpt-4.1-mini" },
          { choices: [{ delta: { content: "lo" } }], model: "gpt-4.1-mini" },
          {
            choices: [{ delta: {}, finish_reason: "stop" }],
            model: "gpt-4.1-mini",
            usage: { prompt_tokens: 3, completion_tokens: 2 },
          },
        ]),
      );
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      const deltas: string[] = [];

      const result = await strategy.generate({ prompt: "hi", onToken: (d) => deltas.push(d) });

      expect(deltas).toEqual(["Hel", "lo"]);
      expect(result.text).toBe("Hello");
      expect(result.model).toBe("gpt-4.1-mini");
      expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 2 });
      expect(result.truncated).toBe(false);

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.stream).toBe(true);
      expect(call.stream_options).toEqual({ include_usage: true });
    });

    it("reports truncated=true when the streamed finish_reason is length", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce(
        asyncChunks([
          { choices: [{ delta: { content: "x" } }], model: "gpt-4.1-mini" },
          {
            choices: [{ delta: {}, finish_reason: "length" }],
            model: "gpt-4.1-mini",
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          },
        ]),
      );
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      const result = await strategy.generate({ prompt: "hi", onToken: () => {} });

      expect(result.truncated).toBe(true);
    });

    it("swallows an onToken callback that throws and still resolves correctly", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce(
        asyncChunks([
          { choices: [{ delta: { content: "ok" } }], model: "gpt-4.1-mini" },
          {
            choices: [{ delta: {}, finish_reason: "stop" }],
            model: "gpt-4.1-mini",
            usage: { prompt_tokens: 1, completion_tokens: 1 },
          },
        ]),
      );
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

      const result = await strategy.generate({
        prompt: "hi",
        onToken: () => {
          throw new Error("callback boom");
        },
      });

      expect(result.text).toBe("ok");
      warnSpy.mockRestore();
    });

    it("does not use the streaming path when onToken is not provided", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "blocking" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      const result = await strategy.generate({ prompt: "hi" });

      expect(result.text).toBe("blocking");
      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.stream).toBeUndefined();
    });

    it("passes signal through to the SDK request options", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 0, completion_tokens: 0 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      const controller = new AbortController();

      await strategy.generate({ prompt: "hi", signal: controller.signal });

      const requestOptions = mockChatCompletionsCreate.mock.calls[0]![1];
      expect(requestOptions?.signal).toBe(controller.signal);
    });

    it("rejects promptly with no network call when signal is pre-aborted", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      const controller = new AbortController();
      controller.abort();

      await expect(strategy.generate({ prompt: "hi", signal: controller.signal })).rejects.toThrow();
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });
  });

  describe("thinking", () => {
    it("maps a valid effort level onto reasoning_effort", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 0, completion_tokens: 0 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await strategy.generate({ prompt: "hi", thinking: { type: "effort", level: "high" } });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.reasoning_effort).toBe("high");
    });

    it("throws InvalidThinkingConfigError for a level outside the allow-list, no network call", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({
          prompt: "hi",
          thinking: { type: "effort", level: "xhigh" as "low" },
        }),
      ).rejects.toBeInstanceOf(InvalidThinkingConfigError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws UnsupportedThinkingModeError for adaptive, no network call", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({ prompt: "hi", thinking: { type: "adaptive" } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws UnsupportedThinkingModeError for budget, no network call", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({ prompt: "hi", thinking: { type: "budget", tokens: 2000 } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("never populates LlmResponse.thinking", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 0, completion_tokens: 0 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      const result = await strategy.generate({ prompt: "hi" });

      expect(result.thinking).toBeUndefined();
    });
  });

  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

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
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [{ role: "user", content: "first" }],
        systemPrompt: "Be concise.",
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages[0]).toEqual({ role: "system", content: "Be concise." });
      expect(call.messages[1]).toEqual({ role: "user", content: "first" });
    });

    it("puts attachments on the last turn's content array, not the first", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/jpeg" }],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages[0]).toEqual({ role: "user", content: "first" });
      expect(call.messages[1]).toEqual({ role: "assistant", content: "second" });
      expect(call.messages[2].role).toBe("user");
      expect(call.messages[2].content[0].type).toBe("image_url");
      expect(call.messages[2].content[0].image_url.url).toMatch(/^data:image\/jpeg;base64,/);
      expect(call.messages[2].content[1]).toEqual({ type: "text", text: "third" });
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });
  });
});

describe("ChatGptStrategy streaming abort", () => {
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
    extra: Partial<Parameters<ChatGptStrategy["generate"]>[0]> = {},
    mode: "silent" | "throw" = "silent",
  ) {
    const controller = new AbortController();
    // Only used in "throw" mode; the openai SDK's APIUserAbortError is a plain Error (name "Error").
    const abortErr = new Error("Request was aborted.");
    mockChatCompletionsCreate.mockResolvedValueOnce(
      abortingChunks(chunks, controller, abortErr, mode),
    );
    const deltas: string[] = [];
    const err = await new ChatGptStrategy({ apiKey: "k" })
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
    expect(err.message).toBe("chatgpt request aborted (5 chars received)");
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
    const result = await new ChatGptStrategy({ apiKey: "k" }).generate({
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
    const result = await new ChatGptStrategy({ apiKey: "k" }).generate({
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
    const err = await new ChatGptStrategy({ apiKey: "k" })
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
      new ChatGptStrategy({ apiKey: "k" }).generate({
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
      new ChatGptStrategy({ apiKey: "k" }).generate({ prompt: "p", signal: controller.signal }),
    ).rejects.toBe(abortErr);
  });

  it("leaves the pre-start abort check unchanged for a streaming call (plain reason, no SDK call)", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(
      new ChatGptStrategy({ apiKey: "k" }).generate({
        prompt: "p",
        onToken: () => {},
        signal: controller.signal,
      }),
    ).rejects.toBe(controller.signal.reason);
    expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
  });
});
