import { beforeEach, describe, expect, it, vi } from "vitest";

const mockMessagesCreate = vi.fn();
const mockMessagesStream = vi.fn();

vi.mock("@anthropic-ai/sdk", () => {
  class Anthropic {
    public readonly messages: { create: typeof mockMessagesCreate; stream: typeof mockMessagesStream };
    constructor(public readonly opts: { apiKey: string }) {
      this.messages = { create: mockMessagesCreate, stream: mockMessagesStream };
    }
  }
  return { default: Anthropic };
});

import {
  InvalidGenerateOptionsError,
  InvalidThinkingConfigError,
  LlmAbortedError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
import { ClaudeStrategy } from "../claude/index";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function fakeStream(events: any[], finalMessage: any) {
  return {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [Symbol.asyncIterator]: async function* () {
      for (const event of events) yield event;
    },
    finalMessage: () => Promise.resolve(finalMessage),
  };
}

describe("ClaudeStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends an image block for image/* attachments", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "hi" }],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 1, output_tokens: 2 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "platform-key" });

    await strategy.generate({
      prompt: "ok",
      attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
    });

    const call = mockMessagesCreate.mock.calls[0]![0];
    const content = call.messages[0].content;
    expect(content[0].type).toBe("image");
    expect(content[0].source.media_type).toBe("image/png");
    expect(content[1].type).toBe("text");
    expect(content[1].text).toBe("ok");
  });

  it("sends a document block for application/pdf attachments", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 0, output_tokens: 0 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    await strategy.generate({
      prompt: "p",
      attachments: [{ data: Buffer.from("pdf"), mimetype: "application/pdf" }],
    });

    const content = mockMessagesCreate.mock.calls[0]![0].messages[0].content;
    expect(content[0].type).toBe("document");
    expect(content[0].source.media_type).toBe("application/pdf");
  });

  it("reports truncated=true when stop_reason is max_tokens", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "cut off" }],
      model: "claude-haiku-4-5",
      stop_reason: "max_tokens",
      usage: { input_tokens: 4, output_tokens: 4096 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "p" });

    expect(result.truncated).toBe(true);
  });

  it("reports truncated=false for a normal end_turn completion", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "done" }],
      model: "claude-haiku-4-5",
      stop_reason: "end_turn",
      usage: { input_tokens: 4, output_tokens: 6 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "p" });

    expect(result.truncated).toBe(false);
  });

  it("concatenates text blocks from the response", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [
        { type: "text", text: "[" },
        { type: "text", text: "]" },
        { type: "tool_use" },
      ],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 4, output_tokens: 6 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "p" });

    expect(result.text).toBe("[]");
    expect(result.model).toBe("claude-haiku-4-5");
    expect(result.usage).toEqual({ inputTokens: 4, outputTokens: 6 });
  });

  it("sends systemPrompt as a cached system block when provided", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 0, output_tokens: 0 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    await strategy.generate({ prompt: "p", systemPrompt: "You are a helpful assistant." });

    const call = mockMessagesCreate.mock.calls[0]![0];
    expect(call.system).toEqual([
      {
        type: "text",
        text: "You are a helpful assistant.",
        cache_control: { type: "ephemeral" },
      },
    ]);
    // systemPrompt must not also leak into the user message content.
    expect(call.messages[0].content).toEqual([{ type: "text", text: "p" }]);
  });

  it("omits the system param entirely when systemPrompt is not provided", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 0, output_tokens: 0 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    await strategy.generate({ prompt: "p" });

    expect(mockMessagesCreate.mock.calls[0]![0].system).toBeUndefined();
  });

  it("honors maxTokens override", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 0, output_tokens: 0 },
    });
    const strategy = new ClaudeStrategy({ apiKey: "k" });

    await strategy.generate({ prompt: "p", maxTokens: 256 });

    expect(mockMessagesCreate.mock.calls[0]![0].max_tokens).toBe(256);
  });

  it("validateKey requests a 1-token completion", async () => {
    mockMessagesCreate.mockResolvedValueOnce({
      content: [{ type: "text", text: "ok" }],
      model: "claude-haiku-4-5",
      usage: { input_tokens: 0, output_tokens: 0 },
    });
    const strategy = new ClaudeStrategy({});

    await strategy.validateKey("u-key");

    const call = mockMessagesCreate.mock.calls[0]![0];
    expect(call.max_tokens).toBe(1);
    expect(call.messages[0].content).toBe("ok");
  });

  it("validateKey wraps rejections as LlmKeyValidationError", async () => {
    mockMessagesCreate.mockRejectedValueOnce(new Error("bad key"));
    const strategy = new ClaudeStrategy({});

    await expect(strategy.validateKey("u-key")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new ClaudeStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });

  it("declares its capability tags", () => {
    const strategy = new ClaudeStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual([
      "code",
      "reasoning",
      "long-context",
      "streaming",
      "thinking",
    ]);
  });

  describe("streaming", () => {
    it("uses messages.stream and forwards deltas in order when onToken is provided", async () => {
      const events = [
        { type: "content_block_delta", delta: { type: "text_delta", text: "Hel" } },
        { type: "content_block_delta", delta: { type: "text_delta", text: "lo" } },
      ];
      const finalMessage = {
        content: [{ type: "text", text: "Hello" }],
        model: "claude-haiku-4-5",
        stop_reason: "end_turn",
        usage: { input_tokens: 1, output_tokens: 2 },
      };
      mockMessagesStream.mockReturnValueOnce(fakeStream(events, finalMessage));
      const strategy = new ClaudeStrategy({ apiKey: "k" });
      const deltas: string[] = [];

      const result = await strategy.generate({
        prompt: "p",
        onToken: (delta) => deltas.push(delta),
      });

      expect(mockMessagesStream).toHaveBeenCalledTimes(1);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
      expect(deltas).toEqual(["Hel", "lo"]);
      expect(result.text).toBe("Hello");
      expect(result.usage).toEqual({ inputTokens: 1, outputTokens: 2 });
    });

    it("does not let a throwing onToken callback propagate out of generate()", async () => {
      const events = [{ type: "content_block_delta", delta: { type: "text_delta", text: "x" } }];
      const finalMessage = {
        content: [{ type: "text", text: "x" }],
        model: "claude-haiku-4-5",
        usage: { input_tokens: 1, output_tokens: 1 },
      };
      mockMessagesStream.mockReturnValueOnce(fakeStream(events, finalMessage));
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      const result = await strategy.generate({
        prompt: "p",
        onToken: () => {
          throw new Error("boom");
        },
      });

      expect(result.text).toBe("x");
    });
  });

  describe("thinking", () => {
    it("sends an adaptive thinking config", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [],
        model: "m",
        usage: { input_tokens: 0, output_tokens: 0 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await strategy.generate({ prompt: "p", thinking: { type: "adaptive" } });

      expect(mockMessagesCreate.mock.calls[0]![0].thinking).toEqual({ type: "adaptive" });
    });

    it("sends a budget thinking config as thinking.enabled + budget_tokens", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [],
        model: "m",
        usage: { input_tokens: 0, output_tokens: 0 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await strategy.generate({
        prompt: "p",
        thinking: { type: "budget", tokens: 2000 },
        maxTokens: 4096,
      });

      expect(mockMessagesCreate.mock.calls[0]![0].thinking).toEqual({
        type: "enabled",
        budget_tokens: 2000,
      });
    });

    it("throws InvalidThinkingConfigError when budget tokens < 1024, no network call", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({ prompt: "p", thinking: { type: "budget", tokens: 500 } }),
      ).rejects.toBeInstanceOf(InvalidThinkingConfigError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
      expect(mockMessagesStream).not.toHaveBeenCalled();
    });

    it("throws InvalidThinkingConfigError when budget tokens >= maxTokens, no network call", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({
          prompt: "p",
          thinking: { type: "budget", tokens: 4096 },
          maxTokens: 4096,
        }),
      ).rejects.toBeInstanceOf(InvalidThinkingConfigError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
    });

    it("throws UnsupportedThinkingModeError for effort, no network call", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({ prompt: "p", thinking: { type: "effort", level: "high" } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
    });

    it("extracts a thinking content block into LlmResponse.thinking", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [
          { type: "thinking", thinking: "reasoning..." },
          { type: "text", text: "answer" },
        ],
        model: "m",
        usage: { input_tokens: 1, output_tokens: 1 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      const result = await strategy.generate({ prompt: "p", thinking: { type: "adaptive" } });

      expect(result.thinking).toBe("reasoning...");
      expect(result.text).toBe("answer");
    });

    it("leaves thinking undefined when no thinking block is present", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [{ type: "text", text: "answer" }],
        model: "m",
        usage: { input_tokens: 0, output_tokens: 0 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      const result = await strategy.generate({ prompt: "p" });

      expect(result.thinking).toBeUndefined();
    });
  });

  describe("signal", () => {
    it("forwards signal to the SDK call and rejects promptly when pre-aborted", async () => {
      mockMessagesCreate.mockImplementationOnce(
        (_params: unknown, options?: { signal?: AbortSignal }) => {
          if (options?.signal?.aborted) return Promise.reject(new Error("aborted"));
          return Promise.resolve({ content: [], model: "m", usage: {} });
        },
      );
      const controller = new AbortController();
      controller.abort();
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await expect(
        strategy.generate({ prompt: "p", signal: controller.signal }),
      ).rejects.toThrow();
      expect(mockMessagesCreate).toHaveBeenCalledTimes(1);
    });
  });

  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [{ type: "text", text: "ok" }],
        model: "claude-haiku-4-5",
        usage: { input_tokens: 1, output_tokens: 1 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockMessagesCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "user", content: [{ type: "text", text: "first" }] },
        { role: "assistant", content: [{ type: "text", text: "second" }] },
        { role: "user", content: [{ type: "text", text: "third" }] },
      ]);
    });

    it("puts attachments on the last turn's content, not the first", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [],
        model: "claude-haiku-4-5",
        usage: { input_tokens: 0, output_tokens: 0 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      });

      const call = mockMessagesCreate.mock.calls[0]![0];
      expect(call.messages[0].content).toEqual([{ type: "text", text: "first" }]);
      expect(call.messages[1].content).toEqual([{ type: "text", text: "second" }]);
      expect(call.messages[2].content).toEqual([
        { type: "image", source: { type: "base64", media_type: "image/png", data: "aW1n" } },
        { type: "text", text: "third" },
      ]);
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
    });
  });
});

describe("ClaudeStrategy streaming abort", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  function abortingStream(events: any[], controller: AbortController, abortErr: unknown) {
    return {
      [Symbol.asyncIterator]: async function* () {
        for (const event of events) yield event;
        controller.abort();
        throw abortErr;
      },
      finalMessage: () => new Promise(() => {}),
    };
  }
  const textDelta = (text: string) => ({
    type: "content_block_delta",
    delta: { type: "text_delta", text },
  });
  const messageStart = (inputTokens: number) => ({
    type: "message_start",
    message: { usage: { input_tokens: inputTokens, output_tokens: 1 } },
  });
  const messageDelta = (outputTokens: number) => ({
    type: "message_delta",
    delta: { stop_reason: null },
    usage: { output_tokens: outputTokens },
  });

  async function runAborted(events: unknown[], extra: Partial<Parameters<ClaudeStrategy["generate"]>[0]> = {}) {
    const controller = new AbortController();
    // Mirrors the SDK: APIUserAbortError is a plain Error subclass (name "Error").
    const abortErr = new Error("Request was aborted.");
    mockMessagesStream.mockReturnValueOnce(abortingStream(events, controller, abortErr));
    const deltas: string[] = [];
    const err = await new ClaudeStrategy({ apiKey: "k" })
      .generate({
        prompt: "12345678",
        onToken: (d) => deltas.push(d),
        signal: controller.signal,
        ...extra,
      })
      .catch((e: unknown) => e);
    return { err: err as LlmAbortedError, deltas, abortErr };
  }

  it("rejects with LlmAbortedError carrying the delivered deltas and the original error", async () => {
    const { err, deltas, abortErr } = await runAborted([textDelta("Hel"), textDelta("lo")]);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("AbortError");
    expect(err.cause).toBe(abortErr);
    expect(err.partialText).toBe("Hello");
    expect(err.partialText).toBe(deltas.join(""));
    expect(err.partialThinking).toBeUndefined();
    expect(err.message).toBe("claude request aborted (5 chars received)");
  });

  it("uses provider-reported usage with usageEstimated=false when message_start and message_delta both arrived", async () => {
    const { err } = await runAborted([
      messageStart(42),
      textDelta("Hel"),
      textDelta("lo"),
      messageDelta(7),
    ]);
    expect(err.usage).toEqual({ inputTokens: 42, outputTokens: 7 });
    expect(err.usageEstimated).toBe(false);
  });

  it("estimates output (chars/4) with usageEstimated=true when only input was reported", async () => {
    const { err } = await runAborted([messageStart(42), textDelta("abcd"), textDelta("efgh")]);
    expect(err.usage).toEqual({ inputTokens: 42, outputTokens: 2 });
    expect(err.usageEstimated).toBe(true);
  });

  it("estimates both sides when nothing was reported (prompt + systemPrompt chars / 4)", async () => {
    const { err } = await runAborted([textDelta("abcde")], { systemPrompt: "abcd" });
    expect(err.usage).toEqual({ inputTokens: 3, outputTokens: 2 });
    expect(err.usageEstimated).toBe(true);
  });

  it("aborting before any delta gives partialText '' and usage estimated from the prompt", async () => {
    const { err } = await runAborted([]);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err.partialText).toBe("");
    expect(err.usage).toEqual({ inputTokens: 2, outputTokens: 0 });
    expect(err.usageEstimated).toBe(true);
  });

  it("keeps a delta whose onToken callback threw in partialText", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const { err } = await runAborted([textDelta("A"), textDelta("B")], {
      onToken: (d) => {
        if (d === "B") throw new Error("consumer bug");
      },
    });
    expect(err.partialText).toBe("AB");
    warn.mockRestore();
  });

  it("does not wrap a non-abort mid-stream error", async () => {
    const boom = new Error("overloaded");
    mockMessagesStream.mockReturnValueOnce({
      [Symbol.asyncIterator]: async function* () {
        yield textDelta("Hel");
        throw boom;
      },
      finalMessage: () => new Promise(() => {}),
    });
    const controller = new AbortController();
    await expect(
      new ClaudeStrategy({ apiKey: "k" }).generate({
        prompt: "p",
        onToken: () => {},
        signal: controller.signal,
      }),
    ).rejects.toBe(boom);
  });

  it("leaves a non-streaming abort unchanged", async () => {
    const controller = new AbortController();
    const abortErr = new Error("Request was aborted.");
    mockMessagesCreate.mockImplementationOnce(() => {
      controller.abort();
      return Promise.reject(abortErr);
    });
    await expect(
      new ClaudeStrategy({ apiKey: "k" }).generate({ prompt: "p", signal: controller.signal }),
    ).rejects.toBe(abortErr);
  });
});
