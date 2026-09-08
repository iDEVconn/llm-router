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
  InvalidThinkingConfigError,
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
});
