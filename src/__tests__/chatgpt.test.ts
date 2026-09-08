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
  InvalidThinkingConfigError,
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
});
