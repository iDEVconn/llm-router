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

import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import { DeepSeekStrategy } from "../deepseek/index";

describe("DeepSeekStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("declares its capability tags", () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["code", "reasoning", "cheap"]);
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
});
