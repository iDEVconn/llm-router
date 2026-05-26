import { beforeEach, describe, expect, it, vi } from "vitest";

const mockMessagesCreate = vi.fn();

vi.mock("@anthropic-ai/sdk", () => {
  class Anthropic {
    public readonly messages: { create: typeof mockMessagesCreate };
    constructor(public readonly opts: { apiKey: string }) {
      this.messages = { create: mockMessagesCreate };
    }
  }
  return { default: Anthropic };
});

import { LlmKeyValidationError } from "../errors";
import { ClaudeStrategy } from "../claude/index";

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
});
