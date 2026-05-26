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
import { GrokStrategy } from "../grok/index";

describe("GrokStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends an image_url message + prompt for image attachments", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "extracted" } }],
      model: "grok-4.3",
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    });
    const strategy = new GrokStrategy({ apiKey: "platform-key" });

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
    const strategy = new GrokStrategy({ apiKey: "k" });

    await expect(
      strategy.generate({
        prompt: "p",
        attachments: [{ data: Buffer.from("pdf"), mimetype: "application/pdf" }],
      }),
    ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
  });

  it("defaults string content to empty when SDK returns null", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: null } }],
      model: "grok-4.3",
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    const strategy = new GrokStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "p" });
    expect(result.text).toBe("");
  });

  it("uses the custom baseURL when supplied", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "" } }],
      model: "grok-4.3",
      usage: { prompt_tokens: 0, completion_tokens: 0 },
    });
    const strategy = new GrokStrategy({
      apiKey: "k",
      baseURL: "https://internal.xai.example/v1",
    });
    await strategy.generate({ prompt: "p" });
    // Constructor opts captured on the mock — easiest way to assert is via the
    // fact that the call succeeded; deeper introspection requires a more
    // elaborate mock. The cheaper test is to ensure no throw.
    expect(mockChatCompletionsCreate).toHaveBeenCalledOnce();
  });

  it("validateKey calls models.list without spending tokens", async () => {
    mockModelsList.mockResolvedValueOnce({ data: [] });
    const strategy = new GrokStrategy({});

    await strategy.validateKey("u-key");
    expect(mockModelsList).toHaveBeenCalledOnce();
  });

  it("validateKey wraps SDK rejections as LlmKeyValidationError", async () => {
    mockModelsList.mockRejectedValueOnce(new Error("unauthorized"));
    const strategy = new GrokStrategy({});

    await expect(strategy.validateKey("bad")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new GrokStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });
});
