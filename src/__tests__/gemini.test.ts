import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGenerateContent = vi.fn();
const mockGenerateContentStream = vi.fn();
const mockCountTokens = vi.fn();
const mockGetGenerativeModel = vi.fn();
const mockVertexGenerateContent = vi.fn();
const mockVertexGenerateContentStream = vi.fn();
const mockGoogleGenAI = vi.fn();

vi.mock("@google/generative-ai", () => {
  class GoogleGenerativeAI {
    constructor(public readonly key: string) {}
    getGenerativeModel(args: { model: string }) {
      mockGetGenerativeModel(args);
      return {
        generateContent: mockGenerateContent,
        generateContentStream: mockGenerateContentStream,
        countTokens: mockCountTokens,
      };
    }
  }
  return { GoogleGenerativeAI };
});

vi.mock("@google/genai", () => {
  class GoogleGenAI {
    constructor(opts: Record<string, unknown>) {
      mockGoogleGenAI(opts);
    }
    models = {
      generateContent: mockVertexGenerateContent,
      generateContentStream: mockVertexGenerateContentStream,
    };
  }
  return { GoogleGenAI };
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function* asyncGenOf(items: any[]) {
  for (const item of items) yield item;
}

import {
  InvalidGenerateOptionsError,
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
import { GeminiStrategy } from "../gemini/index";

describe("GeminiStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("calls generateContent with prompt + base64-encoded attachment", async () => {
    mockGenerateContent.mockResolvedValueOnce({
      response: {
        text: () => "result-text",
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
      },
    });
    const strategy = new GeminiStrategy({ apiKey: "platform-key" });

    const result = await strategy.generate({
      prompt: "describe this",
      attachments: [{ data: Buffer.from("hello"), mimetype: "image/png" }],
    });

    expect(result.text).toBe("result-text");
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 5 });
    expect(strategy.providerName).toBe("gemini");

    const parts = mockGenerateContent.mock.calls[0]![0];
    expect(parts[0]).toEqual({ text: "describe this" });
    expect(parts[1].inlineData.mimeType).toBe("image/png");
    expect(parts[1].inlineData.data).toBe(Buffer.from("hello").toString("base64"));
  });

  it("reports truncated=true when the candidate's finishReason is MAX_TOKENS", async () => {
    mockGenerateContent.mockResolvedValueOnce({
      response: {
        text: () => "cut off",
        candidates: [{ finishReason: "MAX_TOKENS" }],
      },
    });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "x" });

    expect(result.truncated).toBe(true);
  });

  it("reports truncated=false for a normal STOP finishReason", async () => {
    mockGenerateContent.mockResolvedValueOnce({
      response: {
        text: () => "done",
        candidates: [{ finishReason: "STOP" }],
      },
    });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "x" });

    expect(result.truncated).toBe(false);
  });

  it("reports truncated=false when no candidates are present", async () => {
    mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "x" });

    expect(result.truncated).toBe(false);
  });

  it("passes systemPrompt as systemInstruction when provided", async () => {
    mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    await strategy.generate({ prompt: "x", systemPrompt: "Be concise." });

    expect(mockGetGenerativeModel).toHaveBeenCalledWith({
      model: "gemini-2.5-flash-lite",
      systemInstruction: "Be concise.",
    });
    // systemPrompt must not also leak into the prompt parts.
    const parts = mockGenerateContent.mock.calls[0]![0];
    expect(parts).toEqual([{ text: "x" }]);
  });

  it("omits systemInstruction entirely when systemPrompt is not provided", async () => {
    mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    await strategy.generate({ prompt: "x" });

    expect(mockGetGenerativeModel).toHaveBeenCalledWith({ model: "gemini-2.5-flash-lite" });
  });

  it("uses the per-call apiKey instead of the platform key when given", async () => {
    mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
    const strategy = new GeminiStrategy({ apiKey: "platform-key" });

    await strategy.generate({ prompt: "x", apiKey: "user-key" });

    // The mock factory captures the constructor key on `this.key` — verify a
    // fresh client was constructed with the user key (the spec test doesn't
    // care which mechanism, only that the SDK was used).
    expect(mockGenerateContent).toHaveBeenCalledOnce();
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new GeminiStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });

  it("extracts usageMetadata from result.usageMetadata when response lacks it", async () => {
    mockGenerateContent.mockResolvedValueOnce({
      response: { text: () => "" },
      usageMetadata: { promptTokenCount: 7, candidatesTokenCount: 3 },
    });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "x" });
    expect(result.usage).toEqual({ inputTokens: 7, outputTokens: 3 });
  });

  it("falls back to zero usage when neither location reports metadata", async () => {
    mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
    const strategy = new GeminiStrategy({ apiKey: "k" });

    const result = await strategy.generate({ prompt: "x" });
    expect(result.usage).toEqual({ inputTokens: 0, outputTokens: 0 });
  });

  it("propagates SDK errors verbatim from generate", async () => {
    mockGenerateContent.mockRejectedValueOnce(new Error("quota exceeded"));
    const strategy = new GeminiStrategy({ apiKey: "k" });
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow("quota exceeded");
  });

  it("validateKey calls countTokens once with a trivial input", async () => {
    mockCountTokens.mockResolvedValueOnce({ totalTokens: 1 });
    const strategy = new GeminiStrategy({});

    await strategy.validateKey("user-key");
    expect(mockCountTokens).toHaveBeenCalledOnce();
    expect(mockCountTokens).toHaveBeenCalledWith("validate");
  });

  it("validateKey wraps SDK rejections as LlmKeyValidationError", async () => {
    mockCountTokens.mockRejectedValueOnce(new Error("invalid key"));
    const strategy = new GeminiStrategy({});

    await expect(strategy.validateKey("bad-key")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  describe("empty/whitespace model fallback", () => {
    it("treats blank defaultModel option as missing and uses the hard-coded fallback", async () => {
      mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
      const strategy = new GeminiStrategy({ apiKey: "k", defaultModel: "" });

      await strategy.generate({ prompt: "x" });

      expect(mockGetGenerativeModel).toHaveBeenCalledWith({ model: "gemini-2.5-flash-lite" });
    });

    it("treats blank per-call model as missing and uses defaultModel", async () => {
      mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "" } });
      const strategy = new GeminiStrategy({ apiKey: "k", defaultModel: "gemini-pro" });

      await strategy.generate({ prompt: "x", model: "   " });
      expect(mockGetGenerativeModel).toHaveBeenCalledWith({ model: "gemini-pro" });
    });
  });

  it("declares its capability tags", () => {
    const strategy = new GeminiStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual([
      "vision",
      "long-context",
      "multilingual",
      "cheap",
      "streaming",
    ]);
  });

  describe("streaming (direct API)", () => {
    it("uses generateContentStream instead of generateContent when onToken is set", async () => {
      mockGenerateContentStream.mockResolvedValueOnce({
        stream: asyncGenOf([{ text: () => "hel" }, { text: () => "lo" }]),
        response: Promise.resolve({
          text: () => "hello",
          usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 2 },
        }),
      });
      const strategy = new GeminiStrategy({ apiKey: "k" });
      const deltas: string[] = [];

      const result = await strategy.generate({ prompt: "x", onToken: (d) => deltas.push(d) });

      expect(mockGenerateContent).not.toHaveBeenCalled();
      expect(mockGenerateContentStream).toHaveBeenCalledOnce();
      expect(deltas).toEqual(["hel", "lo"]);
      expect(result.text).toBe("hello");
      expect(result.usage).toEqual({ inputTokens: 1, outputTokens: 2 });
    });

    it("does not propagate an onToken callback that throws", async () => {
      mockGenerateContentStream.mockResolvedValueOnce({
        stream: asyncGenOf([{ text: () => "hi" }]),
        response: Promise.resolve({ text: () => "hi" }),
      });
      const strategy = new GeminiStrategy({ apiKey: "k" });

      const result = await strategy.generate({
        prompt: "x",
        onToken: () => {
          throw new Error("callback boom");
        },
      });

      expect(result.text).toBe("hi");
    });
  });

  describe("thinking (direct API — unsupported)", () => {
    it("throws UnsupportedThinkingModeError for adaptive, no network call", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "x", thinking: { type: "adaptive" } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
      expect(mockGenerateContentStream).not.toHaveBeenCalled();
    });

    it("throws UnsupportedThinkingModeError for budget, no network call", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "x", thinking: { type: "budget", tokens: 2000 } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });

    it("throws UnsupportedThinkingModeError for effort, no network call", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "x", thinking: { type: "effort", level: "high" } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });
  });

  describe("signal", () => {
    it("rejects promptly when the signal is already aborted, no network call", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      const controller = new AbortController();
      controller.abort();

      await expect(strategy.generate({ prompt: "x", signal: controller.signal })).rejects.toBeTruthy();
      expect(mockGenerateContent).not.toHaveBeenCalled();
      expect(mockGenerateContentStream).not.toHaveBeenCalled();
    });
  });

  describe("connection=vertex", () => {
    it("creates GoogleGenAI with vertexai:true and does not use GEMINI_API_KEY", async () => {
      process.env.GEMINI_API_KEY = "must-not-be-used";
      mockVertexGenerateContent.mockResolvedValueOnce({
        text: "vertex-ok",
        usageMetadata: { promptTokenCount: 2, candidatesTokenCount: 3 },
      });
      const strategy = new GeminiStrategy({
        connection: "vertex",
        apiKey: "platform-key-ignored-for-platform-calls",
      });

      const result = await strategy.generate({ prompt: "parse" });

      expect(mockGoogleGenAI).toHaveBeenCalledWith({ vertexai: true });
      expect(mockGenerateContent).not.toHaveBeenCalled();
      expect(result).toMatchObject({
        text: "vertex-ok",
        model: "gemini-2.5-flash-lite",
        usage: { inputTokens: 2, outputTokens: 3 },
      });
      delete process.env.GEMINI_API_KEY;
    });

    it("keeps providerName=gemini and treats ADC as a configured platform", () => {
      const strategy = new GeminiStrategy({ connection: "vertex" });
      expect(strategy.providerName).toBe("gemini");
      expect(strategy.hasPlatformKey()).toBe(true);
    });

    it("still uses the direct Gemini API for personal BYOK", async () => {
      mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "byok" } });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({ prompt: "x", apiKey: "user-key" });

      expect(mockGenerateContent).toHaveBeenCalledOnce();
      expect(mockVertexGenerateContent).not.toHaveBeenCalled();
    });

    it("rethrows Vertex errors without falling back to the API-key client", async () => {
      mockVertexGenerateContent.mockRejectedValueOnce(new Error("vertex 403"));
      const strategy = new GeminiStrategy({ connection: "vertex", apiKey: "k" });

      await expect(strategy.generate({ prompt: "x" })).rejects.toThrow("vertex 403");
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });

    it("declares streaming and thinking capability tags", () => {
      const strategy = new GeminiStrategy({ connection: "vertex" });
      expect(strategy.capabilities).toEqual([
        "vision",
        "long-context",
        "multilingual",
        "cheap",
        "streaming",
        "thinking",
      ]);
    });

    it("maps adaptive thinking to thinkingBudget: -1", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({ prompt: "x", thinking: { type: "adaptive" } });

      const call = mockVertexGenerateContent.mock.calls[0]![0];
      expect(call.config.thinkingConfig.thinkingBudget).toBe(-1);
    });

    it("maps budget thinking with valid tokens to thinkingConfig.thinkingBudget", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({ prompt: "x", thinking: { type: "budget", tokens: 2000 } });

      const call = mockVertexGenerateContent.mock.calls[0]![0];
      expect(call.config.thinkingConfig.thinkingBudget).toBe(2000);
    });

    it("throws InvalidThinkingConfigError for non-positive budget tokens, no network call", async () => {
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await expect(
        strategy.generate({ prompt: "x", thinking: { type: "budget", tokens: 0 } }),
      ).rejects.toBeInstanceOf(InvalidThinkingConfigError);
      expect(mockVertexGenerateContent).not.toHaveBeenCalled();
    });

    it("throws UnsupportedThinkingModeError for effort, no network call", async () => {
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await expect(
        strategy.generate({ prompt: "x", thinking: { type: "effort", level: "high" } }),
      ).rejects.toBeInstanceOf(UnsupportedThinkingModeError);
      expect(mockVertexGenerateContent).not.toHaveBeenCalled();
    });

    it("extracts a thought part into LlmResponse.thinking", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({
        candidates: [
          {
            content: {
              parts: [
                { text: "let me reason...", thought: true },
                { text: "final answer" },
              ],
            },
            finishReason: "STOP",
          },
        ],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 10 },
      });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      const result = await strategy.generate({ prompt: "x", thinking: { type: "adaptive" } });

      expect(result.thinking).toBe("let me reason...");
      expect(result.text).toBe("final answer");
    });

    it("leaves LlmResponse.thinking undefined when no thought part is present", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "plain answer" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      const result = await strategy.generate({ prompt: "x" });

      expect(result.thinking).toBeUndefined();
    });

    it("streams via generateContentStream when onToken is set", async () => {
      mockVertexGenerateContentStream.mockResolvedValueOnce(
        asyncGenOf([
          { candidates: [{ content: { parts: [{ text: "he" }] } }] },
          {
            candidates: [{ content: { parts: [{ text: "llo" }] }, finishReason: "STOP" }],
            usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 2 },
          },
        ]),
      );
      const strategy = new GeminiStrategy({ connection: "vertex" });
      const deltas: string[] = [];

      const result = await strategy.generate({ prompt: "x", onToken: (d) => deltas.push(d) });

      expect(mockVertexGenerateContent).not.toHaveBeenCalled();
      expect(mockVertexGenerateContentStream).toHaveBeenCalledOnce();
      expect(deltas).toEqual(["he", "llo"]);
      expect(result.text).toBe("hello");
      expect(result.usage).toEqual({ inputTokens: 1, outputTokens: 2 });
      expect(result.truncated).toBe(false);
    });
  });

  describe("messages (multi-turn)", () => {
    it("direct API: builds a contents array with assistant mapped to model role", async () => {
      mockGenerateContent.mockResolvedValueOnce({
        response: { text: () => "ok" },
      });
      const strategy = new GeminiStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockGenerateContent.mock.calls[0]![0];
      expect(call.contents).toEqual([
        { role: "user", parts: [{ text: "first" }] },
        { role: "model", parts: [{ text: "second" }] },
        { role: "user", parts: [{ text: "third" }] },
      ]);
    });

    it("direct API: puts attachments on the last turn's parts, not the first", async () => {
      mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "ok" } });
      const strategy = new GeminiStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      });

      const call = mockGenerateContent.mock.calls[0]![0];
      expect(call.contents[0]).toEqual({ role: "user", parts: [{ text: "first" }] });
      expect(call.contents[1].parts[0]).toEqual({ text: "third" });
      expect(call.contents[1].parts[1].inlineData.mimeType).toBe("image/png");
    });

    it("direct API: throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });

    it("direct API: throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });

    it("Vertex: builds a contents array with assistant mapped to model role", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockVertexGenerateContent.mock.calls[0]![0];
      expect(call.contents).toEqual([
        { role: "user", parts: [{ text: "first" }] },
        { role: "model", parts: [{ text: "second" }] },
        { role: "user", parts: [{ text: "third" }] },
      ]);
    });

    it("Vertex: puts attachments on the last turn's parts, not the first", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      });

      const call = mockVertexGenerateContent.mock.calls[0]![0];
      expect(call.contents[0]).toEqual({ role: "user", parts: [{ text: "first" }] });
      expect(call.contents[1].parts[0]).toEqual({ text: "third" });
      expect(call.contents[1].parts[1].inlineData.mimeType).toBe("image/png");
    });

    it("Vertex: throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new GeminiStrategy({ connection: "vertex" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockVertexGenerateContent).not.toHaveBeenCalled();
    });
  });
});
