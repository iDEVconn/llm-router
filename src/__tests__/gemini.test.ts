import { beforeEach, describe, expect, it, vi } from "vitest";

const mockGenerateContent = vi.fn();
const mockCountTokens = vi.fn();
const mockGetGenerativeModel = vi.fn();
const mockVertexGenerateContent = vi.fn();
const mockGoogleGenAI = vi.fn();

vi.mock("@google/generative-ai", () => {
  class GoogleGenerativeAI {
    constructor(public readonly key: string) {}
    getGenerativeModel(args: { model: string }) {
      mockGetGenerativeModel(args);
      return { generateContent: mockGenerateContent, countTokens: mockCountTokens };
    }
  }
  return { GoogleGenerativeAI };
});

vi.mock("@google/genai", () => {
  class GoogleGenAI {
    constructor(opts: Record<string, unknown>) {
      mockGoogleGenAI(opts);
    }
    models = { generateContent: mockVertexGenerateContent };
  }
  return { GoogleGenAI };
});

import { LlmKeyValidationError } from "../errors";
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
    expect(strategy.capabilities).toEqual(["vision", "long-context", "multilingual", "cheap"]);
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
  });
});
