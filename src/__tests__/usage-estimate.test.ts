import { describe, expect, it } from "vitest";
import { estimateInputTokens, estimateTokens, resolveAbortUsage } from "../usage-estimate";

describe("estimateTokens", () => {
  it("is 0 for an empty string", () => {
    expect(estimateTokens("")).toBe(0);
  });

  it("is ceil(chars / 4)", () => {
    expect(estimateTokens("a")).toBe(1);
    expect(estimateTokens("abcd")).toBe(1);
    expect(estimateTokens("abcde")).toBe(2);
    expect(estimateTokens("x".repeat(400))).toBe(100);
  });

  it("counts non-Latin scripts by UTF-16 length, no language-specific logic", () => {
    expect(estimateTokens("שלום עולם")).toBe(Math.ceil("שלום עולם".length / 4)); // Hebrew, 9 chars → 3
    expect(estimateTokens("Привет мир")).toBe(3); // Cyrillic, 10 chars
    expect(estimateTokens("😀")).toBe(1); // surrogate pair = 2 code units
  });
});

describe("estimateInputTokens", () => {
  it("is 0 when there is no text at all", () => {
    expect(estimateInputTokens({ prompt: "" })).toBe(0);
    expect(estimateInputTokens({})).toBe(0);
  });

  it("sums systemPrompt + prompt", () => {
    expect(estimateInputTokens({ prompt: "12345678", systemPrompt: "abcd" })).toBe(3);
  });

  it("sums systemPrompt + every multi-turn message", () => {
    expect(
      estimateInputTokens({
        systemPrompt: "ab",
        messages: [
          { role: "user", content: "abcd" },
          { role: "assistant", content: "ef" },
        ],
      }),
    ).toBe(2);
  });
});

describe("resolveAbortUsage", () => {
  const opts = { prompt: "12345678" }; // → 2 input tokens estimated

  it("uses provider-reported numbers and usageEstimated=false only when both sides were reported", () => {
    expect(resolveAbortUsage({ inputTokens: 10, outputTokens: 4 }, opts, "whatever")).toEqual({
      usage: { inputTokens: 10, outputTokens: 4 },
      usageEstimated: false,
    });
  });

  it("treats a reported 0 as reported", () => {
    expect(resolveAbortUsage({ inputTokens: 0, outputTokens: 0 }, opts, "abcd")).toEqual({
      usage: { inputTokens: 0, outputTokens: 0 },
      usageEstimated: false,
    });
  });

  it("estimates only the missing output side", () => {
    expect(resolveAbortUsage({ inputTokens: 10 }, opts, "abcdefgh")).toEqual({
      usage: { inputTokens: 10, outputTokens: 2 },
      usageEstimated: true,
    });
  });

  it("estimates only the missing input side (null counts as missing)", () => {
    expect(resolveAbortUsage({ inputTokens: null, outputTokens: 5 }, opts, "")).toEqual({
      usage: { inputTokens: 2, outputTokens: 5 },
      usageEstimated: true,
    });
  });

  it("estimates both sides when nothing was reported, counting thinking toward output", () => {
    expect(resolveAbortUsage({}, opts, "abcd", "efghij")).toEqual({
      usage: { inputTokens: 2, outputTokens: 3 },
      usageEstimated: true,
    });
  });

  it("is zero output for zero received text", () => {
    expect(resolveAbortUsage({}, opts, "")).toEqual({
      usage: { inputTokens: 2, outputTokens: 0 },
      usageEstimated: true,
    });
  });
});
