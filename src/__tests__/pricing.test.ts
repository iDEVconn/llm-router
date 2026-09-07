import { describe, expect, it } from "vitest";
import { calculateCost, type PricingTable } from "../pricing";

const pricing: PricingTable = {
  gemini: {
    "gemini-2.5-flash": { inputPer1M: 0.3, outputPer1M: 2.5 },
  },
};

describe("calculateCost", () => {
  it("computes cost from input/output tokens at the given per-1M rates", () => {
    const cost = calculateCost(
      { inputTokens: 1_000_000, outputTokens: 500_000 },
      "gemini",
      "gemini-2.5-flash",
      pricing,
    );
    expect(cost).toBeCloseTo(0.3 + 1.25, 6);
  });

  it("scales linearly for small token counts", () => {
    const cost = calculateCost(
      { inputTokens: 1_000, outputTokens: 0 },
      "gemini",
      "gemini-2.5-flash",
      pricing,
    );
    expect(cost).toBeCloseTo(0.0003, 8);
  });

  it("throws when the provider/model pair is not in the pricing table", () => {
    expect(() =>
      calculateCost({ inputTokens: 1, outputTokens: 1 }, "gemini", "unknown-model", pricing),
    ).toThrow(/gemini/);
  });
});
