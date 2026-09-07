import { describe, expect, it, vi } from "vitest";
import { withBudget } from "../budget";
import { BudgetExceededError } from "../errors";
import type { LlmStrategy } from "../types";
import type { PricingTable } from "../pricing";

const pricing: PricingTable = {
  gemini: {
    "gemini-2.5-flash": { inputPer1M: 1, outputPer1M: 1 },
  },
};

function makeStrategy(usage: { inputTokens: number; outputTokens: number }): LlmStrategy {
  return {
    providerName: "gemini",
    defaultModel: "gemini-2.5-flash",
    generate: vi.fn().mockResolvedValue({
      text: "hi",
      model: "gemini-2.5-flash",
      usage,
      truncated: false,
    }),
    validateKey: vi.fn(),
  };
}

describe("withBudget", () => {
  it("calls onCost with the computed cost and usage after a successful call", async () => {
    const strategy = makeStrategy({ inputTokens: 1_000_000, outputTokens: 1_000_000 });
    const onCost = vi.fn();
    const wrapped = withBudget(strategy, { pricing, onCost });

    await wrapped.generate({ prompt: "hi" });

    expect(onCost).toHaveBeenCalledWith({
      provider: "gemini",
      model: "gemini-2.5-flash",
      cost: 2,
      usage: { inputTokens: 1_000_000, outputTokens: 1_000_000 },
    });
  });

  it("throws BudgetExceededError when a single call's cost exceeds maxCostPerCall", async () => {
    const strategy = makeStrategy({ inputTokens: 1_000_000, outputTokens: 1_000_000 });
    const wrapped = withBudget(strategy, { pricing, maxCostPerCall: 1 });

    await expect(wrapped.generate({ prompt: "hi" })).rejects.toThrow(BudgetExceededError);
  });

  it("throws BudgetExceededError before making a further call once maxCostTotal is spent", async () => {
    const strategy = makeStrategy({ inputTokens: 1_000_000, outputTokens: 0 });
    const wrapped = withBudget(strategy, { pricing, maxCostTotal: 1 });

    await wrapped.generate({ prompt: "one" });
    await expect(wrapped.generate({ prompt: "two" })).rejects.toThrow(BudgetExceededError);
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("does not call onCost when the call exceeds maxCostPerCall", async () => {
    const strategy = makeStrategy({ inputTokens: 1_000_000, outputTokens: 1_000_000 });
    const onCost = vi.fn();
    const wrapped = withBudget(strategy, { pricing, maxCostPerCall: 1, onCost });

    await expect(wrapped.generate({ prompt: "hi" })).rejects.toThrow(BudgetExceededError);
    expect(onCost).not.toHaveBeenCalled();
  });
});
