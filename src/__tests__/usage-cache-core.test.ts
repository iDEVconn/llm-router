import { describe, expect, it, vi } from "vitest";
import { withBudget } from "../budget";
import { BudgetExceededError, InvalidGenerateOptionsError } from "../errors";
import { withInstrumentation } from "../instrumentation";
import { calculateCost, splitInputTokens, type PricingTable } from "../pricing";
import type { LlmResponse, LlmStrategy, LlmUsage } from "../types";
import { resolveAbortUsage } from "../usage-estimate";
import { assertApiKeyIfRequired } from "../validate-generate-options";
import * as publicApi from "../index";

const pricing: PricingTable = {
  claude: {
    // Anthropic-style: write 1.25x input, read 0.1x input.
    "claude-sonnet": { inputPer1M: 3, outputPer1M: 15, cacheWritePer1M: 3.75, cacheReadPer1M: 0.3 },
    "no-cache-rates": { inputPer1M: 3, outputPer1M: 15 },
  },
  chatgpt: {
    "gpt-x": { inputPer1M: 2, outputPer1M: 8, cacheReadPer1M: 0.5 },
    "gpt-no-cache-rates": { inputPer1M: 2, outputPer1M: 8 },
  },
};

describe("splitInputTokens", () => {
  it("treats a usage without cache fields as fully uncached (unchanged behaviour)", () => {
    expect(splitInputTokens({ inputTokens: 100, outputTokens: 5 })).toEqual({
      uncachedInputTokens: 100,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
      totalInputTokens: 100,
    });
  });

  it("adds Claude-style cache counters on top of inputTokens (disjoint)", () => {
    expect(
      splitInputTokens({
        inputTokens: 10,
        outputTokens: 5,
        cacheCreationInputTokens: 1000,
        cacheReadInputTokens: 4000,
      }),
    ).toEqual({
      uncachedInputTokens: 10,
      cacheCreationInputTokens: 1000,
      cacheReadInputTokens: 4000,
      totalInputTokens: 5010,
    });
  });

  it("subtracts OpenAI-style cache counters that are already inside inputTokens", () => {
    expect(
      splitInputTokens({
        inputTokens: 5000,
        outputTokens: 5,
        cacheReadInputTokens: 4096,
        inputTokensIncludeCache: true,
      }),
    ).toEqual({
      uncachedInputTokens: 904,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 4096,
      totalInputTokens: 5000,
    });
  });

  it("never goes negative when an included cache count exceeds inputTokens", () => {
    expect(
      splitInputTokens({
        inputTokens: 10,
        outputTokens: 0,
        cacheReadInputTokens: 50,
        inputTokensIncludeCache: true,
      }).uncachedInputTokens,
    ).toBe(0);
  });

  it("ignores NaN / negative / non-finite cache counters", () => {
    expect(
      splitInputTokens({
        inputTokens: 10,
        outputTokens: 0,
        cacheCreationInputTokens: Number.NaN,
        cacheReadInputTokens: -5,
      }),
    ).toEqual({
      uncachedInputTokens: 10,
      cacheCreationInputTokens: 0,
      cacheReadInputTokens: 0,
      totalInputTokens: 10,
    });
  });

  it("is exported from the package root", () => {
    expect(publicApi.splitInputTokens).toBe(splitInputTokens);
  });
});

describe("calculateCost with cache tokens", () => {
  it("is unchanged for a usage without cache fields", () => {
    expect(
      calculateCost({ inputTokens: 1_000_000, outputTokens: 1_000_000 }, "claude", "claude-sonnet", pricing),
    ).toBeCloseTo(18, 9);
  });

  it("prices Claude cache writes and reads at their own rates on top of inputTokens", () => {
    const cost = calculateCost(
      {
        inputTokens: 1_000_000,
        outputTokens: 1_000_000,
        cacheCreationInputTokens: 1_000_000,
        cacheReadInputTokens: 1_000_000,
      },
      "claude",
      "claude-sonnet",
      pricing,
    );
    // 3 (input) + 3.75 (write) + 0.3 (read) + 15 (output)
    expect(cost).toBeCloseTo(22.05, 9);
  });

  it("falls back to inputPer1M for cache tokens when the table has no cache rates", () => {
    const cost = calculateCost(
      { inputTokens: 1_000_000, outputTokens: 0, cacheCreationInputTokens: 1_000_000, cacheReadInputTokens: 1_000_000 },
      "claude",
      "no-cache-rates",
      pricing,
    );
    expect(cost).toBeCloseTo(9, 9);
  });

  it("does not double-count OpenAI-style cached tokens included in inputTokens", () => {
    const cost = calculateCost(
      { inputTokens: 1_000_000, outputTokens: 0, cacheReadInputTokens: 400_000, inputTokensIncludeCache: true },
      "chatgpt",
      "gpt-x",
      pricing,
    );
    // 600k uncached * 2 + 400k cached * 0.5
    expect(cost).toBeCloseTo(1.2 + 0.2, 9);
  });

  it("keeps the old cost for OpenAI-style usage when the table has no cache rates", () => {
    const cost = calculateCost(
      { inputTokens: 1_000_000, outputTokens: 0, cacheReadInputTokens: 400_000, inputTokensIncludeCache: true },
      "chatgpt",
      "gpt-no-cache-rates",
      pricing,
    );
    expect(cost).toBeCloseTo(2, 9);
  });

  it("treats zero cache counters like absent ones", () => {
    const cost = calculateCost(
      { inputTokens: 1_000_000, outputTokens: 0, cacheCreationInputTokens: 0, cacheReadInputTokens: 0 },
      "claude",
      "claude-sonnet",
      pricing,
    );
    expect(cost).toBeCloseTo(3, 9);
  });
});

function fakeStrategy(usage: LlmUsage, model = "claude-sonnet"): LlmStrategy {
  return {
    providerName: "claude",
    defaultModel: model,
    async generate(): Promise<LlmResponse> {
      return { text: "ok", model, usage, truncated: false };
    },
    async validateKey() {},
  };
}

describe("withBudget with cache tokens", () => {
  it("includes cache tokens in the per-call cost and passes the full usage to onCost", async () => {
    const usage: LlmUsage = {
      inputTokens: 0,
      outputTokens: 0,
      cacheCreationInputTokens: 1_000_000,
      cacheReadInputTokens: 1_000_000,
    };
    const onCost = vi.fn();
    const strategy = withBudget(fakeStrategy(usage), { pricing, onCost });
    await strategy.generate({ prompt: "x" });
    expect(onCost).toHaveBeenCalledOnce();
    expect(onCost.mock.calls[0]![0].cost).toBeCloseTo(4.05, 9);
    expect(onCost.mock.calls[0]![0].usage).toEqual(usage);
  });

  it("enforces maxCostPerCall against the cache-inclusive cost", async () => {
    const strategy = withBudget(
      fakeStrategy({ inputTokens: 0, outputTokens: 0, cacheCreationInputTokens: 1_000_000 }),
      { pricing, maxCostPerCall: 1 },
    );
    await expect(strategy.generate({ prompt: "x" })).rejects.toBeInstanceOf(BudgetExceededError);
  });
});

describe("withInstrumentation with cache tokens", () => {
  it("forwards the cache counters on the call event unchanged", async () => {
    const usage: LlmUsage = { inputTokens: 1, outputTokens: 2, cacheReadInputTokens: 3 };
    const onCall = vi.fn();
    await withInstrumentation(fakeStrategy(usage), { onCall }).generate({ prompt: "x" });
    expect(onCall.mock.calls[0]![0].usage).toEqual(usage);
  });
});

describe("resolveAbortUsage with cache tokens", () => {
  const opts = { prompt: "12345678" };

  it("carries reported cache counters when the input side was reported", () => {
    expect(
      resolveAbortUsage(
        { inputTokens: 10, outputTokens: 3, cacheCreationInputTokens: 100, cacheReadInputTokens: 0 },
        opts,
        "",
      ),
    ).toEqual({
      usage: { inputTokens: 10, outputTokens: 3, cacheCreationInputTokens: 100, cacheReadInputTokens: 0 },
      usageEstimated: false,
    });
  });

  it("carries the include flag together with the cache counters", () => {
    expect(
      resolveAbortUsage(
        { inputTokens: 10, outputTokens: 3, cacheReadInputTokens: 4, inputTokensIncludeCache: true },
        opts,
        "",
      ).usage,
    ).toEqual({ inputTokens: 10, outputTokens: 3, cacheReadInputTokens: 4, inputTokensIncludeCache: true });
  });

  it("drops cache counters when the input side is an estimate (it already covers the whole prompt)", () => {
    expect(
      resolveAbortUsage({ cacheReadInputTokens: 50 }, opts, "abcd").usage,
    ).toEqual({ inputTokens: 2, outputTokens: 1 });
  });

  it("omits null / NaN / negative cache counters", () => {
    expect(
      resolveAbortUsage(
        { inputTokens: 10, outputTokens: 3, cacheCreationInputTokens: null, cacheReadInputTokens: -1 },
        opts,
        "",
      ).usage,
    ).toEqual({ inputTokens: 10, outputTokens: 3 });
  });
});

describe("assertApiKeyIfRequired", () => {
  it("does nothing when requireApiKey is not set", () => {
    expect(() => assertApiKeyIfRequired({})).not.toThrow();
    expect(() => assertApiKeyIfRequired({ apiKey: "" })).not.toThrow();
  });

  it("throws InvalidGenerateOptionsError for a missing or blank key when requireApiKey is true", () => {
    expect(() => assertApiKeyIfRequired({ requireApiKey: true })).toThrow(InvalidGenerateOptionsError);
    expect(() => assertApiKeyIfRequired({ requireApiKey: true, apiKey: "  " })).toThrow(
      InvalidGenerateOptionsError,
    );
  });

  it("passes when requireApiKey is true and a key is given", () => {
    expect(() => assertApiKeyIfRequired({ requireApiKey: true, apiKey: "sk-x" })).not.toThrow();
  });
});
