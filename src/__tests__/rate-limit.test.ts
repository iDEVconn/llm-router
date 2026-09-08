import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { withRateLimit } from "../rate-limit";
import { RateLimitExceededError } from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

function makeStrategy(): LlmStrategy & {
  generate: ReturnType<typeof vi.fn<(opts: LlmGenerateOptions) => Promise<LlmResponse>>>;
} {
  return {
    providerName: "chatgpt",
    defaultModel: "gpt-4.1-mini",
    generate: vi.fn<(opts: LlmGenerateOptions) => Promise<LlmResponse>>(),
    validateKey: vi.fn(),
  };
}

const okResponse = {
  text: "ok",
  model: "gpt-4.1-mini",
  usage: { inputTokens: 1, outputTokens: 1 },
  truncated: false,
};

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("withRateLimit", () => {
  it("passes calls through immediately when capacity is available", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockResolvedValue(okResponse);
    const wrapped = withRateLimit(strategy, { tokensPerSecond: 10 });

    await expect(wrapped.generate({ prompt: "hi" })).resolves.toEqual(okResponse);
  });

  it("waits for the bucket to refill once the initial capacity is exhausted", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockResolvedValue(okResponse);
    const wrapped = withRateLimit(strategy, { tokensPerSecond: 1, maxWaitMs: 5_000 });

    await wrapped.generate({ prompt: "1" }); // consumes the only token

    const promise = wrapped.generate({ prompt: "2" });
    // 1 token/sec -> refills after ~1s; advance past the exact boundary
    // (not exactly 1_000ms) so floating-point summation of the 25ms poll
    // increments can't leave the promise pending by a fraction of a token.
    await vi.advanceTimersByTimeAsync(1_100);
    await expect(promise).resolves.toEqual(okResponse);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("throws RateLimitExceededError when capacity never frees up within maxWaitMs", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockResolvedValue(okResponse);
    const wrapped = withRateLimit(strategy, {
      tokensPerSecond: 0.01, // effectively never refills within the wait window
      maxWaitMs: 1_000,
    });

    await wrapped.generate({ prompt: "1" });

    const promise = wrapped.generate({ prompt: "2" });
    const assertion = expect(promise).rejects.toBeInstanceOf(RateLimitExceededError);
    await vi.advanceTimersByTimeAsync(1_100);
    await assertion;
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("gates on maxConcurrent independently of the token bucket", async () => {
    const strategy = makeStrategy();
    let resolveFirst!: () => void;
    strategy.generate
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirst = () => resolve(okResponse); }))
      .mockResolvedValueOnce(okResponse);
    const wrapped = withRateLimit(strategy, { tokensPerSecond: 100, maxConcurrent: 1, maxWaitMs: 5_000 });

    const first = wrapped.generate({ prompt: "1" });
    await vi.advanceTimersByTimeAsync(0);

    const second = wrapped.generate({ prompt: "2" });
    await vi.advanceTimersByTimeAsync(50); // second is waiting on the concurrency slot, not the bucket
    expect(strategy.generate).toHaveBeenCalledTimes(1);

    resolveFirst();
    await first;
    await vi.advanceTimersByTimeAsync(50);
    await expect(second).resolves.toEqual(okResponse);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("releases the concurrency slot even when the wrapped call throws", async () => {
    const strategy = makeStrategy();
    strategy.generate
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(okResponse);
    const wrapped = withRateLimit(strategy, { tokensPerSecond: 100, maxConcurrent: 1 });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("boom");
    await expect(wrapped.generate({ prompt: "2" })).resolves.toEqual(okResponse);
  });

  it("stops waiting immediately when the signal aborts", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockResolvedValue(okResponse);
    const wrapped = withRateLimit(strategy, { tokensPerSecond: 0.01, maxWaitMs: 10_000 });
    await wrapped.generate({ prompt: "1" });

    const controller = new AbortController();
    const promise = wrapped.generate({ prompt: "2", signal: controller.signal });
    await vi.advanceTimersByTimeAsync(0);
    const abortReason = new Error("cancelled");
    controller.abort(abortReason);

    await expect(promise).rejects.toThrow("cancelled");
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("calls onThrottle only when an actual wait happened", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockResolvedValue(okResponse);
    const onThrottle = vi.fn();
    const wrapped = withRateLimit(strategy, { tokensPerSecond: 1, maxWaitMs: 5_000, onThrottle });

    await wrapped.generate({ prompt: "1" });
    expect(onThrottle).not.toHaveBeenCalled();

    const promise = wrapped.generate({ prompt: "2" });
    // Same floating-point-boundary margin as the refill test above.
    await vi.advanceTimersByTimeAsync(1_100);
    await promise;
    expect(onThrottle).toHaveBeenCalledTimes(1);
    expect(onThrottle.mock.calls[0]![0].providerName).toBe("chatgpt");
    expect(onThrottle.mock.calls[0]![0].waitedMs).toBeGreaterThan(0);
  });
});
