import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { withRetry } from "../retry";
import {
  BudgetExceededError,
  CircuitBreakerOpenError,
  RateLimitExceededError,
} from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

function makeStrategy(): LlmStrategy & {
  generate: ReturnType<typeof vi.fn<(opts: LlmGenerateOptions) => Promise<LlmResponse>>>;
} {
  return {
    providerName: "claude",
    defaultModel: "claude-haiku-4-5",
    generate: vi.fn<(opts: LlmGenerateOptions) => Promise<LlmResponse>>(),
    validateKey: vi.fn(),
  };
}

const okResponse = {
  text: "ok",
  model: "claude-haiku-4-5",
  usage: { inputTokens: 1, outputTokens: 1 },
  truncated: false,
};

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("withRetry", () => {
  it("succeeds on the Nth attempt after transient failures", async () => {
    const strategy = makeStrategy();
    strategy.generate
      .mockRejectedValueOnce(new Error("blip 1"))
      .mockRejectedValueOnce(new Error("blip 2"))
      .mockResolvedValueOnce(okResponse);
    const wrapped = withRetry(strategy, {
      maxAttempts: 3,
      initialBackoffMs: 100,
      maxBackoffMs: 1_000,
      multiplier: 2,
      jitter: false,
    });

    const promise = wrapped.generate({ prompt: "hi" });
    await vi.runAllTimersAsync();

    await expect(promise).resolves.toEqual(okResponse);
    expect(strategy.generate).toHaveBeenCalledTimes(3);
  });

  it("exhausts attempts and rethrows the original error type", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new Error("always fails"));
    const wrapped = withRetry(strategy, {
      maxAttempts: 3,
      initialBackoffMs: 10,
      maxBackoffMs: 100,
      multiplier: 2,
      jitter: false,
    });

    const promise = wrapped.generate({ prompt: "hi" });
    const assertion = expect(promise).rejects.toThrow("always fails");
    await vi.runAllTimersAsync();

    await assertion;
    expect(strategy.generate).toHaveBeenCalledTimes(3);
  });

  it("does not retry a caller-fault error", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new BudgetExceededError("perCall", 5, 1));
    const wrapped = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 10,
      maxBackoffMs: 100,
      multiplier: 2,
    });

    await expect(wrapped.generate({ prompt: "hi" })).rejects.toBeInstanceOf(BudgetExceededError);
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("does not retry CircuitBreakerOpenError", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new CircuitBreakerOpenError("claude", 5_000));
    const wrapped = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 10,
      maxBackoffMs: 100,
      multiplier: 2,
    });

    await expect(wrapped.generate({ prompt: "hi" })).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("does not retry RateLimitExceededError", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new RateLimitExceededError("claude", 30_000));
    const wrapped = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 10,
      maxBackoffMs: 100,
      multiplier: 2,
    });

    await expect(wrapped.generate({ prompt: "hi" })).rejects.toBeInstanceOf(RateLimitExceededError);
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("does not retry an aborted call", async () => {
    const strategy = makeStrategy();
    const controller = new AbortController();
    const abortErr = new Error("aborted");
    strategy.generate.mockImplementation(() => {
      controller.abort(abortErr);
      return Promise.reject(abortErr);
    });
    const wrapped = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 10,
      maxBackoffMs: 100,
      multiplier: 2,
    });

    await expect(
      wrapped.generate({ prompt: "hi", signal: controller.signal }),
    ).rejects.toThrow("aborted");
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("skips all retry logic and calls exactly once when onToken is set", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValueOnce(new Error("blip"));
    const wrapped = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 10,
      maxBackoffMs: 100,
      multiplier: 2,
    });

    await expect(
      wrapped.generate({ prompt: "hi", onToken: () => {} }),
    ).rejects.toThrow("blip");
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });

  it("computes exponential backoff and calls onRetry with the delay", async () => {
    const strategy = makeStrategy();
    strategy.generate
      .mockRejectedValueOnce(new Error("blip 1"))
      .mockRejectedValueOnce(new Error("blip 2"))
      .mockResolvedValueOnce(okResponse);
    const onRetry = vi.fn();
    const wrapped = withRetry(strategy, {
      maxAttempts: 3,
      initialBackoffMs: 100,
      maxBackoffMs: 10_000,
      multiplier: 2,
      jitter: false,
      onRetry,
    });

    const promise = wrapped.generate({ prompt: "hi" });
    await vi.runAllTimersAsync();
    await promise;

    expect(onRetry).toHaveBeenNthCalledWith(1, {
      providerName: "claude",
      attempt: 1,
      error: expect.any(Error),
      delayMs: 100,
    });
    expect(onRetry).toHaveBeenNthCalledWith(2, {
      providerName: "claude",
      attempt: 2,
      error: expect.any(Error),
      delayMs: 200,
    });
  });

  it("caps backoff at maxBackoffMs", async () => {
    const strategy = makeStrategy();
    strategy.generate
      .mockRejectedValueOnce(new Error("1"))
      .mockRejectedValueOnce(new Error("2"))
      .mockRejectedValueOnce(new Error("3"))
      .mockResolvedValueOnce(okResponse);
    const onRetry = vi.fn();
    const wrapped = withRetry(strategy, {
      maxAttempts: 4,
      initialBackoffMs: 100,
      maxBackoffMs: 250,
      multiplier: 10,
      jitter: false,
      onRetry,
    });

    const promise = wrapped.generate({ prompt: "hi" });
    await vi.runAllTimersAsync();
    await promise;

    const delays = onRetry.mock.calls.map((c) => c[0].delayMs);
    expect(delays).toEqual([100, 250, 250]);
  });

  it("applies jitter within the 0.5x-1.0x range of the base delay", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValueOnce(new Error("blip")).mockResolvedValueOnce(okResponse);
    const onRetry = vi.fn();
    const originalRandom = Math.random;
    Math.random = () => 0.5;
    try {
      const wrapped = withRetry(strategy, {
        maxAttempts: 2,
        initialBackoffMs: 1_000,
        maxBackoffMs: 10_000,
        multiplier: 2,
        jitter: true,
        onRetry,
      });
      const promise = wrapped.generate({ prompt: "hi" });
      await vi.runAllTimersAsync();
      await promise;
    } finally {
      Math.random = originalRandom;
    }

    // Math.random() fixed at 0.5 -> jitter factor 0.5 + 0.5*0.5 = 0.75
    expect(onRetry.mock.calls[0]![0].delayMs).toBe(750);
  });

  it("stops retrying immediately when the signal aborts during a backoff sleep", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new Error("blip"));
    const controller = new AbortController();
    const wrapped = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 10_000,
      maxBackoffMs: 10_000,
      multiplier: 1,
      jitter: false,
    });

    const promise = wrapped.generate({ prompt: "hi", signal: controller.signal });
    const abortReason = new Error("cancelled mid-backoff");
    // Give the first failed attempt a tick to register before aborting.
    await vi.advanceTimersByTimeAsync(0);
    controller.abort(abortReason);

    await expect(promise).rejects.toThrow("cancelled mid-backoff");
    expect(strategy.generate).toHaveBeenCalledTimes(1);
  });
});
