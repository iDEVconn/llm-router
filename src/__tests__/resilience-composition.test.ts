import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { compose } from "../instrumentation";
import { withCircuitBreaker } from "../circuit-breaker";
import { withRetry } from "../retry";
import { CircuitBreakerOpenError } from "../errors";
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

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
});

describe("withCircuitBreaker + withRetry composed", () => {
  it("once the breaker opens mid-retry, retry stops immediately instead of exhausting all attempts, and a subsequent call adds zero more strategy calls", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new Error("provider down"));

    const resilient = compose(
      strategy,
      (s) => withCircuitBreaker(s, { threshold: 2, samplingWindowMs: 60_000, resetTimeoutMs: 30_000 }),
      (s) => withRetry(s, { maxAttempts: 5, initialBackoffMs: 10, maxBackoffMs: 100, multiplier: 2, jitter: false }),
    );

    // First call: retry attempts 1 and 2 both reach the provider and fail,
    // tripping the breaker (threshold=2). Attempt 3 hits the now-open
    // breaker and gets CircuitBreakerOpenError instead of "provider down" —
    // withRetry does not retry that error type, so the WHOLE call rejects
    // with CircuitBreakerOpenError (not the original "provider down"),
    // and only 2 strategy calls happen, not all 5 configured attempts.
    const firstCall = resilient.generate({ prompt: "1" });
    await vi.runAllTimersAsync();
    await expect(firstCall).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    expect(strategy.generate).toHaveBeenCalledTimes(2);

    // Second call: breaker is still open (resetTimeoutMs=30_000 hasn't
    // elapsed). withRetry must not retry a CircuitBreakerOpenError, so
    // this call fails immediately with zero additional strategy calls.
    const secondCall = resilient.generate({ prompt: "2" });
    await vi.runAllTimersAsync();
    await expect(secondCall).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });
});
