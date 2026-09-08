import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { withCircuitBreaker } from "../circuit-breaker";
import { CircuitBreakerOpenError } from "../errors";
import { BudgetExceededError } from "../errors";
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

describe("withCircuitBreaker", () => {
  it("stays closed and passes calls through below the failure threshold", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValueOnce(new Error("blip")).mockResolvedValueOnce(okResponse);
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 3,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 10_000,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("blip");
    await expect(wrapped.generate({ prompt: "2" })).resolves.toEqual(okResponse);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("does not reset the failure count on an interleaved success", async () => {
    const strategy = makeStrategy();
    strategy.generate
      .mockRejectedValueOnce(new Error("down"))
      .mockResolvedValueOnce(okResponse)
      .mockRejectedValueOnce(new Error("down"));
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 2,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 10_000,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("down");
    await expect(wrapped.generate({ prompt: "2" })).resolves.toEqual(okResponse);
    await expect(wrapped.generate({ prompt: "3" })).rejects.toThrow("down");

    // Both failures (1 and 3) are within the same window despite the
    // interleaved success — the breaker should now be open.
    await expect(wrapped.generate({ prompt: "4" })).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    expect(strategy.generate).toHaveBeenCalledTimes(3);
  });

  it("opens after `threshold` failures within the sampling window and fails fast without calling the strategy", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new Error("down"));
    const onStateChange = vi.fn();
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 2,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 10_000,
      onStateChange,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("down");
    await expect(wrapped.generate({ prompt: "2" })).rejects.toThrow("down");
    expect(onStateChange).toHaveBeenCalledWith({ providerName: "claude", state: "open" });

    await expect(wrapped.generate({ prompt: "3" })).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("does not count failures older than the sampling window", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new Error("down"));
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 2,
      samplingWindowMs: 1_000,
      resetTimeoutMs: 10_000,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("down");
    await vi.advanceTimersByTimeAsync(2_000);
    await expect(wrapped.generate({ prompt: "2" })).rejects.toThrow("down");

    // Only 1 failure inside the window at any point so far — still closed.
    await expect(wrapped.generate({ prompt: "3" })).rejects.toThrow("down");
    expect(strategy.generate).toHaveBeenCalledTimes(3);
  });

  it("transitions open -> half-open after resetTimeoutMs, and half-open success -> closed", async () => {
    const strategy = makeStrategy();
    strategy.generate
      .mockRejectedValueOnce(new Error("down"))
      .mockRejectedValueOnce(new Error("down"))
      .mockResolvedValueOnce(okResponse);
    const onStateChange = vi.fn();
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 2,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 5_000,
      onStateChange,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("down");
    await expect(wrapped.generate({ prompt: "2" })).rejects.toThrow("down");

    await vi.advanceTimersByTimeAsync(5_001);

    await expect(wrapped.generate({ prompt: "3" })).resolves.toEqual(okResponse);
    expect(onStateChange.mock.calls.map((c) => c[0].state)).toEqual(["open", "half-open", "closed"]);

    // Breaker is closed again: further failures start a fresh count.
    strategy.generate.mockRejectedValueOnce(new Error("down"));
    await expect(wrapped.generate({ prompt: "4" })).rejects.toThrow("down");
    strategy.generate.mockResolvedValueOnce(okResponse);
    await expect(wrapped.generate({ prompt: "5" })).resolves.toEqual(okResponse);
  });

  it("half-open failure reopens the breaker (resets the timeout window)", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new Error("still down"));
    const onStateChange = vi.fn();
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 1,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 5_000,
      onStateChange,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("still down");
    expect(onStateChange).toHaveBeenLastCalledWith({ providerName: "claude", state: "open" });

    await vi.advanceTimersByTimeAsync(5_001);
    await expect(wrapped.generate({ prompt: "2" })).rejects.toThrow("still down");
    expect(onStateChange.mock.calls.map((c) => c[0].state)).toEqual(["open", "half-open", "open"]);

    // Immediately after re-opening, still fails fast (fresh resetTimeoutMs window).
    await expect(wrapped.generate({ prompt: "3" })).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("only lets one concurrent call become the half-open trial", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValueOnce(new Error("down"));
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 1,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 5_000,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toThrow("down");
    await vi.advanceTimersByTimeAsync(5_001);

    let resolveTrial!: () => void;
    strategy.generate.mockImplementationOnce(
      () => new Promise((resolve) => { resolveTrial = () => resolve(okResponse); }),
    );

    const trial = wrapped.generate({ prompt: "trial" });
    const concurrent = wrapped.generate({ prompt: "concurrent" });

    await expect(concurrent).rejects.toBeInstanceOf(CircuitBreakerOpenError);
    resolveTrial();
    await expect(trial).resolves.toEqual(okResponse);
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("does not count a caller-fault error as a circuit-breaker failure", async () => {
    const strategy = makeStrategy();
    strategy.generate.mockRejectedValue(new BudgetExceededError("perCall", 5, 1));
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 1,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 5_000,
    });

    await expect(wrapped.generate({ prompt: "1" })).rejects.toBeInstanceOf(BudgetExceededError);
    await expect(wrapped.generate({ prompt: "2" })).rejects.toBeInstanceOf(BudgetExceededError);
    // Still closed — real provider errors, not BudgetExceededError, would have opened it after 1.
    expect(strategy.generate).toHaveBeenCalledTimes(2);
  });

  it("does not count an aborted signal as a circuit-breaker failure", async () => {
    const strategy = makeStrategy();
    const controller = new AbortController();
    const abortErr = new Error("aborted");
    strategy.generate.mockImplementation(() => {
      controller.abort(abortErr);
      return Promise.reject(abortErr);
    });
    const wrapped = withCircuitBreaker(strategy, {
      threshold: 1,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 5_000,
    });

    await expect(
      wrapped.generate({ prompt: "1", signal: controller.signal }),
    ).rejects.toThrow("aborted");

    const controller2 = new AbortController();
    strategy.generate.mockResolvedValueOnce(okResponse);
    await expect(
      wrapped.generate({ prompt: "2", signal: controller2.signal }),
    ).resolves.toEqual(okResponse);
  });
});
