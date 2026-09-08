import { CircuitBreakerOpenError } from "./errors";
import { isAbortError, isCallerFaultError } from "./resilience-errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "./types";

export interface CircuitBreakerStateChangeEvent {
  providerName: string;
  state: "closed" | "open" | "half-open";
}

export interface WithCircuitBreakerOptions {
  /**
   * Failures within the rolling window before the breaker opens (not
   * required to be consecutive — a success in between does not reset the
   * count, only time-based pruning does).
   */
  threshold: number;
  /** Rolling window (ms) failures are counted over. */
  samplingWindowMs: number;
  /** How long the breaker stays open before allowing one half-open trial call. */
  resetTimeoutMs: number;
  onStateChange?: (event: CircuitBreakerStateChangeEvent) => void;
}

type BreakerState = "closed" | "open" | "half-open";

/**
 * Wraps a strategy so repeated provider failures trip a breaker: once
 * `threshold` failures land inside `samplingWindowMs`, further calls fail
 * immediately with `CircuitBreakerOpenError` (no network hit) until
 * `resetTimeoutMs` elapses, at which point exactly one trial call is let
 * through to test recovery.
 */
export function withCircuitBreaker(
  strategy: LlmStrategy,
  opts: WithCircuitBreakerOptions,
): LlmStrategy {
  let state: BreakerState = "closed";
  let failureTimestamps: number[] = [];
  let openedAt = 0;
  let halfOpenTrialInFlight = false;

  function setState(next: BreakerState): void {
    if (state === next) return;
    state = next;
    opts.onStateChange?.({ providerName: strategy.providerName, state: next });
  }

  function pruneFailures(now: number): void {
    failureTimestamps = failureTimestamps.filter((t) => now - t < opts.samplingWindowMs);
  }

  function recordFailure(): void {
    const now = Date.now();
    pruneFailures(now);
    failureTimestamps.push(now);
    if (failureTimestamps.length >= opts.threshold) {
      setState("open");
      openedAt = now;
    }
  }

  return {
    ...strategy,
    async generate(genOpts: LlmGenerateOptions): Promise<LlmResponse> {
      const now = Date.now();

      if (state === "open") {
        if (now - openedAt >= opts.resetTimeoutMs) {
          setState("half-open");
        } else {
          throw new CircuitBreakerOpenError(
            strategy.providerName,
            opts.resetTimeoutMs - (now - openedAt),
          );
        }
      }

      if (state === "half-open") {
        if (halfOpenTrialInFlight) {
          throw new CircuitBreakerOpenError(strategy.providerName, opts.resetTimeoutMs);
        }
        halfOpenTrialInFlight = true;
        try {
          const response = await strategy.generate(genOpts);
          failureTimestamps = [];
          setState("closed");
          return response;
        } catch (err) {
          if (!isCallerFaultError(err) && !isAbortError(err, genOpts.signal)) {
            setState("open");
            openedAt = Date.now();
          }
          throw err;
        } finally {
          halfOpenTrialInFlight = false;
        }
      }

      try {
        return await strategy.generate(genOpts);
      } catch (err) {
        if (!isCallerFaultError(err) && !isAbortError(err, genOpts.signal)) {
          recordFailure();
        }
        throw err;
      }
    },
  };
}
