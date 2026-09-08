import { RateLimitExceededError } from "./errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "./types";

export interface ThrottleEvent {
  providerName: string;
  waitedMs: number;
}

export interface WithRateLimitOptions {
  tokensPerSecond: number;
  maxConcurrent?: number;
  /** Max time to wait for capacity before throwing. Default 30_000. */
  maxWaitMs?: number;
  onThrottle?: (event: ThrottleEvent) => void;
}

const DEFAULT_MAX_WAIT_MS = 30_000;
const POLL_INTERVAL_MS = 25;

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (ms <= 0) {
      resolve();
      return;
    }
    if (signal?.aborted) {
      reject(signal.reason ?? new Error("Aborted"));
      return;
    }
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal!.reason ?? new Error("Aborted"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

/**
 * Wraps a strategy with a token-bucket rate limiter (capacity =
 * `max(1, tokensPerSecond)`, refilling continuously from elapsed time rather than a
 * running interval timer) plus an optional `maxConcurrent` in-flight cap.
 * A call with no available capacity blocks until capacity frees up, up to
 * `maxWaitMs`, then throws `RateLimitExceededError`.
 */
export function withRateLimit(strategy: LlmStrategy, opts: WithRateLimitOptions): LlmStrategy {
  const maxWaitMs = opts.maxWaitMs ?? DEFAULT_MAX_WAIT_MS;
  // Bucket capacity is at least 1 even when `tokensPerSecond` is fractional,
  // so a fresh limiter can always serve one call immediately; refill after
  // that is still governed by the (possibly sub-1) configured rate.
  const capacity = Math.max(1, opts.tokensPerSecond);
  let tokens = capacity;
  let lastRefill = Date.now();
  let inFlight = 0;

  function refill(): void {
    const now = Date.now();
    const elapsedSec = (now - lastRefill) / 1000;
    tokens = Math.min(capacity, tokens + elapsedSec * opts.tokensPerSecond);
    lastRefill = now;
  }

  return {
    ...strategy,
    async generate(genOpts: LlmGenerateOptions): Promise<LlmResponse> {
      const start = Date.now();

      for (;;) {
        genOpts.signal?.throwIfAborted();

        refill();
        const hasConcurrencySlot =
          opts.maxConcurrent === undefined || inFlight < opts.maxConcurrent;
        const hasToken = tokens >= 1;

        if (hasConcurrencySlot && hasToken) break;

        const waitedMs = Date.now() - start;
        if (waitedMs >= maxWaitMs) {
          throw new RateLimitExceededError(strategy.providerName, waitedMs);
        }

        await sleep(Math.min(POLL_INTERVAL_MS, maxWaitMs - waitedMs), genOpts.signal);
      }

      const waitedMs = Date.now() - start;
      if (waitedMs > 0) {
        opts.onThrottle?.({ providerName: strategy.providerName, waitedMs });
      }

      tokens -= 1;
      inFlight += 1;
      try {
        return await strategy.generate(genOpts);
      } finally {
        inFlight -= 1;
      }
    },
  };
}
