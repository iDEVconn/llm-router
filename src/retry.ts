import { CircuitBreakerOpenError, RateLimitExceededError } from "./errors";
import { isAbortError, isCallerFaultError } from "./resilience-errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "./types";

export interface RetryEvent {
  providerName: string;
  /** 1-based: which attempt just failed. */
  attempt: number;
  error: unknown;
  /** How long before the next attempt. */
  delayMs: number;
}

export interface WithRetryOptions {
  maxAttempts: number;
  initialBackoffMs: number;
  maxBackoffMs: number;
  multiplier: number;
  /** Randomize each computed delay to 0.5x-1.0x of its base value. Default true. */
  jitter?: boolean;
  onRetry?: (event: RetryEvent) => void;
}

function sleep(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new Error("Aborted"));
      return;
    }
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason ?? new Error("Aborted"));
      },
      { once: true },
    );
  });
}

function isRetryable(err: unknown, signal: AbortSignal | undefined): boolean {
  if (isCallerFaultError(err)) return false;
  if (isAbortError(err, signal)) return false;
  if (err instanceof CircuitBreakerOpenError) return false;
  if (err instanceof RateLimitExceededError) return false;
  return true;
}

/**
 * Wraps a strategy with exponential-backoff retry. Never retries a
 * streaming call (`genOpts.onToken` set) — a mid-stream failure means some
 * tokens already reached the caller, and retrying would re-emit them from
 * the start. Never retries caller-fault errors, an open circuit breaker, an
 * exhausted rate limit, or an intentional abort — only genuine
 * provider-side failures are retried.
 */
export function withRetry(strategy: LlmStrategy, opts: WithRetryOptions): LlmStrategy {
  const jitter = opts.jitter ?? true;

  return {
    ...strategy,
    async generate(genOpts: LlmGenerateOptions): Promise<LlmResponse> {
      if (genOpts.onToken) {
        return strategy.generate(genOpts);
      }

      let attempt = 0;
      for (;;) {
        attempt += 1;
        try {
          return await strategy.generate(genOpts);
        } catch (err) {
          const isLastAttempt = attempt >= opts.maxAttempts;
          if (isLastAttempt || !isRetryable(err, genOpts.signal)) {
            throw err;
          }

          const rawDelay = Math.min(
            opts.initialBackoffMs * Math.pow(opts.multiplier, attempt - 1),
            opts.maxBackoffMs,
          );
          const delayMs = jitter ? rawDelay * (0.5 + Math.random() * 0.5) : rawDelay;

          opts.onRetry?.({
            providerName: strategy.providerName,
            attempt,
            error: err,
            delayMs,
          });

          await sleep(delayMs, genOpts.signal);
        }
      }
    },
  };
}
