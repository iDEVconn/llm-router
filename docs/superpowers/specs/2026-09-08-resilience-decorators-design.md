# Resilience Decorators — Design Spec

**Sub-project A of 4** in the "resilience & routing" initiative (this repo's `@idevconn/llm-router`). The other three — load-balanced provider pool, tier resolution, cache cost analytics — are separate specs, built after this one lands.

## Motivation

Researched two competing packages (`llm-router` on npm by vikaskumar, and `@kb-labs/llm-router`) for prior art. Both offer resilience patterns this package lacks entirely today: if a provider call fails, throws a 5xx, or the caller is about to exceed a rate limit, `@idevconn/llm-router` has no built-in way to retry, back off, circuit-break, or rate-limit — callers must roll their own. This spec adds three decorators that close that gap, following the exact shape of the two decorators that already exist (`withBudget`, `withInstrumentation` in `src/budget.ts`/`src/instrumentation.ts`).

## Non-goals

- No new runtime dependency (`cockatiel` etc.) — hand-rolled, per user decision. Rationale: keeps the package's "zero SDK dependency on core" philosophy intact; each pattern is small enough (~40-80 lines) that the maintenance cost of owning it is lower than the cost of a dependency whose error types and edge-case behavior we'd have to work around anyway.
- No load-balancing / multi-instance pool (that's sub-project B).
- No change to any provider adapter (`src/claude/index.ts` etc.) — these decorators wrap `LlmStrategy` from the outside, exactly like `withBudget` does today.
- No weighting of rate limits by actual token cost — `withRateLimit` consumes 1 bucket token per *request*, not per output token, because token cost is only known after the call completes (same simplification the `llm-router` npm package uses).

## Architecture

All three decorators share the existing shape:

```ts
function withX(strategy: LlmStrategy, opts: WithXOptions): LlmStrategy
```

Composed via the existing `compose()` in `src/instrumentation.ts` — no changes to `compose()` itself are needed, it's already decorator-count-agnostic.

**Recommended composition order** (documented in each decorator's doc comment and in the README, not enforced by code):

```ts
const resilient = compose(
  baseStrategy,
  withCircuitBreaker({ threshold: 5, samplingWindowMs: 60_000, resetTimeoutMs: 30_000 }),
  withRateLimit({ tokensPerSecond: 10, maxConcurrent: 5 }),
  withRetry({ maxAttempts: 3, initialBackoffMs: 200, maxBackoffMs: 2_000, multiplier: 2 }),
);
```

`compose(s, a, b, c)` behaves like `c(b(a(s)))` (existing semantics, unchanged) — so `withRetry` is outermost here. Each retry attempt re-enters the rate-limiter and circuit-breaker layers, meaning: an open breaker or an exhausted rate-limit wait fails fast on the *first* attempt (both errors are excluded from what `withRetry` considers retryable — see below), without wasting the remaining retry attempts hammering a provider that's already known to be down or throttled.

## Shared error-classification list

Three of the four new/existing pieces (circuit breaker's failure counting, retry's retryable-error check) need to agree on "is this the provider's fault, or the caller's/config's fault." Define one shared list, exported from a new small module `src/resilience-errors.ts`:

```ts
// src/resilience-errors.ts
import {
  BudgetExceededError,
  InvalidGenerateOptionsError,
  InvalidThinkingConfigError,
  UnsupportedAttachmentError,
  UnsupportedMultiTurnError,
  UnsupportedThinkingModeError,
} from "./errors";

/**
 * Errors that mean "this call was never going to succeed, regardless of
 * which provider or how many times you tried" — caller-input or config
 * problems, not the provider misbehaving. Shared by `withCircuitBreaker`
 * (these never count as a provider failure) and `withRetry` (these are
 * never retried). `withRetry` additionally excludes `CircuitBreakerOpenError`
 * and `RateLimitExceededError` (both defined in `./errors`, imported directly
 * — no circular import, since neither error class depends on this module).
 */
export function isCallerFaultError(err: unknown): boolean {
  return (
    err instanceof InvalidGenerateOptionsError ||
    err instanceof InvalidThinkingConfigError ||
    err instanceof UnsupportedAttachmentError ||
    err instanceof UnsupportedThinkingModeError ||
    err instanceof UnsupportedMultiTurnError ||
    err instanceof BudgetExceededError
  );
}

/** True when `err` is (or wraps) an intentional AbortSignal cancellation — never the provider's fault, never retried. */
export function isAbortError(err: unknown, signal: AbortSignal | undefined): boolean {
  if (!signal?.aborted) return false;
  if (err === signal.reason) return true;
  return err instanceof Error && err.name === "AbortError";
}
```

`withRetry`'s own module additionally excludes `CircuitBreakerOpenError` and `RateLimitExceededError` by `instanceof` check, importing both directly from `./errors` (where they're defined — no cycle).

## Component 1: `withCircuitBreaker`

New file `src/circuit-breaker.ts`.

```ts
export interface CircuitBreakerStateChangeEvent {
  providerName: string;
  state: "closed" | "open" | "half-open";
}

export interface WithCircuitBreakerOptions {
  /** Failures within the rolling window before the breaker opens (not required to be consecutive — a success in between does not reset the count, only time-based pruning does; this matches standard circuit-breaker semantics of "failure rate within a window", not "consecutive streak"). */
  threshold: number;
  /** Rolling window (ms) failures are counted over. */
  samplingWindowMs: number;
  /** How long the breaker stays open before allowing one half-open trial call. */
  resetTimeoutMs: number;
  onStateChange?: (event: CircuitBreakerStateChangeEvent) => void;
}

export function withCircuitBreaker(
  strategy: LlmStrategy,
  opts: WithCircuitBreakerOptions,
): LlmStrategy;
```

**State machine** (per wrapped-strategy-instance, closure-held state — same pattern as `withBudget`'s `totalCost`):

- **closed**: calls pass through. Each failure (per `isCallerFaultError`/`isAbortError`-filtered classification above) is timestamped and pushed onto a failure-timestamp list; timestamps older than `samplingWindowMs` are pruned on each check. If the pruned list's length reaches `threshold`, transition to **open** and fire `onStateChange`.
- **open**: every call immediately throws `CircuitBreakerOpenError` (see below) without calling the wrapped strategy. When `resetTimeoutMs` has elapsed since opening, the *next* call transitions to **half-open** first (fires `onStateChange`), then proceeds as the trial call.
- **half-open**: exactly one call is let through (a mutex flag prevents concurrent calls from all becoming "the trial" — while a trial is in flight, further concurrent calls still throw `CircuitBreakerOpenError` as if open). If the trial succeeds, transition to **closed** (clear failure list, fire `onStateChange`). If it fails, transition back to **open** (reset the `resetTimeoutMs` timer, fire `onStateChange`).

New error, appended to `src/errors.ts`:

```ts
export class CircuitBreakerOpenError extends Error {
  constructor(
    public readonly providerName: string,
    public readonly retryAfterMs: number,
  ) {
    super(`Circuit breaker for "${providerName}" is open; retry after ~${retryAfterMs}ms.`);
    this.name = "CircuitBreakerOpenError";
  }
}
```

Doc-table entry: `CircuitBreakerOpenError → 503`.

## Component 2: `withRetry`

New file `src/retry.ts`.

```ts
export interface RetryEvent {
  providerName: string;
  attempt: number;      // 1-based: which attempt just failed
  error: unknown;
  delayMs: number;       // how long before the next attempt
}

export interface WithRetryOptions {
  maxAttempts: number;
  initialBackoffMs: number;
  maxBackoffMs: number;
  multiplier: number;
  /** Randomize each computed delay by 0-100%. Default true. */
  jitter?: boolean;
  onRetry?: (event: RetryEvent) => void;
}

export function withRetry(strategy: LlmStrategy, opts: WithRetryOptions): LlmStrategy;
```

**Behavior:**

1. If `genOpts.onToken` is set: call `strategy.generate(genOpts)` exactly once, no retry logic — return/throw whatever it does. (Streaming calls are never retried — see Motivation/prior brainstorming: a mid-stream failure means some tokens already reached the caller's `onToken`, and retrying would re-emit them from the start.)
2. Otherwise, loop up to `maxAttempts` times:
   - Call `strategy.generate(genOpts)`. On success, return immediately.
   - On failure: if `isCallerFaultError(err)`, or `isAbortError(err, genOpts.signal)`, or `err instanceof CircuitBreakerOpenError`, or `err instanceof RateLimitExceededError` — re-throw immediately, no further attempts.
   - Otherwise, if this was the last allowed attempt, re-throw the error (no wrapping).
   - Otherwise: compute `delay = min(initialBackoffMs * multiplier^(attempt-1), maxBackoffMs)`, apply jitter if enabled (`delay * (0.5 + Math.random() * 0.5)`, so jittered delay is 50-100% of the computed value — avoids the thundering-herd risk of literal 0-100% jitter sometimes producing a ~0ms retry), fire `onRetry`, then sleep for `delay` — but if `genOpts.signal` fires during the sleep, abort the sleep and re-throw the abort reason immediately rather than completing the wait.

## Component 3: `withRateLimit`

New file `src/rate-limit.ts`.

```ts
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

export function withRateLimit(strategy: LlmStrategy, opts: WithRateLimitOptions): LlmStrategy;
```

**Behavior:** classic token bucket, capacity `tokensPerSecond`, refilling continuously (computed from elapsed-time-since-last-refill rather than a running interval timer, to avoid keeping a timer handle alive/needing cleanup). Each call:

1. If `maxConcurrent` is set and already at that many in-flight calls, wait for a slot.
2. Wait for the bucket to have ≥1 token available (compute wait time from the refill rate rather than polling).
3. Both waits are bounded by a single overall `maxWaitMs` budget for the call; if exceeded, throw `RateLimitExceededError` (below) without ever calling the wrapped strategy. Fire `onThrottle` whenever any non-zero wait happened (whether it succeeded or timed out).
4. Respect `genOpts.signal` — abort the wait immediately (not just at `maxWaitMs`) if the signal fires.
5. Once through both gates, consume 1 token, increment the in-flight counter, call the wrapped strategy, and decrement the in-flight counter in a `finally` (so a thrown error still frees the concurrency slot).

New error, appended to `src/errors.ts`:

```ts
export class RateLimitExceededError extends Error {
  constructor(
    public readonly providerName: string,
    public readonly waitedMs: number,
  ) {
    super(`Rate limit for "${providerName}" exceeded; waited ${waitedMs}ms with no capacity.`);
    this.name = "RateLimitExceededError";
  }
}
```

Doc-table entry: `RateLimitExceededError → 429`.

## Exports

All three decorators + their option/event types, plus the two new error classes, exported top-level from `src/index.ts` (same tier as `withBudget`/`withInstrumentation` — no SDK dependency, always available). `isCallerFaultError`/`isAbortError` from `src/resilience-errors.ts` are NOT exported publicly (internal implementation detail shared between the three decorator modules).

## Testing

- `vi.useFakeTimers()` throughout — no real `setTimeout`/wall-clock waits in any test.
- `src/__tests__/circuit-breaker.test.ts`: closed→open (threshold reached within window, not reached outside window — test the pruning), open→half-open (after `resetTimeoutMs`), half-open success→closed, half-open failure→open (timer resets), concurrent calls during half-open (only one becomes the trial, others get `CircuitBreakerOpenError`), caller-fault errors and abort don't count toward the failure threshold.
- `src/__tests__/retry.test.ts`: succeeds on Nth attempt, exhausts and rethrows the *original* error type, skips retry entirely for each caller-fault error type + `CircuitBreakerOpenError` + `RateLimitExceededError` + abort, skips ALL retry logic when `onToken` is set (single call, no backoff), exponential backoff timing (advance fake timers, assert call counts at expected times), jitter bounds (0.5-1.0× the base delay), signal-abort during a backoff sleep stops immediately.
- `src/__tests__/rate-limit.test.ts`: bucket refill timing, `maxConcurrent` gating (Nth+1 concurrent call waits), `maxWaitMs` timeout throws `RateLimitExceededError`, signal-abort during wait, `onThrottle` fires only when an actual wait happened.
- `src/__tests__/resilience-composition.test.ts` (new, small): breaker + retry composed together — once breaker opens, subsequent `generate()` calls (that would otherwise retry) fail fast with exactly 1 call reaching the mock strategy (not `maxAttempts`), proving the exclusion-list wiring works end-to-end across the two modules.

## README updates

New "## Resilience: retry, circuit breaker, rate limiting" section (placed after "## Cost control" or wherever the existing `withBudget` section lives — read the current README structure at implementation time to match), documenting all three decorators with the composition-order example from Architecture above, and the two new errors added to the existing error-mapping table/example.

## Changeset

Minor bump: "Add withRetry, withCircuitBreaker, and withRateLimit decorators (hand-rolled, no new dependencies) for resilient provider calls; new CircuitBreakerOpenError and RateLimitExceededError typed errors."
