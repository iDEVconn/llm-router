# Resilience Decorators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three hand-rolled resilience decorators — `withRetry`, `withCircuitBreaker`, `withRateLimit` — to `@idevconn/llm-router`, matching the existing `withBudget`/`withInstrumentation` decorator shape and composable via the existing `compose()`.

**Architecture:** Each decorator is `(strategy: LlmStrategy, opts) => LlmStrategy`, holding its own state in a closure (same pattern as `withBudget`'s `totalCost`). A shared internal helper module (`src/resilience-errors.ts`) classifies which thrown errors are "the caller's fault" (never count as a circuit-breaker failure, never retried) vs. "the provider's fault" (do count, do get retried). No new runtime dependencies — every pattern (token bucket, exponential backoff with jitter, 3-state circuit breaker) is hand-rolled.

**Tech Stack:** TypeScript, vitest (`vi.useFakeTimers()` for all timing-sensitive tests — no real waits).

**Spec:** `docs/superpowers/specs/2026-09-08-resilience-decorators-design.md` — read this alongside the plan; it has the full rationale for each design decision (composition order, streaming-skip for retry, block-and-wait for rate limiting, etc.).

## Global Constraints

- No new runtime dependency — every decorator is hand-rolled.
- `withRetry` must call the wrapped strategy exactly once (no retry logic at all) when `genOpts.onToken` is set — never retry a streaming call.
- The "caller-fault" error classification (never a circuit-breaker failure, never retried) applies to exactly these 6 existing error classes: `InvalidGenerateOptionsError`, `InvalidThinkingConfigError`, `UnsupportedAttachmentError`, `UnsupportedThinkingModeError`, `UnsupportedMultiTurnError`, `BudgetExceededError`. An intentional `AbortSignal` cancellation is also never a circuit-breaker failure and never retried.
- `withRetry` additionally never retries `CircuitBreakerOpenError` or `RateLimitExceededError` (both from sibling decorators) — an open breaker or exhausted rate-limit wait fails the whole call immediately on the first attempt.
- All decorators respect `genOpts.signal`: an abort during a backoff sleep (retry) or a capacity wait (rate limit) must stop immediately, not wait out the full delay/timeout.
- All timing-based tests use `vi.useFakeTimers()` — no real `setTimeout`/wall-clock waits in the test suite.
- Every new export (3 decorators + their option/event types + 2 new error classes) goes into `src/index.ts`, same tier as `withBudget`/`withInstrumentation` (no SDK dependency).

---

## File Structure

- Create `src/resilience-errors.ts` — internal (not exported from `src/index.ts`) `isCallerFaultError`/`isAbortError` helpers.
- Create `src/__tests__/resilience-errors.test.ts`.
- Modify `src/errors.ts` — add `CircuitBreakerOpenError`, `RateLimitExceededError`, update the doc-comment table.
- Modify `src/__tests__/errors.test.ts` — append tests for the 2 new error classes.
- Create `src/circuit-breaker.ts` — `withCircuitBreaker`.
- Create `src/__tests__/circuit-breaker.test.ts`.
- Create `src/retry.ts` — `withRetry`.
- Create `src/__tests__/retry.test.ts`.
- Create `src/rate-limit.ts` — `withRateLimit`.
- Create `src/__tests__/rate-limit.test.ts`.
- Create `src/__tests__/resilience-composition.test.ts` — breaker+retry composed together.
- Modify `src/index.ts` — export all 3 decorators + types + 2 new errors.
- Modify `src/__tests__/index-exports.test.ts` — assert the new exports exist.
- Modify `README.md` — new "Resilience" section + error-mapping table update.
- Add a changeset via `npx changeset`.

---

### Task 1: Shared error-classification helper + 2 new error classes

**Files:**
- Create: `src/resilience-errors.ts`
- Create: `src/__tests__/resilience-errors.test.ts`
- Modify: `src/errors.ts` (doc table + append 2 classes)
- Modify: `src/__tests__/errors.test.ts` (append tests)

**Interfaces:**
- Produces: `export function isCallerFaultError(err: unknown): boolean`, `export function isAbortError(err: unknown, signal: AbortSignal | undefined): boolean`, `new CircuitBreakerOpenError(providerName: string, retryAfterMs: number)`, `new RateLimitExceededError(providerName: string, waitedMs: number)`.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/resilience-errors.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  BudgetExceededError,
  InvalidGenerateOptionsError,
  InvalidThinkingConfigError,
  UnsupportedAttachmentError,
  UnsupportedMultiTurnError,
  UnsupportedThinkingModeError,
} from "../errors";
import { isAbortError, isCallerFaultError } from "../resilience-errors";

describe("isCallerFaultError", () => {
  it("returns true for each caller-fault error type", () => {
    expect(isCallerFaultError(new InvalidGenerateOptionsError("x"))).toBe(true);
    expect(isCallerFaultError(new InvalidThinkingConfigError("x"))).toBe(true);
    expect(isCallerFaultError(new UnsupportedAttachmentError("p", "image/x"))).toBe(true);
    expect(isCallerFaultError(new UnsupportedThinkingModeError("p", "adaptive", []))).toBe(true);
    expect(isCallerFaultError(new UnsupportedMultiTurnError("p"))).toBe(true);
    expect(isCallerFaultError(new BudgetExceededError("perCall", 5, 1))).toBe(true);
  });

  it("returns false for a generic error", () => {
    expect(isCallerFaultError(new Error("network blip"))).toBe(false);
  });

  it("returns false for non-Error values", () => {
    expect(isCallerFaultError("just a string")).toBe(false);
    expect(isCallerFaultError(undefined)).toBe(false);
  });
});

describe("isAbortError", () => {
  it("returns false when the signal is not aborted", () => {
    const controller = new AbortController();
    expect(isAbortError(new Error("x"), controller.signal)).toBe(false);
  });

  it("returns false when signal is undefined", () => {
    expect(isAbortError(new Error("x"), undefined)).toBe(false);
  });

  it("returns true when the thrown error is the signal's abort reason", () => {
    const controller = new AbortController();
    const reason = new Error("aborted by caller");
    controller.abort(reason);
    expect(isAbortError(reason, controller.signal)).toBe(true);
  });

  it("returns true when the thrown error is a DOMException-style AbortError, even if not the exact reason object", () => {
    const controller = new AbortController();
    controller.abort();
    const abortErr = new Error("The operation was aborted");
    abortErr.name = "AbortError";
    expect(isAbortError(abortErr, controller.signal)).toBe(true);
  });

  it("returns false when the signal is aborted but the thrown error is unrelated", () => {
    const controller = new AbortController();
    controller.abort(new Error("aborted"));
    expect(isAbortError(new Error("unrelated failure"), controller.signal)).toBe(false);
  });
});
```

Append to `src/__tests__/errors.test.ts` (imports already exist near the top — add to the existing `import { ... } from "../errors";` block, keeping it at the top of the file):

```ts
describe("CircuitBreakerOpenError", () => {
  it("names the provider and the retry-after estimate", () => {
    const err = new CircuitBreakerOpenError("claude", 15_000);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("CircuitBreakerOpenError");
    expect(err.providerName).toBe("claude");
    expect(err.retryAfterMs).toBe(15_000);
    expect(err.message).toMatch(/claude/);
  });
});

describe("RateLimitExceededError", () => {
  it("names the provider and how long it waited", () => {
    const err = new RateLimitExceededError("gemini", 30_000);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("RateLimitExceededError");
    expect(err.providerName).toBe("gemini");
    expect(err.waitedMs).toBe(30_000);
    expect(err.message).toMatch(/gemini/);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/resilience-errors.test.ts src/__tests__/errors.test.ts`
Expected: FAIL — `Cannot find module '../resilience-errors'`, and `CircuitBreakerOpenError`/`RateLimitExceededError` not exported from `../errors`.

- [ ] **Step 3: Implement**

Create `src/resilience-errors.ts`:

```ts
import {
  BudgetExceededError,
  InvalidGenerateOptionsError,
  InvalidThinkingConfigError,
  UnsupportedAttachmentError,
  UnsupportedMultiTurnError,
  UnsupportedThinkingModeError,
} from "./errors";

/**
 * True for errors that mean "this call was never going to succeed,
 * regardless of provider or attempt count" — caller-input or config
 * problems, not the provider misbehaving. Used by `withCircuitBreaker`
 * (these never count as a provider failure) and `withRetry` (these are
 * never retried).
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

/**
 * True when `err` is the intentional result of `signal` being aborted —
 * never the provider's fault, never a circuit-breaker failure, never
 * retried.
 */
export function isAbortError(err: unknown, signal: AbortSignal | undefined): boolean {
  if (!signal?.aborted) return false;
  if (err === signal.reason) return true;
  return err instanceof Error && err.name === "AbortError";
}
```

Update the doc-comment table at the top of `src/errors.ts` — replace:

```
 *   - `InvalidGenerateOptionsError`  → 400
 *   - `UnsupportedMultiTurnError`    → 400
 */
```

with:

```
 *   - `InvalidGenerateOptionsError`  → 400
 *   - `UnsupportedMultiTurnError`    → 400
 *   - `CircuitBreakerOpenError`      → 503 (breaker open; provider skipped)
 *   - `RateLimitExceededError`       → 429
 */
```

Append at the end of `src/errors.ts`:

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

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/resilience-errors.test.ts src/__tests__/errors.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/resilience-errors.ts src/__tests__/resilience-errors.test.ts src/errors.ts src/__tests__/errors.test.ts
git commit -m "feat: add caller-fault/abort error classification helper, CircuitBreakerOpenError, RateLimitExceededError"
```

---

### Task 2: `withCircuitBreaker`

**Files:**
- Create: `src/circuit-breaker.ts`
- Create: `src/__tests__/circuit-breaker.test.ts`

**Interfaces:**
- Consumes: `isCallerFaultError`/`isAbortError` (Task 1), `CircuitBreakerOpenError` (Task 1).
- Produces: `export function withCircuitBreaker(strategy: LlmStrategy, opts: WithCircuitBreakerOptions): LlmStrategy`, `export interface WithCircuitBreakerOptions { threshold: number; samplingWindowMs: number; resetTimeoutMs: number; onStateChange?: (event: CircuitBreakerStateChangeEvent) => void }`, `export interface CircuitBreakerStateChangeEvent { providerName: string; state: "closed" | "open" | "half-open" }`.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/circuit-breaker.test.ts`:

```ts
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/circuit-breaker.test.ts`
Expected: FAIL — `Cannot find module '../circuit-breaker'`.

- [ ] **Step 3: Implement**

Create `src/circuit-breaker.ts`:

```ts
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/circuit-breaker.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/circuit-breaker.ts src/__tests__/circuit-breaker.test.ts
git commit -m "feat: add withCircuitBreaker decorator"
```

---

### Task 3: `withRetry`

**Files:**
- Create: `src/retry.ts`
- Create: `src/__tests__/retry.test.ts`

**Interfaces:**
- Consumes: `isCallerFaultError`/`isAbortError` (Task 1), `CircuitBreakerOpenError`/`RateLimitExceededError` (Task 1).
- Produces: `export function withRetry(strategy: LlmStrategy, opts: WithRetryOptions): LlmStrategy`, `export interface WithRetryOptions { maxAttempts: number; initialBackoffMs: number; maxBackoffMs: number; multiplier: number; jitter?: boolean; onRetry?: (event: RetryEvent) => void }`, `export interface RetryEvent { providerName: string; attempt: number; error: unknown; delayMs: number }`.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/retry.test.ts`:

```ts
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
    await vi.runAllTimersAsync();

    await expect(promise).rejects.toThrow("always fails");
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/retry.test.ts`
Expected: FAIL — `Cannot find module '../retry'`.

- [ ] **Step 3: Implement**

Create `src/retry.ts`:

```ts
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/retry.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retry.ts src/__tests__/retry.test.ts
git commit -m "feat: add withRetry decorator"
```

---

### Task 4: `withRateLimit`

**Files:**
- Create: `src/rate-limit.ts`
- Create: `src/__tests__/rate-limit.test.ts`

**Interfaces:**
- Consumes: `RateLimitExceededError` (Task 1).
- Produces: `export function withRateLimit(strategy: LlmStrategy, opts: WithRateLimitOptions): LlmStrategy`, `export interface WithRateLimitOptions { tokensPerSecond: number; maxConcurrent?: number; maxWaitMs?: number; onThrottle?: (event: ThrottleEvent) => void }`, `export interface ThrottleEvent { providerName: string; waitedMs: number }`.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/rate-limit.test.ts`:

```ts
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
    await vi.advanceTimersByTimeAsync(1_100);
    await expect(promise).rejects.toBeInstanceOf(RateLimitExceededError);
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/rate-limit.test.ts`
Expected: FAIL — `Cannot find module '../rate-limit'`.

- [ ] **Step 3: Implement**

Create `src/rate-limit.ts`:

```ts
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

/**
 * Wraps a strategy with a token-bucket rate limiter (capacity =
 * `tokensPerSecond`, refilling continuously from elapsed time rather than a
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/rate-limit.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/rate-limit.ts src/__tests__/rate-limit.test.ts
git commit -m "feat: add withRateLimit decorator"
```

---

### Task 5: Composition test (circuit breaker + retry)

**Files:**
- Create: `src/__tests__/resilience-composition.test.ts`

**Interfaces:**
- Consumes: `withCircuitBreaker` (Task 2), `withRetry` (Task 3), `compose` (already exists in `src/instrumentation.ts`), `CircuitBreakerOpenError` (Task 1).

- [ ] **Step 1: Write the test**

Create `src/__tests__/resilience-composition.test.ts`:

```ts
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
```

- [ ] **Step 2: Run to verify it passes**

Run: `npx vitest run src/__tests__/resilience-composition.test.ts`
Expected: PASS (this test only exercises Tasks 1-3's already-implemented code, so it should pass immediately — no separate implementation step).

- [ ] **Step 3: Commit**

```bash
git add src/__tests__/resilience-composition.test.ts
git commit -m "test: verify withCircuitBreaker + withRetry compose correctly"
```

---

### Task 6: Exports

**Files:**
- Modify: `src/index.ts`
- Modify: `src/__tests__/index-exports.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/__tests__/index-exports.test.ts` inside `describe("public exports", ...)`, before the final closing `});`. First add these names to the existing `import { ... } from "../index";` block at the top of the file (merge alphabetically with what's already imported): `CircuitBreakerOpenError`, `RateLimitExceededError`, `withCircuitBreaker`, `withRateLimit`, `withRetry`.

```ts
  it("exports the resilience decorators and their errors", () => {
    expect(typeof withRetry).toBe("function");
    expect(typeof withCircuitBreaker).toBe("function");
    expect(typeof withRateLimit).toBe("function");
    expect(typeof CircuitBreakerOpenError).toBe("function");
    expect(typeof RateLimitExceededError).toBe("function");
  });
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/index-exports.test.ts`
Expected: FAIL — the 5 new names aren't exported from `../index` yet.

- [ ] **Step 3: Implement**

In `src/index.ts`, add to the error export block (merge alphabetically):

```ts
export {
  BudgetExceededError,
  CircuitBreakerOpenError,
  InvalidGenerateOptionsError,
  InvalidPlatformProviderError,
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  NoAvailableProviderError,
  NoPlatformProviderError,
  RateLimitExceededError,
  TaskDecompositionError,
  UnknownProviderError,
  UnsupportedAttachmentError,
  UnsupportedMultiTurnError,
  UnsupportedThinkingModeError,
} from "./errors";
```

Add three new export lines right after the existing `export { compose, withInstrumentation } from "./instrumentation";` / `export type { LlmCallEvent, WithInstrumentationOptions } from "./instrumentation";` pair:

```ts
export { withCircuitBreaker } from "./circuit-breaker";
export type { CircuitBreakerStateChangeEvent, WithCircuitBreakerOptions } from "./circuit-breaker";
export { withRetry } from "./retry";
export type { RetryEvent, WithRetryOptions } from "./retry";
export { withRateLimit } from "./rate-limit";
export type { ThrottleEvent, WithRateLimitOptions } from "./rate-limit";
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/index-exports.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/index.ts src/__tests__/index-exports.test.ts
git commit -m "feat: export resilience decorators and their errors from index"
```

---

### Task 7: README updates

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Insert a new "Resilience" section between "## Instrumentation" and "## Prompt injection defense"**

Read the current `README.md` to confirm the exact surrounding text still matches (it was last touched by an earlier, unrelated plan), then insert this new section immediately after the existing "## Instrumentation" section's closing (`compose(s, a, b)` explanation paragraph) and before the "## Prompt injection defense" heading:

```md
## Resilience: retry, circuit breaker, rate limiting

Three more `compose()`-able decorators, same shape as `withBudget`/`withInstrumentation`. No new runtime dependency — all three are hand-rolled.

```ts
import { compose, withCircuitBreaker, withRateLimit, withRetry } from "@idevconn/llm-router";

const resilient = compose(
  strategy,
  (s) => withCircuitBreaker(s, { threshold: 5, samplingWindowMs: 60_000, resetTimeoutMs: 30_000 }),
  (s) => withRateLimit(s, { tokensPerSecond: 10, maxConcurrent: 5 }),
  (s) => withRetry(s, { maxAttempts: 3, initialBackoffMs: 200, maxBackoffMs: 2_000, multiplier: 2 }),
);
```

Put `withRetry` outermost (as above) — each retry attempt then re-enters the rate-limiter and circuit-breaker layers, so an open breaker or an exhausted rate-limit wait fails the whole call fast on the first attempt instead of wasting the remaining retry attempts.

- **`withCircuitBreaker`** — after `threshold` failures land inside the rolling `samplingWindowMs`, further calls throw `CircuitBreakerOpenError` immediately (no network call) until `resetTimeoutMs` elapses, then lets exactly one trial call through to test recovery. Caller-fault errors (`InvalidGenerateOptionsError`, `InvalidThinkingConfigError`, `UnsupportedAttachmentError`, `UnsupportedThinkingModeError`, `UnsupportedMultiTurnError`, `BudgetExceededError`) and an intentional `signal` abort never count as a failure.
- **`withRetry`** — exponential backoff (`initialBackoffMs * multiplier^attempt`, capped at `maxBackoffMs`, jittered to 0.5-1.0x by default). Never retries a streaming call (`onToken` set) — a mid-stream failure means some tokens already reached the caller, and retrying would re-emit them from the start. Never retries the same caller-fault errors as above, nor `CircuitBreakerOpenError`/`RateLimitExceededError`, nor an aborted `signal`.
- **`withRateLimit`** — token bucket (`tokensPerSecond`, refilling continuously) plus an optional `maxConcurrent` in-flight cap. A call with no capacity blocks until capacity frees up, up to `maxWaitMs` (default 30s), then throws `RateLimitExceededError`.

All three respect `signal` — an abort during a backoff sleep or a capacity wait stops immediately rather than waiting out the full delay.
```

- [ ] **Step 2: Update the error-mapping example**

In the "## Error mapping" section, add two lines to the existing NestJS example, right after the `BudgetExceededError` line:

Replace:

```ts
  if (err instanceof BudgetExceededError) throw new HttpException(err.message, 402);
  throw err;
```

with:

```ts
  if (err instanceof BudgetExceededError) throw new HttpException(err.message, 402);
  if (err instanceof CircuitBreakerOpenError) throw new HttpException(err.message, 503);
  if (err instanceof RateLimitExceededError) throw new HttpException(err.message, 429);
  throw err;
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document resilience decorators (retry, circuit breaker, rate limit)"
```

---

### Task 8: Changeset

**Files:**
- Create: `.changeset/<auto-generated-name>.md`

- [ ] **Step 1: Create the changeset**

Run `npx changeset add --empty`, then edit the generated file to:

```md
---
"@idevconn/llm-router": minor
---

Add `withRetry` (exponential backoff), `withCircuitBreaker` (per-strategy auto-recovery), and `withRateLimit` (token bucket) decorators — hand-rolled, no new runtime dependencies. New `CircuitBreakerOpenError` and `RateLimitExceededError` typed errors. All three compose with the existing `withBudget`/`withInstrumentation` via `compose()`.
```

- [ ] **Step 2: Verify and commit**

Run `cat .changeset/*.md` to confirm, then:

```bash
git add .changeset/
git commit -m "chore: add changeset for resilience decorators"
```

---

### Task 9: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Typecheck**

Run: `npm run typecheck`
Expected: PASS, zero errors.

- [ ] **Step 2: Lint**

Run: `npm run lint`
Expected: PASS, zero errors (the one pre-existing unrelated warning in `claude.test.ts` is fine).

- [ ] **Step 3: Full test suite**

Run: `npm test`
Expected: PASS — every existing test plus all new tests from Tasks 1-6, unmodified pre-existing tests still green.

- [ ] **Step 4: Build**

Run: `npm run build`
Expected: PASS.

- [ ] **Step 5: Final status check**

Run: `git status`
Expected: clean (everything already committed per-task).
