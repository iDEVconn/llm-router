/**
 * Plain `Error` subclasses so the pkg stays framework-agnostic.
 * Consumers (NestJS, Express, Fastify, …) wrap these at the controller
 * boundary to map onto the right HTTP status:
 *   - `UnknownProviderError`         → 404
 *   - `NoPlatformProviderError`      → 400
 *   - `InvalidPlatformProviderError` → throw at boot (config error)
 *   - `LlmKeyValidationError`        → 400 / 401 depending on caller intent
 *   - `UnsupportedAttachmentError`   → 400
 *   - `TaskDecompositionError`       → 502 (upstream model didn't cooperate)
 *   - `NoAvailableProviderError`     → 400 (no BYOK/platform key for the routed provider)
 *   - `BudgetExceededError`         → 402 (cost budget exhausted)
 *   - `UnsupportedThinkingModeError` → 400
 *   - `InvalidThinkingConfigError`   → 400
 *   - `InvalidGenerateOptionsError`  → 400
 *   - `UnsupportedMultiTurnError`    → 400
 *   - `CircuitBreakerOpenError`      → 503 (breaker open; provider skipped)
 *   - `RateLimitExceededError`       → 429
 *   - `LlmAbortedError`              → caller cancelled; usually no response
 *                                      (persist `partialText`/`usage` instead)
 */

import type { LlmUsage } from "./types";

export class UnknownProviderError extends Error {
  constructor(public readonly providerName: string) {
    super(`Unknown LLM provider: "${providerName}"`);
    this.name = "UnknownProviderError";
  }
}

export class NoPlatformProviderError extends Error {
  constructor() {
    super(
      "No platform LLM provider is configured. Either set the platform name explicitly or have callers supply a BYOK key.",
    );
    this.name = "NoPlatformProviderError";
  }
}

export class InvalidPlatformProviderError extends Error {
  constructor(
    public readonly providerName: string,
    public readonly knownProviders: readonly string[],
  ) {
    super(
      `Invalid platform LLM provider "${providerName}". Expected one of: ${knownProviders.join(", ")}.`,
    );
    this.name = "InvalidPlatformProviderError";
  }
}

export class LlmKeyValidationError extends Error {
  constructor(
    public readonly providerName: string,
    public override readonly cause: unknown,
  ) {
    const causeMessage = cause instanceof Error ? cause.message : String(cause);
    super(`${providerName} rejected the API key: ${causeMessage}`);
    this.name = "LlmKeyValidationError";
  }
}

export class UnsupportedAttachmentError extends Error {
  constructor(
    public readonly providerName: string,
    public readonly mimetype: string,
    public readonly hint?: string,
  ) {
    super(
      `${providerName} does not accept "${mimetype}".${hint ? ` ${hint}` : ""}`,
    );
    this.name = "UnsupportedAttachmentError";
  }
}

export class TaskDecompositionError extends Error {
  constructor(public override readonly cause: unknown) {
    const causeMessage = cause instanceof Error ? cause.message : String(cause);
    super(`Failed to decompose the task into subtasks: ${causeMessage}`);
    this.name = "TaskDecompositionError";
  }
}

export class NoAvailableProviderError extends Error {
  constructor(public readonly subtaskId: string) {
    super(`No available provider (platform key or BYOK) can handle subtask "${subtaskId}".`);
    this.name = "NoAvailableProviderError";
  }
}

export class BudgetExceededError extends Error {
  constructor(
    public readonly kind: "perCall" | "total",
    public readonly cost: number,
    public readonly limit: number,
  ) {
    super(
      kind === "perCall"
        ? `Call cost ${cost} exceeds maxCostPerCall ${limit}.`
        : `Accumulated cost ${cost} has reached maxCostTotal ${limit}; refusing further calls.`,
    );
    this.name = "BudgetExceededError";
  }
}

export class UnsupportedThinkingModeError extends Error {
  constructor(
    public readonly provider: string,
    public readonly requested: string,
    public readonly supported: readonly string[],
  ) {
    super(
      `${provider} does not support thinking mode "${requested}"` +
        (supported.length ? ` (supports: ${supported.join(", ")})` : " (no thinking support)"),
    );
    this.name = "UnsupportedThinkingModeError";
  }
}

export class InvalidThinkingConfigError extends Error {
  constructor(public readonly reason: string) {
    super(`Invalid thinking configuration: ${reason}`);
    this.name = "InvalidThinkingConfigError";
  }
}

export class InvalidGenerateOptionsError extends Error {
  constructor(reason: string) {
    super(`Invalid LlmGenerateOptions: ${reason}`);
    this.name = "InvalidGenerateOptionsError";
  }
}

export class UnsupportedMultiTurnError extends Error {
  constructor(public readonly providerName: string) {
    super(
      `${providerName} does not support multi-turn \`messages\`. Pass a single-turn \`prompt\` instead, or switch to a provider that supports multi-turn.`,
    );
    this.name = "UnsupportedMultiTurnError";
  }
}

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

export interface LlmAbortedErrorInit {
  providerName: string;
  partialText: string;
  partialThinking?: string;
  usage: LlmUsage;
  usageEstimated: boolean;
  cause: unknown;
}

const LLM_ABORTED_ERROR_BRAND = Symbol.for("@idevconn/llm-router/LlmAbortedError");

/**
 * Thrown by a strategy when a STREAMING call (`onToken` set) is aborted via
 * `signal`, carrying everything received before the abort so the caller can
 * persist the partial answer and account for the tokens spent.
 *
 * `name` is deliberately `"AbortError"` (not the class name) so existing
 * `err.name === "AbortError"` checks — including `isAbortError`, which keeps
 * `withRetry` / `withCircuitBreaker` from retrying an abort or counting it as
 * a provider failure — keep recognising it. Use `instanceof LlmAbortedError`
 * for the typed check.
 *
 *  - `partialText`: concatenation of every text delta delivered to `onToken`.
 *  - `partialThinking`: reasoning deltas received, for strategies that stream
 *    them (undefined otherwise).
 *  - `usage` / `usageEstimated`: provider-reported counts when the provider
 *    sent both sides mid-stream (`usageEstimated: false`); otherwise the
 *    missing side is a rough `chars / 4` estimate (`usageEstimated: true`).
 *  - `cause`: the original provider/abort error.
 */
export class LlmAbortedError extends Error {
  /**
   * Brand-based `instanceof`: the CJS build inlines this class separately
   * into the root entry and every provider subpath (`/claude`, `/gemini`,
   * …), so an error thrown by `@idevconn/llm-router/claude` would otherwise
   * fail `instanceof LlmAbortedError` imported from `@idevconn/llm-router`
   * (same for an ESM/CJS dual-package mix). A global-registry symbol makes
   * every copy of the class recognise every other copy's instances.
   * The brand is set in this constructor, so subclasses inherit both it and
   * this `instanceof` behaviour (a subclass instance is also an
   * `LlmAbortedError`, as with ordinary prototype-based `instanceof`).
   */
  static override [Symbol.hasInstance](value: unknown): boolean {
    return (
      typeof value === "object" &&
      value !== null &&
      (value as Record<symbol, unknown>)[LLM_ABORTED_ERROR_BRAND] === true
    );
  }

  public readonly providerName: string;
  public readonly partialText: string;
  public readonly partialThinking?: string;
  public readonly usage: LlmUsage;
  public readonly usageEstimated: boolean;
  public override readonly cause: unknown;

  constructor(init: LlmAbortedErrorInit) {
    super(`${init.providerName} request aborted (${init.partialText.length} chars received)`);
    this.name = "AbortError";
    this.providerName = init.providerName;
    this.partialText = init.partialText;
    this.partialThinking = init.partialThinking;
    this.usage = init.usage;
    this.usageEstimated = init.usageEstimated;
    this.cause = init.cause;
    Object.defineProperty(this, LLM_ABORTED_ERROR_BRAND, { value: true });
  }
}
