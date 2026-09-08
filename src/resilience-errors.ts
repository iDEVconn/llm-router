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
