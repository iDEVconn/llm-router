---
"@idevconn/llm-router": minor
---

Add `withRetry` (exponential backoff), `withCircuitBreaker` (per-strategy auto-recovery), and `withRateLimit` (token bucket) decorators — hand-rolled, no new runtime dependencies. New `CircuitBreakerOpenError` and `RateLimitExceededError` typed errors. All three compose with the existing `withBudget`/`withInstrumentation` via `compose()`.

