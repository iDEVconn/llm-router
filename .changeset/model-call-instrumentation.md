---
"@idevconn/llm-router": minor
---

Add `withInstrumentation(strategy, { onCall })` in `src/instrumentation.ts`, a decorator that wraps any `LlmStrategy` and emits an `LlmCallEvent` (`{ provider, model, usage, truncated, latencyMs, timestamp, error? }`) on both success and failure, without imposing a specific logger — the caller's `onCall` decides where events go. Also adds `compose(strategy, ...decorators)` so `withBudget` and `withInstrumentation` (or any other `LlmStrategy` decorator) can be chained without manual nesting: `compose(s, a, b)` behaves like `b(a(s))`.
