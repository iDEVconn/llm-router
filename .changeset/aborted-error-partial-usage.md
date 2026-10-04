---
"@idevconn/llm-router": minor
---

Add `LlmAbortedError`: when a streaming `generate()` call (`onToken` set) is aborted via `signal`, all five built-in strategies (Claude, Gemini, ChatGPT, Grok, DeepSeek) now reject with an `LlmAbortedError` that carries what was received so far — `partialText` (the deltas delivered to `onToken`), `partialThinking` (where the strategy streams reasoning), `usage` plus `usageEstimated` (provider-reported counts when both sides were already reported mid-stream, otherwise a deliberately rough `chars / 4` estimate for the missing side), `providerName`, and the original error as `cause`. Callers can persist the partial answer and account for the tokens spent.

Additive and backwards-compatible: the error's `name` is `"AbortError"`, so existing `err.name === "AbortError"` catch blocks keep working, and `withRetry` / `withCircuitBreaker` keep treating the abort as intentional (never retried, never a provider failure). Non-streaming calls, calls aborted before they start, and non-abort errors are unchanged.
