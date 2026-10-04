---
"@idevconn/llm-router": minor
---

Add `LlmAbortedError`: when a streaming `generate()` call (`onToken` set) is aborted via `signal`, all five built-in strategies (Claude, Gemini, ChatGPT, Grok, DeepSeek) now reject with an `LlmAbortedError` that carries what was received so far — `partialText` (the deltas delivered to `onToken`), `partialThinking` (where the strategy streams reasoning), `usage` plus `usageEstimated` (provider-reported counts when both sides were already reported mid-stream, otherwise a deliberately rough `chars / 4` estimate for the missing side), `providerName`, and the original error (or the signal's abort reason) as `cause`. Callers can persist the partial answer and account for the tokens spent.

Additive and backwards-compatible: the error's `name` is `"AbortError"`, so existing `err.name === "AbortError"` catch blocks keep working, and `withRetry` / `withCircuitBreaker` keep treating the abort as intentional (never retried, never a provider failure). Non-streaming calls, calls aborted before the request starts (these still throw the plain `signal.reason`; Claude's streaming path gains the same pre-start check the other strategies already had), and non-abort errors are unchanged. For ChatGPT/Grok/DeepSeek, whose `openai` SDK ends an aborted stream silently, a stream that stops without a `finish_reason` after the signal aborted now rejects with `LlmAbortedError` instead of resolving with truncated text and zero usage; an abort after `finish_reason` still resolves normally.

Also fixes an unhandled promise rejection in the Gemini direct-API streaming path: `result.response` (derived from the same SDK stream) rejected unobserved whenever stream iteration failed, which could crash the process.
