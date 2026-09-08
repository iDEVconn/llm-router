---
"@idevconn/llm-router": minor
---

Add streaming and reasoning/thinking support across all five strategies. `LlmGenerateOptions` gains `onToken?: (delta: string) => void` (incremental text deltas; silently unused by strategies without streaming), `thinking?: {type:'adaptive'} | {type:'budget', tokens} | {type:'effort', level}` (provider-agnostic reasoning-depth request), and `signal?: AbortSignal` (cancels the in-flight call, including open streams). `LlmResponse` gains `thinking?: string` for a returned reasoning trace. New typed errors `UnsupportedThinkingModeError` and `InvalidThinkingConfigError` in `src/errors.ts`. New `'streaming'`/`'thinking'` tags in `KNOWN_CAPABILITY_TAGS`.

All changes are additive and backward compatible — the non-streaming, non-thinking code path is unchanged in every strategy.

Per-provider support, verified against each installed SDK's own type definitions (not assumed):
- **Claude**: streaming via `messages.stream()`; thinking `adaptive` and `budget` (`budget_tokens` validated ≥1024 and < `maxTokens`); `effort` unsupported.
- **Gemini**: streaming via `generateContentStream` on both connection modes. Thinking only on the `vertex` connection (`@google/genai`'s `thinkingConfig`) — the direct API's `@google/generative-ai` SDK has no thinking support at all in the installed version, so any `thinking` request on that connection throws `UnsupportedThinkingModeError`. `capabilities` is now connection-dependent.
- **ChatGPT (OpenAI)**: streaming via `stream:true` + `stream_options.include_usage`; thinking `effort` only (maps to Chat Completions' `reasoning_effort`, validated against the `low`/`medium`/`high` allow-list). Chat Completions never returns a reasoning trace, so `LlmResponse.thinking` is always `undefined` for this strategy.
- **Grok (xAI)**: streaming fully implemented (same OpenAI-compatible pattern). **Thinking is not implemented** — xAI's reasoning-effort parameter contract could not be independently verified against primary documentation this round (the docs site is unreachable to raw fetch, and no reliable confirmation of the exact parameter name/shape was obtained), so any `thinking` request throws `UnsupportedThinkingModeError`. Revisit once verified against docs.x.ai directly.
- **DeepSeek**: streaming fully implemented. **Thinking is not implemented as a request option** for the same reason as Grok — no independently verified request-side toggle. `deepseek-reasoner`'s `reasoning_content` is instead surfaced passively via `LlmResponse.thinking` whenever the model returns it (blocking or streamed), regardless of whether `thinking` was requested, since that's inherent to the model rather than a caller-controlled option.

`onToken` callback errors are caught and logged (`console.warn`), never propagate out of `generate()`. `signal` is honored natively by each SDK's own request options, not via `Promise.race`. `withBudget`/`withInstrumentation` already forward all three fields unmodified via their existing `...strategy` spread and untouched `genOpts` passthrough — confirmed with new passthrough tests, no changes needed to either decorator.

**Security**: reasoning/thinking budget and effort level are runtime-validated against allow-lists before any network call (never forwarded as unvalidated strings); `response.thinking` is untrusted model output exactly like `response.text` and must be sanitized before being re-fed into another prompt.
