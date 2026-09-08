# @idevconn/llm-router

## 0.10.0

### Minor Changes

- 1ae1da5: Add streaming and reasoning/thinking support across all five strategies. `LlmGenerateOptions` gains `onToken?: (delta: string) => void` (incremental text deltas; silently unused by strategies without streaming), `thinking?: {type:'adaptive'} | {type:'budget', tokens} | {type:'effort', level}` (provider-agnostic reasoning-depth request), and `signal?: AbortSignal` (cancels the in-flight call, including open streams). `LlmResponse` gains `thinking?: string` for a returned reasoning trace. New typed errors `UnsupportedThinkingModeError` and `InvalidThinkingConfigError` in `src/errors.ts`. New `'streaming'`/`'thinking'` tags in `KNOWN_CAPABILITY_TAGS`.
  
  All changes are additive and backward compatible — the non-streaming, non-thinking code path is unchanged in every strategy.
  
  Per-provider support, verified against each installed SDK's own type definitions (not assumed):
  - **Claude**: streaming via `messages.stream()`; thinking `adaptive` and `budget` (`budget_tokens` validated ≥1024 and < `maxTokens`); `effort` unsupported.
  - **Gemini**: streaming via `generateContentStream` on both connection modes. Thinking only on the `vertex` connection (`@google/genai`'s `thinkingConfig`) — the direct API's `@google/generative-ai` SDK has no thinking support at all in the installed version, so any `thinking` request on that connection throws `UnsupportedThinkingModeError`. `capabilities` is now connection-dependent.
  - **ChatGPT (OpenAI)**: streaming via `stream:true` + `stream_options.include_usage`; thinking `effort` only (maps to Chat Completions' `reasoning_effort`, validated against the `low`/`medium`/`high` allow-list). Chat Completions never returns a reasoning trace, so `LlmResponse.thinking` is always `undefined` for this strategy.
  - **Grok (xAI)**: streaming fully implemented (same OpenAI-compatible pattern). **Thinking is not implemented** — xAI's reasoning-effort parameter contract could not be independently verified against primary documentation this round (the docs site is unreachable to raw fetch, and no reliable confirmation of the exact parameter name/shape was obtained), so any `thinking` request throws `UnsupportedThinkingModeError`. Revisit once verified against docs.x.ai directly.
  - **DeepSeek**: streaming fully implemented. **Thinking is not implemented as a request option** for the same reason as Grok — no independently verified request-side toggle. `deepseek-reasoner`'s `reasoning_content` is instead surfaced passively via `LlmResponse.thinking` whenever the model returns it (blocking or streamed), regardless of whether `thinking` was requested, since that's inherent to the model rather than a caller-controlled option.
  
  `onToken` callback errors are caught and logged (`console.warn`), never propagate out of `generate()`. `signal` is honored natively by each SDK's own request options, not via `Promise.race`. `withBudget`/`withInstrumentation` already forward all three fields unmodified via their existing `...strategy` spread and untouched `genOpts` passthrough — confirmed with new passthrough tests, no changes needed to either decorator.
  
  **Security**: reasoning/thinking budget and effort level are runtime-validated against allow-lists before any network call (never forwarded as unvalidated strings); `response.thinking` is untrusted model output exactly like `response.text` and must be sanitized before being re-fed into another prompt.

## 0.9.0

### Minor Changes

- 38c46a9: Add cost control: `calculateCost(usage, provider, model, pricing)` in `src/pricing.ts` computes cost from a caller-supplied `PricingTable` (prices are not baked in, since they drift independently of this package), and `withBudget(strategy, opts)` in `src/budget.ts` wraps any `LlmStrategy` to track spend across calls, throwing the new `BudgetExceededError` when `maxCostPerCall` is exceeded by a call's actual cost, or when `maxCostTotal` has already been reached before the next call starts. `opts.onCost` fires with `{ provider, model, cost, usage }` after each successful call.
- 38c46a9: Add `withInstrumentation(strategy, { onCall })` in `src/instrumentation.ts`, a decorator that wraps any `LlmStrategy` and emits an `LlmCallEvent` (`{ provider, model, usage, truncated, latencyMs, timestamp, error? }`) on both success and failure, without imposing a specific logger — the caller's `onCall` decides where events go. Also adds `compose(strategy, ...decorators)` so `withBudget` and `withInstrumentation` (or any other `LlmStrategy` decorator) can be chained without manual nesting: `compose(s, a, b)` behaves like `b(a(s))`.
- 38c46a9: Add prompt injection defense in `src/injection-defense.ts`: `sanitizeUntrustedContent(text, opts?)` wraps untrusted text (retrieved documents, tool output, ...) in explicit delimiters and a data-only instruction before it's embedded in a prompt, and `detectPromptInjection(text)` is a cheap, synchronous regex/keyword gate that flags common injection phrasing ("ignore previous instructions", role reassignment, fake `system:` prefixes, premature closing of a delimited block) before any generation call is made. For a probabilistic second opinion, `detectPromptInjectionWithModel(text, strategy)` runs the same check through an `LlmStrategy` — deliberately separate and opt-in, since it costs a model call and can't be part of the cheap pre-generation gate.
- 38c46a9: Add RAG support. `EmbeddingStrategy` (`src/embeddings/types.ts`) mirrors `LlmStrategy` for embeddings providers, and `VectorStore` (`src/embeddings/vector-store.ts`) is a provider-neutral upsert/query/delete interface. `Retriever` (`src/rag.ts`) ties them together: `retrieve(query, opts?)` embeds the query, queries the store, and returns `{ chunks, sources }` — every chunk is run through `sanitizeUntrustedContent` before it's returned, since retrieved content is a classic prompt-injection vector and this is not optional.
  
  Ships one concrete adapter, `PgVectorStore`, via the new `@idevconn/llm-router/embeddings/pgvector` subpath export (same optional-peer-dependency pattern as the LLM adapters — declare `pg` yourself). It covers Postgres directly and Supabase in one shot, since Supabase is Postgres with pgvector built in; Mongo Atlas Vector Search and Vertex AI-backed stores need their own adapters later. It expects a table with `id text primary key`, `embedding vector(n)`, and `metadata jsonb` columns and never runs DDL. Filter keys are validated against a simple-identifier pattern before being interpolated into SQL (values are always parametrized) to prevent injection through metadata filter keys.

## 0.8.0

### Minor Changes

- cbe2187: `GeminiStrategy` accepts `connection: "vertex"` to run platform-funded calls through Vertex AI with Application Default Credentials, using `@google/genai`. Per-call BYOK still routes through the direct Gemini API. No automatic failover.

## Unreleased

### Minor Changes

- `GeminiStrategy` accepts `connection: "vertex"` for platform-funded calls through Vertex AI (ADC, `@google/genai`). `providerName` stays `"gemini"`. Per-call BYOK still uses the direct Gemini API. No automatic failover.

## 0.7.0

### Minor Changes

- a240b54: Add `TaskRouter` and `Orchestrator` for capability-based multi-provider task routing, plus `ChatGptStrategy` and `DeepSeekStrategy` adapters. `LlmStrategy` gains two optional members, `hasPlatformKey?()` and `capabilities?`, which existing custom strategies can ignore without breaking.

## 0.6.0

### Minor Changes

- 622498f: Add `truncated` to `LlmResponse`, reporting whether the provider stopped generating because it hit the output-token limit rather than finishing naturally — `text` is very likely incomplete when this is `true` (e.g. truncated JSON that will fail to parse).
  
  Each strategy reads its own provider's exact signal rather than guessing from token counts:
  
  - **Claude**: `response.stop_reason === "max_tokens"`
  - **Gemini**: `response.candidates[0].finishReason === "MAX_TOKENS"`
  - **Grok**: `response.choices[0].finish_reason === "length"`
  
  Callers that previously had to inspect a provider-specific field (e.g. Anthropic's `stop_reason`) to detect truncation can now check `result.truncated` uniformly across all three providers.

## 0.5.0

### Minor Changes

- dd0d374: Add `systemPrompt` to `LlmGenerateOptions` for stable, cacheable instructions separate from the per-call `prompt`.
  
  - **Claude**: `systemPrompt` is sent as a `cache_control: ephemeral` system block, so repeated calls that reuse the same instructions (the common case for report-generation prompts) are billed at Anthropic's cheaper cache-read rate instead of full input-token price.
  - **Gemini**: `systemPrompt` is passed as `systemInstruction` — correctly separates instructions from the prompt, but does not reduce cost (Gemini's actual Cached Content API is a separate create/lifecycle/TTL flow with a much higher minimum token count, out of scope here).
  - **Grok**: `systemPrompt` is sent as a leading `system`-role message — correct OpenAI-wire-format shape, and positions the request to benefit from any automatic prefix-based caching xAI's backend may apply (undocumented, not guaranteed).
  
  `systemPrompt` is optional and fully backward compatible — omitting it preserves the exact existing single-message behavior for all three strategies.

## 0.2.0

### Minor Changes

- d736bdc: Initial release.

  Library-agnostic LLM router. Main entry ships pure types + `LlmRegistry`
  with env-driven platform selection, BYOK support, and boot-time env-key
  audit — zero SDK dependencies. Concrete adapters for Gemini, Claude,
  and Grok live behind subpath exports (`@idevconn/llm-router/gemini`,
  `/claude`, `/grok`) with their SDKs declared as **optional** peer
  dependencies, so consumers install only what they actually use.

  Errors are plain `Error` subclasses (`UnknownProviderError`,
  `NoPlatformProviderError`, `InvalidPlatformProviderError`,
  `LlmKeyValidationError`, `UnsupportedAttachmentError`) — callers map
  them to framework-specific exceptions at their controller boundary.
