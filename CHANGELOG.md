# @idevconn/llm-router

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
