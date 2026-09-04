# @idevconn/llm-router

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
