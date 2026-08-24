---
"@idevconn/llm-router": minor
---

Add `truncated` to `LlmResponse`, reporting whether the provider stopped generating because it hit the output-token limit rather than finishing naturally — `text` is very likely incomplete when this is `true` (e.g. truncated JSON that will fail to parse).

Each strategy reads its own provider's exact signal rather than guessing from token counts:

- **Claude**: `response.stop_reason === "max_tokens"`
- **Gemini**: `response.candidates[0].finishReason === "MAX_TOKENS"`
- **Grok**: `response.choices[0].finish_reason === "length"`

Callers that previously had to inspect a provider-specific field (e.g. Anthropic's `stop_reason`) to detect truncation can now check `result.truncated` uniformly across all three providers.
