---
"@idevconn/llm-router": minor
---

Add `systemPrompt` to `LlmGenerateOptions` for stable, cacheable instructions separate from the per-call `prompt`.

- **Claude**: `systemPrompt` is sent as a `cache_control: ephemeral` system block, so repeated calls that reuse the same instructions (the common case for report-generation prompts) are billed at Anthropic's cheaper cache-read rate instead of full input-token price.
- **Gemini**: `systemPrompt` is passed as `systemInstruction` — correctly separates instructions from the prompt, but does not reduce cost (Gemini's actual Cached Content API is a separate create/lifecycle/TTL flow with a much higher minimum token count, out of scope here).
- **Grok**: `systemPrompt` is sent as a leading `system`-role message — correct OpenAI-wire-format shape, and positions the request to benefit from any automatic prefix-based caching xAI's backend may apply (undocumented, not guaranteed).

`systemPrompt` is optional and fully backward compatible — omitting it preserves the exact existing single-message behavior for all three strategies.
