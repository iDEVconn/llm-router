---
"@idevconn/llm-router": minor
---

Add prompt injection defense in `src/injection-defense.ts`: `sanitizeUntrustedContent(text, opts?)` wraps untrusted text (retrieved documents, tool output, ...) in explicit delimiters and a data-only instruction before it's embedded in a prompt, and `detectPromptInjection(text)` is a cheap, synchronous regex/keyword gate that flags common injection phrasing ("ignore previous instructions", role reassignment, fake `system:` prefixes, premature closing of a delimited block) before any generation call is made. For a probabilistic second opinion, `detectPromptInjectionWithModel(text, strategy)` runs the same check through an `LlmStrategy` — deliberately separate and opt-in, since it costs a model call and can't be part of the cheap pre-generation gate.
