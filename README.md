# @idevconn/llm-router

Library-agnostic LLM router. Provider-neutral `LlmStrategy` interface + `LlmRegistry` with env-driven platform selection, BYOK support, and boot-time env-key audit. Opt-in adapters for Gemini, Claude, and Grok via subpath exports — install only the SDKs you actually use.

## Features

- Pure router core: zero SDK dependencies on the main entry. Just types + `LlmRegistry`.
- Subpath adapters: `@idevconn/llm-router/gemini`, `/claude`, `/grok`. Each declares its SDK as an **optional** peer dependency, so consumers install only what they need.
- BYOK first-class: every strategy accepts a per-call `apiKey` that overrides the platform key for that one request.
- Platform-fallback fully optional: pass `platform: null` to `LlmRegistry` to require BYOK from every caller — useful for SaaS that doesn't subsidize AI usage.
- Typed errors: `UnknownProviderError`, `NoPlatformProviderError`, `InvalidPlatformProviderError`, `LlmKeyValidationError`, `UnsupportedAttachmentError`. No framework-specific exceptions.

## Install

```bash
npm install @idevconn/llm-router

# Then install only the SDKs for the providers you use:
npm install @google/generative-ai   # for Gemini
npm install @anthropic-ai/sdk       # for Claude
npm install openai                  # for Grok (xAI is OpenAI-compatible)
```

## Quick start

```ts
import { LlmRegistry } from "@idevconn/llm-router";
import { GeminiStrategy } from "@idevconn/llm-router/gemini";
import { ClaudeStrategy } from "@idevconn/llm-router/claude";
import { GrokStrategy } from "@idevconn/llm-router/grok";

const registry = new LlmRegistry<"gemini" | "claude" | "grok">({
  strategies: [
    new GeminiStrategy({ apiKey: process.env.GEMINI_API_KEY, defaultModel: process.env.GEMINI_MODEL }),
    new ClaudeStrategy({ apiKey: process.env.CLAUDE_API_KEY, defaultModel: process.env.CLAUDE_MODEL }),
    new GrokStrategy({ apiKey: process.env.XAI_API_KEY, defaultModel: process.env.GROK_MODEL }),
  ],
  platform: process.env.ML_STRATEGY as "gemini" | "claude" | "grok" | null,
  providerEnvKeys: {
    gemini: "GEMINI_API_KEY",
    claude: "CLAUDE_API_KEY",
    grok: "XAI_API_KEY",
  },
  env: process.env,
});

// Platform call
const platform = registry.getPlatform();
const result = await platform.generate({
  prompt: "Summarize this invoice.",
  attachments: [{ data: fileBuffer, mimetype: "application/pdf" }],
});

// BYOK call — same registry, user-supplied provider + key
const strategy = registry.get("claude");
const byok = await strategy.generate({
  prompt: "Summarize this invoice.",
  attachments: [{ data: fileBuffer, mimetype: "image/png" }],
  apiKey: user.claudeApiKey,
  model: user.preferredModel,
});

// Live key check (used in BYOK save flows)
await strategy.validateKey(user.claudeApiKey, user.preferredModel);
```

## Adding a custom provider

Implement `LlmStrategy` and pass it to `LlmRegistry`. The SDK choice is yours — the pkg never imports it. Useful for Bedrock, Vertex, local models via Ollama, internal LLM gateways, etc.

```ts
import type { LlmStrategy } from "@idevconn/llm-router";

class OllamaStrategy implements LlmStrategy {
  readonly providerName = "ollama";
  readonly defaultModel = "llama3.1";

  async generate(opts) { /* call your gateway */ }
  async validateKey(apiKey, model) { /* ping endpoint */ }
}
```

## Error mapping

The pkg throws plain `Error` subclasses so it stays framework-agnostic. Wrap at the controller boundary:

```ts
// NestJS example
try {
  return await registry.getPlatform().generate(opts);
} catch (err) {
  if (err instanceof UnknownProviderError) throw new NotFoundException(err.message);
  if (err instanceof NoPlatformProviderError) throw new BadRequestException(err.message);
  if (err instanceof LlmKeyValidationError) throw new BadRequestException(err.message);
  if (err instanceof UnsupportedAttachmentError) throw new BadRequestException(err.message);
  throw err;
}
```

## Stability

Pre-1.0 — minor versions may break API. Pin a tilde range until the first real second consumer surfaces real-world feedback on the shape.

## License

Apache-2.0
