# @idevconn/llm-router

Library-agnostic LLM router. Provider-neutral `LlmStrategy` interface + `LlmRegistry` with env-driven platform selection, BYOK support, and boot-time env-key audit. Opt-in adapters for Gemini, Claude, Grok, ChatGPT, and DeepSeek via subpath exports — install only the SDKs you actually use. A `TaskRouter` + `Orchestrator` on top can split a free-text task into subtasks and run each on whichever registered provider fits best.

## Features

- Pure router core: zero SDK dependencies on the main entry. Just types + `LlmRegistry`.
- Subpath adapters: `@idevconn/llm-router/gemini`, `/claude`, `/grok`, `/chatgpt`, `/deepseek`. Each declares its SDK as an **optional** peer dependency, so consumers install only what they need.
- BYOK first-class: every strategy accepts a per-call `apiKey` that overrides the platform key for that one request.
- Platform-fallback fully optional: pass `platform: null` to `LlmRegistry` to require BYOK from every caller — useful for SaaS that doesn't subsidize AI usage.
- Typed errors: `UnknownProviderError`, `NoPlatformProviderError`, `InvalidPlatformProviderError`, `LlmKeyValidationError`, `UnsupportedAttachmentError`, `TaskDecompositionError`, `NoAvailableProviderError`, `BudgetExceededError`. No framework-specific exceptions.
- Cost control: `withBudget` decorator enforces per-call and total spend caps against a caller-supplied pricing table.
- Instrumentation: `withInstrumentation` decorator emits a call event (usage, latency, truncation, errors) to any logger you choose.
- Prompt injection defense: `sanitizeUntrustedContent` + `detectPromptInjection` (cheap heuristic gate) + `detectPromptInjectionWithModel` (opt-in LLM-based second opinion).

## Install

```bash
npm install @idevconn/llm-router

# Then install only the SDKs for the providers you use:
npm install @google/generative-ai   # for Gemini (direct API / BYOK)
npm install @google/genai           # optional; only for GeminiStrategy({ connection: "vertex" })
npm install @anthropic-ai/sdk       # for Claude
npm install openai                  # for Grok, ChatGPT, and DeepSeek (all OpenAI-compatible)
```

## Quick start

```ts
import { LlmRegistry } from "@idevconn/llm-router";
import { GeminiStrategy } from "@idevconn/llm-router/gemini";
import { ClaudeStrategy } from "@idevconn/llm-router/claude";
import { GrokStrategy } from "@idevconn/llm-router/grok";

const registry = new LlmRegistry<"gemini" | "claude" | "grok">({
  strategies: [
    new GeminiStrategy({
      apiKey: process.env.GEMINI_API_KEY,
      defaultModel: process.env.GEMINI_MODEL,
      // connection: "vertex" → Vertex AI via ADC; BYOK still uses the API key
    }),
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

`hasPlatformKey()` and `capabilities` are optional, so a strategy like the one
above works fine for direct `registry.get("ollama").generate(...)` calls. It
just won't participate in `TaskRouter`'s automatic matching: without
`hasPlatformKey()` the router treats it as unavailable, and without
`capabilities` it never wins the capability-tag rule stage. To opt a custom
strategy in without implementing either, pass a `providerOverrides` entry
(a `ProviderDescriptor`: `{ provider, available?, capabilities? }`) on the
`TaskRouter.route()` / `Orchestrator.run()` call — or supply that provider's
key in `apiKeys`, which marks it available for that call. Either way it then
routes like a built-in. See [Task orchestration](#task-orchestration) below.

## Task orchestration

`Orchestrator` splits a free-text task into subtasks and runs each on
whichever registered provider is best suited, using only providers that
have a usable key (platform or BYOK for this call):

```ts
import { LlmRegistry, Orchestrator } from "@idevconn/llm-router";
import { ClaudeStrategy } from "@idevconn/llm-router/claude";
import { GeminiStrategy } from "@idevconn/llm-router/gemini";

const registry = new LlmRegistry({
  strategies: [
    new ClaudeStrategy({ apiKey: process.env.CLAUDE_API_KEY }),
    new GeminiStrategy({ apiKey: process.env.GEMINI_API_KEY }),
  ],
  platform: "claude",
});

const orchestrator = new Orchestrator({ registry });
const result = await orchestrator.run(
  "Summarize this contract and flag any unusual liability clauses.",
  { synthesize: true },
);

for (const subtask of result.subtasks) {
  console.log(subtask.subtask.description, "->", subtask.decision?.provider, subtask.result);
}
console.log(result.final);
```

Routing is capability-tag matching first (`TaskRouter`'s rule stage),
falling back to a one-shot LLM classifier call when tags don't decide it.
Each subtask runs through a bounded critique/retry loop (`maxRounds`,
default 1) before being flagged `unresolved: true` in its result. One
subtask's failure never aborts the run — see the
[task-orchestrator design doc](https://github.com/iDEVconn/llm-router/blob/main/docs/superpowers/specs/2026-08-25-task-orchestrator-design.md)
for the full design.

## Cost control

`calculateCost` turns `LlmResponse.usage` into a dollar figure using a pricing
table you own and pass in (prices drift independently of this package's
release cycle, so nothing is hardcoded). `withBudget` wraps any `LlmStrategy`
to enforce spend limits across calls:

```ts
import { withBudget, type PricingTable } from "@idevconn/llm-router";

const pricing: PricingTable = {
  gemini: { "gemini-2.5-flash": { inputPer1M: 0.3, outputPer1M: 2.5 } },
};

const budgeted = withBudget(strategy, {
  pricing,
  maxCostPerCall: 0.5,
  maxCostTotal: 20,
  onCost: ({ provider, model, cost }) => console.log(provider, model, cost),
});
```

Cost is only known after a call returns (token counts come from the
response), so `maxCostPerCall` is checked against the call that just
finished, while `maxCostTotal` is checked up front against the accumulated
total before the next call is allowed to start. Either limit being exceeded
throws `BudgetExceededError`.

## Instrumentation

`withInstrumentation` wraps any `LlmStrategy` and emits an `LlmCallEvent` on
both success and failure — no bundled logger, you decide where events go:

```ts
import { compose, withBudget, withInstrumentation } from "@idevconn/llm-router";

const instrumented = compose(
  strategy,
  (s) => withBudget(s, { pricing }),
  (s) => withInstrumentation(s, { onCall: (event) => logger.info(event) }),
);
```

`compose(s, a, b)` behaves like `b(a(s))`, so decorators chain without manual
nesting.

## Prompt injection defense

Retrieved documents, tool output, and other untrusted text embedded in a
prompt are a classic injection vector. `sanitizeUntrustedContent` wraps such
text in explicit delimiters plus a data-only instruction; `detectPromptInjection`
is a cheap synchronous heuristic gate meant to run before generation:

```ts
import { detectPromptInjection, sanitizeUntrustedContent } from "@idevconn/llm-router";

const { suspicious, reasons } = detectPromptInjection(userSuppliedText);
if (suspicious) {
  // log, reject, or route to stricter handling — reasons explains why
}

const prompt = `Answer using this context:\n${sanitizeUntrustedContent(retrievedDoc)}`;
```

For a probabilistic second opinion, `detectPromptInjectionWithModel(text, strategy)`
runs the same check through an `LlmStrategy` — deliberately separate and
opt-in, since it costs a model call.

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
  if (err instanceof BudgetExceededError) throw new HttpException(err.message, 402);
  throw err;
}
```

## Stability

Pre-1.0 — minor versions may break API. Pin a tilde range until the first real second consumer surfaces real-world feedback on the shape.

## License

Apache-2.0
