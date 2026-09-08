# @idevconn/llm-router

Library-agnostic LLM router. Provider-neutral `LlmStrategy` interface + `LlmRegistry` with env-driven platform selection, BYOK support, and boot-time env-key audit. Opt-in adapters for Gemini, Claude, Grok, ChatGPT, and DeepSeek via subpath exports — install only the SDKs you actually use. A `TaskRouter` + `Orchestrator` on top can split a free-text task into subtasks and run each on whichever registered provider fits best.

## Features

- Pure router core: zero SDK dependencies on the main entry. Just types + `LlmRegistry`.
- Subpath adapters: `@idevconn/llm-router/gemini`, `/claude`, `/grok`, `/chatgpt`, `/deepseek`. Each declares its SDK as an **optional** peer dependency, so consumers install only what they need.
- BYOK first-class: every strategy accepts a per-call `apiKey` that overrides the platform key for that one request.
- Platform-fallback fully optional: pass `platform: null` to `LlmRegistry` to require BYOK from every caller — useful for SaaS that doesn't subsidize AI usage.
- Streaming (`onToken`) and reasoning/thinking (`thinking`) request options, and call cancellation (`signal`) — provider-agnostic on `LlmGenerateOptions`, implemented per-provider where the underlying SDK actually supports it (see [Streaming and reasoning/thinking](#streaming-and-reasoningthinking)).
- Typed errors: `UnknownProviderError`, `NoPlatformProviderError`, `InvalidPlatformProviderError`, `LlmKeyValidationError`, `UnsupportedAttachmentError`, `TaskDecompositionError`, `NoAvailableProviderError`, `BudgetExceededError`, `UnsupportedThinkingModeError`, `InvalidThinkingConfigError`. No framework-specific exceptions.
- Cost control: `withBudget` decorator enforces per-call and total spend caps against a caller-supplied pricing table.
- Instrumentation: `withInstrumentation` decorator emits a call event (usage, latency, truncation, errors) to any logger you choose.
- Prompt injection defense: `sanitizeUntrustedContent` + `detectPromptInjection` (cheap heuristic gate) + `detectPromptInjectionWithModel` (opt-in LLM-based second opinion).
- RAG: `Retriever` + `EmbeddingStrategy`/`VectorStore` interfaces, with a `pgvector`-backed `VectorStore` adapter via a subpath export.

## Install

```bash
npm install @idevconn/llm-router

# Then install only the SDKs for the providers you use:
npm install @google/generative-ai   # for Gemini (direct API / BYOK)
npm install @google/genai           # optional; only for GeminiStrategy({ connection: "vertex" })
npm install @anthropic-ai/sdk       # for Claude
npm install openai                  # for Grok, ChatGPT, and DeepSeek (all OpenAI-compatible)
npm install pg                      # for the pgvector VectorStore adapter
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

## Streaming and reasoning/thinking

`LlmGenerateOptions` has three additive fields. A strategy that doesn't
implement one of them simply ignores it — every existing caller keeps
working unchanged.

- **`onToken?: (delta: string) => void`** — called with each incremental
  text delta as it arrives. When set, a strategy that supports streaming
  uses the provider's streaming endpoint internally, but still resolves
  the same `Promise<LlmResponse>` once the stream ends. A strategy MAY
  never invoke it (no streaming support) — don't assume it fires. A
  throwing callback is caught and logged, never propagates out of
  `generate()`. Keep it fast/sync-safe: it is not awaited between deltas,
  so slow async work inside it will not backpressure the provider stream.
- **`thinking?: {type:'adaptive'} | {type:'budget', tokens} | {type:'effort', level:'low'|'medium'|'high'}`**
  — provider-agnostic reasoning-depth request. Check the `'thinking'`
  capability tag before requesting it, or catch
  `UnsupportedThinkingModeError`. `budget.tokens`/`effort.level` are
  runtime-validated against allow-lists before any network call — an
  out-of-range or unsupported value throws `InvalidThinkingConfigError`
  or `UnsupportedThinkingModeError` up front, never silently downgraded.
- **`signal?: AbortSignal`** — cancels the call, including an open stream.

`LlmResponse.thinking?: string` carries a returned reasoning trace, when
the provider returned one. **Treat it exactly like `response.text` —
untrusted model output.** Never re-feed it into another prompt (e.g. a
"show your reasoning" UI that summarizes it via another LLM call) without
running it through `sanitizeUntrustedContent`/`detectPromptInjection` first.

Per-provider support (check `strategy.capabilities` for `'streaming'`/`'thinking'`):

| Provider | Streaming | Thinking |
|---|---|---|
| Claude | ✅ `messages.stream()` | `adaptive`, `budget` (`budget_tokens` ≥1024 and < `maxTokens`). `effort` unsupported. |
| Gemini | ✅ both connection modes | **Vertex only** (`GeminiStrategy({ connection: "vertex" })`) — `adaptive`, `budget`. The direct API SDK has no thinking support at all; any `thinking` request on that connection throws `UnsupportedThinkingModeError`. |
| ChatGPT | ✅ `stream:true` | `effort` only (`reasoning_effort`). Chat Completions never returns a reasoning trace — `response.thinking` is always `undefined` for this strategy. |
| Grok | ✅ `stream:true` | **Not implemented.** xAI's reasoning-effort parameter contract could not be verified against primary documentation — any `thinking` request throws `UnsupportedThinkingModeError` until this is confirmed against docs.x.ai. |
| DeepSeek | ✅ `stream:true` | **Not implemented as a request option**, same reason as Grok. `deepseek-reasoner`'s `reasoning_content` is instead surfaced passively via `response.thinking` whenever the model returns it — inherent to the model, not caller-controlled. |

```ts
const stream = await strategy.generate({
  prompt: "Explain quantum entanglement.",
  onToken: (delta) => process.stdout.write(delta),
  thinking: { type: "budget", tokens: 2048 },
  signal: abortController.signal,
});
console.log(stream.thinking); // reasoning trace, if the provider returned one
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

## RAG

`EmbeddingStrategy` mirrors `LlmStrategy` for embeddings providers, and
`VectorStore` is a provider-neutral upsert/query/delete interface. `Retriever`
ties them together and always runs retrieved chunks through
`sanitizeUntrustedContent` before returning them:

```ts
import { Retriever } from "@idevconn/llm-router";
import { PgVectorStore } from "@idevconn/llm-router/embeddings/pgvector";
import { Pool } from "pg";

const vectorStore = new PgVectorStore({ pool: new Pool() });
const retriever = new Retriever({ embeddingStrategy, vectorStore });

const { chunks, sources } = await retriever.retrieve("What's our refund policy?", { topK: 5 });
const prompt = `Context:\n${chunks.join("\n\n")}\n\nQuestion: ...`;
```

`PgVectorStore` (subpath export, optional `pg` peer dependency) expects a
table with `id text primary key`, `embedding vector(n)`, and `metadata jsonb`
columns — create the table and its pgvector index yourself; the adapter never
runs DDL.

`PgVectorStore` is just one `VectorStore` implementation, not a hard
dependency on Postgres — Supabase works out of the box (it's Postgres with
pgvector under the hood, so it's just a `Pool` pointed at a Supabase
connection string). A different backend (Mongo Atlas Vector Search,
Firestore vector search, Pinecone, etc.) means writing a second
implementation of the same three-method `VectorStore` interface
(`upsert`/`query`/`delete`) and passing it to `Retriever` — the same
opt-in-adapter pattern as adding a new `LlmStrategy` provider.

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
  if (err instanceof UnsupportedThinkingModeError) throw new BadRequestException(err.message);
  if (err instanceof InvalidThinkingConfigError) throw new BadRequestException(err.message);
  if (err instanceof BudgetExceededError) throw new HttpException(err.message, 402);
  throw err;
}
```

## Stability

Pre-1.0 — minor versions may break API. Pin a tilde range until the first real second consumer surfaces real-world feedback on the shape.

## License

Apache-2.0
