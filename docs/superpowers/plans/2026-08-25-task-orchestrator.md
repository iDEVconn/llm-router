# Task Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `TaskRouter` (capability-based provider routing) and an `Orchestrator` (decompose → route → execute-with-critique → optional synthesis) on top of `LlmRegistry`, plus two new provider strategies (`chatgpt`, `deepseek`) so the router has more candidates.

**Architecture:** `LlmStrategy` gains two optional members (`hasPlatformKey?()`, `capabilities?`). `TaskRouter` (src/task-router.ts) is pure decision logic — no execution loop. `Orchestrator` (src/orchestrator.ts) owns the pipeline and is the only piece that calls `generate()` for the actual work. Both live in the main entry (no SDK import). `ChatGptStrategy`/`DeepSeekStrategy` copy `GrokStrategy`'s shape exactly.

**Tech Stack:** TypeScript, vitest, tsup, the `openai` SDK (already an optional peer dep) for the two new strategies.

**Spec:** `docs/superpowers/specs/2026-08-25-task-orchestrator-design.md`

## Global Constraints

- `hasPlatformKey` and `capabilities` on `LlmStrategy` are **optional** — never make them required members. A strategy that omits either keeps compiling untouched.
- Safe default: a provider with unknown/absent `hasPlatformKey` is treated as **unavailable** unless `apiKeys[name]` or a `providerOverrides` entry says otherwise.
- Capability tags are exactly this closed set — no others, anywhere: `vision`, `code`, `long-context`, `cheap`, `reasoning`, `multilingual` (exported as `KNOWN_CAPABILITY_TAGS` from `src/task-router.ts`).
- `unresolved: true` iff the critique loop exhausted all rounds without ever receiving an explicit `approved: true` verdict. Never report an unverified retry as resolved.
- One subtask's routing or execution failure never aborts the whole `run()` — every other subtask still completes. `Orchestrator.run()` only rejects on `TaskDecompositionError` (nothing to route yet).
- Defaults: `maxRounds = 1`, `critique = "self"`, `synthesize = false`, `maxConcurrency = 3`, `maxSubtasks = 20`.
- `task-router.ts` and `orchestrator.ts` import no provider SDK (README's "zero SDK dependencies on the main entry").
- `chatgpt` and `deepseek` strategies reuse the existing optional peer dep `openai` — no new dependency.
- Run `npm run typecheck` and `npm run test` after every task; both must be clean before moving on.

---

## File Structure

Create:
- `src/task-router.ts` — `TaskRouter`, `Subtask`, `RoutingDecision`, `ProviderDescriptor`, `KNOWN_CAPABILITY_TAGS`.
- `src/orchestrator.ts` — `Orchestrator`, `RunOptions`, `SubtaskResult`, `OrchestratorResult`.
- `src/chatgpt/index.ts` — `ChatGptStrategy`.
- `src/deepseek/index.ts` — `DeepSeekStrategy`.
- `src/__tests__/errors.test.ts`, `src/__tests__/chatgpt.test.ts`, `src/__tests__/deepseek.test.ts`, `src/__tests__/task-router.test.ts`, `src/__tests__/orchestrator.test.ts`.

Modify:
- `src/types.ts` — add `hasPlatformKey?()` and `capabilities?` to `LlmStrategy`.
- `src/errors.ts` — add `TaskDecompositionError`, `NoAvailableProviderError`.
- `src/claude/index.ts`, `src/gemini/index.ts`, `src/grok/index.ts` — add a `capabilities` field.
- `src/index.ts` — export the new public surface.
- `package.json`, `tsup.config.ts` — new `./chatgpt` and `./deepseek` subpath exports.
- `README.md` — document the new adapters and the orchestrator.

---

### Task 1: `LlmStrategy` interface additions + new error classes

**Files:**
- Modify: `src/types.ts`
- Modify: `src/errors.ts`
- Test: `src/__tests__/errors.test.ts`

**Interfaces:**
- Produces: `LlmStrategy.hasPlatformKey?(): boolean`, `LlmStrategy.capabilities?: readonly string[]`, `TaskDecompositionError`, `NoAvailableProviderError` — every later task depends on these.

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/errors.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { NoAvailableProviderError, TaskDecompositionError } from "../errors";

describe("TaskDecompositionError", () => {
  it("wraps an Error cause with its message", () => {
    const err = new TaskDecompositionError(new Error("bad json"));
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("TaskDecompositionError");
    expect(err.message).toBe("Failed to decompose the task into subtasks: bad json");
    expect(err.cause).toBeInstanceOf(Error);
  });

  it("stringifies a non-Error cause", () => {
    const err = new TaskDecompositionError("not json");
    expect(err.message).toBe("Failed to decompose the task into subtasks: not json");
  });
});

describe("NoAvailableProviderError", () => {
  it("names the unroutable subtask", () => {
    const err = new NoAvailableProviderError("subtask-2");
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("NoAvailableProviderError");
    expect(err.subtaskId).toBe("subtask-2");
    expect(err.message).toMatch(/subtask-2/);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- errors.test.ts`
Expected: FAIL — `TaskDecompositionError`/`NoAvailableProviderError` are not exported from `../errors`.

- [ ] **Step 3: Implement**

In `src/types.ts`, inside the `LlmStrategy` interface, after the `validateKey` method, add:

```ts
  /**
   * True when this strategy has a usable platform-level key configured
   * (constructor-supplied, not per-call BYOK). Optional for backward
   * compatibility — a strategy that omits this is treated by TaskRouter
   * as "unavailable," never assumed usable.
   */
  hasPlatformKey?(): boolean;

  /**
   * Free-text capability tags used by TaskRouter's rule-matching stage
   * (see `KNOWN_CAPABILITY_TAGS` in `task-router.ts`). Optional; a
   * strategy without tags never wins the rule stage and routes through
   * the LLM-fallback stage instead.
   */
  readonly capabilities?: readonly string[];
```

In `src/errors.ts`, update the top doc comment's list to add two lines:

```
 *   - `TaskDecompositionError`      → 502 (upstream model didn't cooperate)
 *   - `NoAvailableProviderError`    → 400 (no BYOK/platform key for the routed provider)
```

Then append at the end of the file:

```ts
export class TaskDecompositionError extends Error {
  constructor(public override readonly cause: unknown) {
    const causeMessage = cause instanceof Error ? cause.message : String(cause);
    super(`Failed to decompose the task into subtasks: ${causeMessage}`);
    this.name = "TaskDecompositionError";
  }
}

export class NoAvailableProviderError extends Error {
  constructor(public readonly subtaskId: string) {
    super(`No available provider (platform key or BYOK) can handle subtask "${subtaskId}".`);
    this.name = "NoAvailableProviderError";
  }
}
```

- [ ] **Step 4: Run test to verify it passes, and typecheck**

Run: `npm test -- errors.test.ts && npm run typecheck`
Expected: PASS, no type errors (the two new `LlmStrategy` members are optional, so `ClaudeStrategy`/`GeminiStrategy`/`GrokStrategy` still satisfy the interface even though `capabilities` isn't on them yet).

- [ ] **Step 5: Commit**

```bash
git add src/types.ts src/errors.ts src/__tests__/errors.test.ts
git commit -m "feat: add hasPlatformKey/capabilities to LlmStrategy, add routing errors"
```

---

### Task 2: Capability tags on existing strategies (claude, gemini, grok)

**Files:**
- Modify: `src/claude/index.ts`
- Modify: `src/gemini/index.ts`
- Modify: `src/grok/index.ts`
- Test: `src/__tests__/claude.test.ts`, `src/__tests__/gemini.test.ts`, `src/__tests__/grok.test.ts`

**Interfaces:**
- Consumes: `LlmStrategy.capabilities` from Task 1.
- Produces: `ClaudeStrategy.capabilities = ["code", "reasoning", "long-context"]`, `GeminiStrategy.capabilities = ["vision", "long-context", "multilingual", "cheap"]`, `GrokStrategy.capabilities = ["vision", "cheap"]` — Task 5's router tests and Task 6+'s default routing behavior assume these exact values.

- [ ] **Step 1: Write the failing tests**

In `src/__tests__/claude.test.ts`, add inside the `describe("ClaudeStrategy", ...)` block:

```ts
  it("declares its capability tags", () => {
    const strategy = new ClaudeStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["code", "reasoning", "long-context"]);
  });
```

In `src/__tests__/gemini.test.ts`, add inside `describe("GeminiStrategy", ...)`:

```ts
  it("declares its capability tags", () => {
    const strategy = new GeminiStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["vision", "long-context", "multilingual", "cheap"]);
  });
```

In `src/__tests__/grok.test.ts`, add inside `describe("GrokStrategy", ...)`:

```ts
  it("declares its capability tags", () => {
    const strategy = new GrokStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["vision", "cheap"]);
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- claude.test.ts gemini.test.ts grok.test.ts`
Expected: FAIL on all three new assertions — `capabilities` is `undefined`.

- [ ] **Step 3: Implement**

In `src/claude/index.ts`, change:

```ts
export class ClaudeStrategy implements LlmStrategy {
  readonly providerName = "claude";
  readonly defaultModel: string;
```

to:

```ts
export class ClaudeStrategy implements LlmStrategy {
  readonly providerName = "claude";
  readonly capabilities = ["code", "reasoning", "long-context"] as const;
  readonly defaultModel: string;
```

In `src/gemini/index.ts`, change:

```ts
export class GeminiStrategy implements LlmStrategy {
  readonly providerName = "gemini";
  readonly defaultModel: string;
```

to:

```ts
export class GeminiStrategy implements LlmStrategy {
  readonly providerName = "gemini";
  readonly capabilities = ["vision", "long-context", "multilingual", "cheap"] as const;
  readonly defaultModel: string;
```

In `src/grok/index.ts`, change:

```ts
export class GrokStrategy implements LlmStrategy {
  readonly providerName = "grok";
  readonly defaultModel: string;
```

to:

```ts
export class GrokStrategy implements LlmStrategy {
  readonly providerName = "grok";
  readonly capabilities = ["vision", "cheap"] as const;
  readonly defaultModel: string;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- claude.test.ts gemini.test.ts grok.test.ts && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/claude/index.ts src/gemini/index.ts src/grok/index.ts \
  src/__tests__/claude.test.ts src/__tests__/gemini.test.ts src/__tests__/grok.test.ts
git commit -m "feat: declare capability tags on claude, gemini, and grok strategies"
```

---

### Task 3: `ChatGptStrategy`

**Files:**
- Create: `src/chatgpt/index.ts`
- Test: `src/__tests__/chatgpt.test.ts`
- Modify: `package.json`
- Modify: `tsup.config.ts`

**Interfaces:**
- Produces: `ChatGptStrategy` (providerName `"chatgpt"`, capabilities `["code", "reasoning", "vision", "multilingual"]`, default model `"gpt-4.1-mini"`) — a `LlmStrategy` the registry can register like any other.

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/chatgpt.test.ts`:

```ts
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockChatCompletionsCreate = vi.fn();
const mockModelsList = vi.fn();

vi.mock("openai", () => {
  class OpenAI {
    public readonly chat: { completions: { create: typeof mockChatCompletionsCreate } };
    public readonly models: { list: typeof mockModelsList };
    constructor(public readonly opts: { apiKey: string; baseURL?: string }) {
      this.chat = { completions: { create: mockChatCompletionsCreate } };
      this.models = { list: mockModelsList };
    }
  }
  return { default: OpenAI };
});

import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import { ChatGptStrategy } from "../chatgpt/index";

describe("ChatGptStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("declares its capability tags", () => {
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["code", "reasoning", "vision", "multilingual"]);
  });

  it("sends an image_url message + prompt for image attachments", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "extracted" } }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 10, completion_tokens: 5 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "platform-key" });

    const result = await strategy.generate({
      prompt: "describe",
      attachments: [{ data: Buffer.from("img"), mimetype: "image/jpeg" }],
    });

    expect(result.text).toBe("extracted");
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 5 });

    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    const content = call.messages[0].content;
    expect(content[0].type).toBe("image_url");
    expect(content[0].image_url.url).toMatch(/^data:image\/jpeg;base64,/);
    expect(content[1].type).toBe("text");
    expect(content[1].text).toBe("describe");
  });

  it("throws UnsupportedAttachmentError on PDF inputs", async () => {
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    await expect(
      strategy.generate({
        prompt: "p",
        attachments: [{ data: Buffer.from("pdf"), mimetype: "application/pdf" }],
      }),
    ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
  });

  it("reports truncated=true when finish_reason is length", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "cut off" }, finish_reason: "length" }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 4, completion_tokens: 4096 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    const result = await strategy.generate({ prompt: "p" });
    expect(result.truncated).toBe(true);
  });

  it("sends systemPrompt as a leading system message when provided", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "ok" } }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "k" });
    await strategy.generate({ prompt: "p", systemPrompt: "Be concise." });

    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.messages[0]).toEqual({ role: "system", content: "Be concise." });
    expect(call.messages[1].role).toBe("user");
  });

  it("uses the custom baseURL when supplied", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "" } }],
      model: "gpt-4.1-mini",
      usage: { prompt_tokens: 0, completion_tokens: 0 },
    });
    const strategy = new ChatGptStrategy({ apiKey: "k", baseURL: "https://internal.openai.example/v1" });
    await strategy.generate({ prompt: "p" });
    expect(mockChatCompletionsCreate).toHaveBeenCalledOnce();
  });

  it("validateKey calls models.list without spending tokens", async () => {
    mockModelsList.mockResolvedValueOnce({ data: [] });
    const strategy = new ChatGptStrategy({});
    await strategy.validateKey("u-key");
    expect(mockModelsList).toHaveBeenCalledOnce();
  });

  it("validateKey wraps SDK rejections as LlmKeyValidationError", async () => {
    mockModelsList.mockRejectedValueOnce(new Error("unauthorized"));
    const strategy = new ChatGptStrategy({});
    await expect(strategy.validateKey("bad")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new ChatGptStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });

  it("hasPlatformKey reflects whether a constructor apiKey was given", () => {
    expect(new ChatGptStrategy({ apiKey: "k" }).hasPlatformKey()).toBe(true);
    expect(new ChatGptStrategy({}).hasPlatformKey()).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- chatgpt.test.ts`
Expected: FAIL — `../chatgpt/index` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/chatgpt/index.ts`:

```ts
import OpenAI from "openai";
import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const DEFAULT_BASE_URL = "https://api.openai.com/v1";
const FALLBACK_DEFAULT_MODEL = "gpt-4.1-mini";

const SUPPORTED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/gif",
  "image/webp",
]);

export interface ChatGptStrategyOptions {
  apiKey?: string;
  defaultModel?: string;
  /** Override the OpenAI API base URL. Default `https://api.openai.com/v1`. */
  baseURL?: string;
}

function toBase64(data: string | Buffer): string {
  if (typeof data === "string") return data;
  return data.toString("base64");
}

/**
 * OpenAI ChatGPT adapter. Chat Completions' vision input only accepts
 * images, not PDFs — callers must convert PDFs client-side. Hitting the
 * adapter with a non-image MIME yields `UnsupportedAttachmentError` up
 * front rather than an opaque 4xx mid-stream.
 */
export class ChatGptStrategy implements LlmStrategy {
  readonly providerName = "chatgpt";
  readonly capabilities = ["code", "reasoning", "vision", "multilingual"] as const;
  readonly defaultModel: string;
  private platformClient: OpenAI | null = null;
  private readonly platformApiKey: string | undefined;
  private readonly baseURL: string;

  constructor(opts: ChatGptStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
    this.baseURL = opts.baseURL?.trim() || DEFAULT_BASE_URL;
  }

  private getPlatformClient(): OpenAI {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "ChatGPT platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new OpenAI({ apiKey: this.platformApiKey, baseURL: this.baseURL });
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    for (const attachment of opts.attachments ?? []) {
      if (!SUPPORTED_IMAGE_TYPES.has(attachment.mimetype)) {
        throw new UnsupportedAttachmentError(
          this.providerName,
          attachment.mimetype,
          "OpenAI Chat Completions vision only accepts image inputs. Convert the file to PNG or JPEG, or switch to a provider with PDF support.",
        );
      }
    }

    const client = opts.apiKey
      ? new OpenAI({ apiKey: opts.apiKey, baseURL: this.baseURL })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;

    const messageContent: Array<
      | { type: "text"; text: string }
      | { type: "image_url"; image_url: { url: string; detail: "high" | "low" | "auto" } }
    > = [];

    for (const attachment of opts.attachments ?? []) {
      const data = toBase64(attachment.data);
      messageContent.push({
        type: "image_url",
        image_url: { url: `data:${attachment.mimetype};base64,${data}`, detail: "high" },
      });
    }
    messageContent.push({ type: "text", text: opts.prompt });

    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: messageContent },
        ]
      : [{ role: "user" as const, content: messageContent }];

    const response = await client.chat.completions.create({
      model: modelName,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: messages as any,
      ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
    });

    const raw = response.choices[0]?.message?.content ?? "";
    const text = typeof raw === "string" ? raw : "";

    return {
      text,
      model: response.model,
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
      },
      truncated: response.choices[0]?.finish_reason === "length",
    };
  }

  /**
   * Cheapest auth-checked call against OpenAI. `models.list` is free and
   * account-wide — no token spend. `model` is accepted to satisfy
   * `LlmStrategy.validateKey` but ignored, same as the Grok adapter.
   */
  async validateKey(apiKey: string, _model?: string): Promise<void> {
    const client = new OpenAI({ apiKey, baseURL: this.baseURL });
    try {
      await client.models.list();
    } catch (cause) {
      throw new LlmKeyValidationError(this.providerName, cause);
    }
  }

  hasPlatformKey(): boolean {
    return this.platformApiKey !== undefined;
  }
}
```

In `tsup.config.ts`, add an entry:

```ts
  entry: {
    index: "src/index.ts",
    gemini: "src/gemini/index.ts",
    claude: "src/claude/index.ts",
    grok: "src/grok/index.ts",
    chatgpt: "src/chatgpt/index.ts",
    deepseek: "src/deepseek/index.ts",
  },
```

(This adds both `chatgpt` and `deepseek` now — Task 4 doesn't need to touch this file again.)

In `package.json`, add to `"exports"` (after `"./grok"`):

```json
    "./chatgpt": {
      "types": "./dist/chatgpt.d.ts",
      "import": "./dist/chatgpt.js",
      "require": "./dist/chatgpt.cjs"
    },
    "./deepseek": {
      "types": "./dist/deepseek.d.ts",
      "import": "./dist/deepseek.js",
      "require": "./dist/deepseek.cjs"
    }
```

(Same here — both subpaths added now so `package.json` only needs one edit pass.)

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test -- chatgpt.test.ts && npm run typecheck && npm run build`
Expected: PASS, and `dist/chatgpt.js`/`dist/deepseek.js` are NOT produced yet (Task 4 creates `src/deepseek/index.ts`) — `npm run build` will fail on the missing `deepseek` entry. Run `npm run build` only after Task 4; for this task, `npm test -- chatgpt.test.ts && npm run typecheck` is the real gate.

- [ ] **Step 5: Commit**

```bash
git add src/chatgpt/index.ts src/__tests__/chatgpt.test.ts tsup.config.ts package.json
git commit -m "feat: add ChatGptStrategy (OpenAI adapter)"
```

---

### Task 4: `DeepSeekStrategy`

**Files:**
- Create: `src/deepseek/index.ts`
- Test: `src/__tests__/deepseek.test.ts`

**Interfaces:**
- Produces: `DeepSeekStrategy` (providerName `"deepseek"`, capabilities `["code", "reasoning", "cheap"]`, default model `"deepseek-chat"`) — no vision support at all, unlike the other four strategies.

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/deepseek.test.ts`:

```ts
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockChatCompletionsCreate = vi.fn();
const mockModelsList = vi.fn();

vi.mock("openai", () => {
  class OpenAI {
    public readonly chat: { completions: { create: typeof mockChatCompletionsCreate } };
    public readonly models: { list: typeof mockModelsList };
    constructor(public readonly opts: { apiKey: string; baseURL?: string }) {
      this.chat = { completions: { create: mockChatCompletionsCreate } };
      this.models = { list: mockModelsList };
    }
  }
  return { default: OpenAI };
});

import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import { DeepSeekStrategy } from "../deepseek/index";

describe("DeepSeekStrategy", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("declares its capability tags", () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    expect(strategy.capabilities).toEqual(["code", "reasoning", "cheap"]);
  });

  it("sends a plain-text user message", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "answer" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "platform-key" });

    const result = await strategy.generate({ prompt: "write a function" });

    expect(result.text).toBe("answer");
    expect(result.usage).toEqual({ inputTokens: 3, outputTokens: 2 });
    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.messages).toEqual([{ role: "user", content: "write a function" }]);
  });

  it("sends systemPrompt as a leading system message when provided", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "ok" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    await strategy.generate({ prompt: "p", systemPrompt: "Be concise." });

    const call = mockChatCompletionsCreate.mock.calls[0]![0];
    expect(call.messages).toEqual([
      { role: "system", content: "Be concise." },
      { role: "user", content: "p" },
    ]);
  });

  it("throws UnsupportedAttachmentError for ANY attachment (no vision endpoint)", async () => {
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    await expect(
      strategy.generate({
        prompt: "p",
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      }),
    ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
  });

  it("reports truncated=true when finish_reason is length", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "cut" }, finish_reason: "length" }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 4, completion_tokens: 4096 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k" });
    const result = await strategy.generate({ prompt: "p" });
    expect(result.truncated).toBe(true);
  });

  it("uses the custom baseURL when supplied", async () => {
    mockChatCompletionsCreate.mockResolvedValueOnce({
      choices: [{ message: { content: "" } }],
      model: "deepseek-chat",
      usage: { prompt_tokens: 0, completion_tokens: 0 },
    });
    const strategy = new DeepSeekStrategy({ apiKey: "k", baseURL: "https://internal.deepseek.example" });
    await strategy.generate({ prompt: "p" });
    expect(mockChatCompletionsCreate).toHaveBeenCalledOnce();
  });

  it("validateKey calls models.list without spending tokens", async () => {
    mockModelsList.mockResolvedValueOnce({ data: [] });
    const strategy = new DeepSeekStrategy({});
    await strategy.validateKey("u-key");
    expect(mockModelsList).toHaveBeenCalledOnce();
  });

  it("validateKey wraps SDK rejections as LlmKeyValidationError", async () => {
    mockModelsList.mockRejectedValueOnce(new Error("unauthorized"));
    const strategy = new DeepSeekStrategy({});
    await expect(strategy.validateKey("bad")).rejects.toBeInstanceOf(LlmKeyValidationError);
  });

  it("throws if no apiKey is configured AND none is passed per-call", async () => {
    const strategy = new DeepSeekStrategy({});
    await expect(strategy.generate({ prompt: "x" })).rejects.toThrow(
      /platform API key is not configured/,
    );
  });

  it("hasPlatformKey reflects whether a constructor apiKey was given", () => {
    expect(new DeepSeekStrategy({ apiKey: "k" }).hasPlatformKey()).toBe(true);
    expect(new DeepSeekStrategy({}).hasPlatformKey()).toBe(false);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- deepseek.test.ts`
Expected: FAIL — `../deepseek/index` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/deepseek/index.ts`:

```ts
import OpenAI from "openai";
import { LlmKeyValidationError, UnsupportedAttachmentError } from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

const DEFAULT_BASE_URL = "https://api.deepseek.com";
const FALLBACK_DEFAULT_MODEL = "deepseek-chat";

export interface DeepSeekStrategyOptions {
  apiKey?: string;
  defaultModel?: string;
  /** Override the DeepSeek API base URL. Default `https://api.deepseek.com`. */
  baseURL?: string;
}

/**
 * DeepSeek adapter. DeepSeek's API is OpenAI-compatible, so this reuses
 * the `openai` SDK with `baseURL` pointed at DeepSeek. DeepSeek has no
 * vision-capable endpoint — any attachment throws
 * `UnsupportedAttachmentError` up front rather than an opaque 4xx
 * mid-stream.
 */
export class DeepSeekStrategy implements LlmStrategy {
  readonly providerName = "deepseek";
  readonly capabilities = ["code", "reasoning", "cheap"] as const;
  readonly defaultModel: string;
  private platformClient: OpenAI | null = null;
  private readonly platformApiKey: string | undefined;
  private readonly baseURL: string;

  constructor(opts: DeepSeekStrategyOptions = {}) {
    this.platformApiKey = opts.apiKey?.trim() || undefined;
    this.defaultModel = opts.defaultModel?.trim() || FALLBACK_DEFAULT_MODEL;
    this.baseURL = opts.baseURL?.trim() || DEFAULT_BASE_URL;
  }

  private getPlatformClient(): OpenAI {
    if (!this.platformClient) {
      if (!this.platformApiKey) {
        throw new Error(
          "DeepSeek platform API key is not configured. Pass `apiKey` per call (BYOK) or supply one to the strategy constructor.",
        );
      }
      this.platformClient = new OpenAI({ apiKey: this.platformApiKey, baseURL: this.baseURL });
    }
    return this.platformClient;
  }

  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    const attachments = opts.attachments ?? [];
    if (attachments.length > 0) {
      throw new UnsupportedAttachmentError(
        this.providerName,
        attachments[0]!.mimetype,
        "DeepSeek has no vision-capable endpoint. Switch to a provider with vision support.",
      );
    }

    const client = opts.apiKey
      ? new OpenAI({ apiKey: opts.apiKey, baseURL: this.baseURL })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;

    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: opts.prompt },
        ]
      : [{ role: "user" as const, content: opts.prompt }];

    const response = await client.chat.completions.create({
      model: modelName,
      messages,
      ...(opts.maxTokens !== undefined ? { max_tokens: opts.maxTokens } : {}),
    });

    const raw = response.choices[0]?.message?.content ?? "";
    const text = typeof raw === "string" ? raw : "";

    return {
      text,
      model: response.model,
      usage: {
        inputTokens: response.usage?.prompt_tokens ?? 0,
        outputTokens: response.usage?.completion_tokens ?? 0,
      },
      truncated: response.choices[0]?.finish_reason === "length",
    };
  }

  async validateKey(apiKey: string, _model?: string): Promise<void> {
    const client = new OpenAI({ apiKey, baseURL: this.baseURL });
    try {
      await client.models.list();
    } catch (cause) {
      throw new LlmKeyValidationError(this.providerName, cause);
    }
  }

  hasPlatformKey(): boolean {
    return this.platformApiKey !== undefined;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test && npm run typecheck && npm run build`
Expected: PASS. `npm run build` now succeeds — both `chatgpt` and `deepseek` tsup entries resolve.

- [ ] **Step 5: Commit**

```bash
git add src/deepseek/index.ts src/__tests__/deepseek.test.ts
git commit -m "feat: add DeepSeekStrategy (OpenAI-compatible adapter, no vision)"
```

---

### Task 5: `TaskRouter`

**Files:**
- Create: `src/task-router.ts`
- Test: `src/__tests__/task-router.test.ts`

**Interfaces:**
- Consumes: `LlmRegistry` (`src/registry.ts`), `LlmStrategy.hasPlatformKey`/`capabilities` (Task 1), `NoAvailableProviderError` (Task 1).
- Produces: `KNOWN_CAPABILITY_TAGS`, `Subtask`, `RoutingDecision`, `ProviderDescriptor`, `TaskRouterOptions`, `RouteOptions`, `TaskRouter` with `route()` and `listAvailableProviders()` — Task 6 depends on all of these exact names and shapes.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/task-router.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { NoAvailableProviderError } from "../errors";
import { LlmRegistry } from "../registry";
import { TaskRouter } from "../task-router";
import type { LlmStrategy } from "../types";

function makeStrategy(
  name: string,
  opts: {
    capabilities?: readonly string[];
    hasPlatformKey?: () => boolean;
    generate?: ReturnType<typeof vi.fn>;
  } = {},
): LlmStrategy {
  return {
    providerName: name,
    defaultModel: `${name}-default`,
    capabilities: opts.capabilities,
    generate: opts.generate ?? vi.fn(),
    validateKey: vi.fn(),
    hasPlatformKey: opts.hasPlatformKey,
  };
}

function fallbackResponse(json: string) {
  return { text: json, model: "m", usage: { inputTokens: 1, outputTokens: 1 }, truncated: false };
}

describe("TaskRouter.route", () => {
  it("routes via the rule stage when one provider uniquely wins the tag match", async () => {
    const vision = makeStrategy("vision-co", { capabilities: ["vision"], hasPlatformKey: () => true });
    const code = makeStrategy("code-co", { capabilities: ["code"], hasPlatformKey: () => true });
    const registry = new LlmRegistry({ strategies: [vision, code], platform: "code-co" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route([
      { id: "t1", description: "describe an image", requiredCapabilities: ["vision"] },
    ]);

    expect(decisions).toEqual([
      { subtaskId: "t1", provider: "vision-co", method: "rule", rationale: expect.stringContaining("vision") },
    ]);
  });

  it("falls back to the LLM classifier on a tag-score tie", async () => {
    const generate = vi.fn().mockResolvedValue(fallbackResponse('{"provider": "b", "rationale": "better fit"}'));
    const a = makeStrategy("a", { capabilities: ["code"], hasPlatformKey: () => true });
    const b = makeStrategy("b", { capabilities: ["code"], hasPlatformKey: () => true, generate });
    const registry = new LlmRegistry({ strategies: [a, b], platform: "b" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route([
      { id: "t1", description: "write a function", requiredCapabilities: ["code"] },
    ]);

    expect(decisions[0]).toMatchObject({ subtaskId: "t1", provider: "b", method: "llm-fallback" });
    expect(generate).toHaveBeenCalledOnce();
  });

  it("falls back to the LLM classifier when no provider has any matching tag", async () => {
    const generate = vi.fn().mockResolvedValue(fallbackResponse('{"provider": "only"}'));
    const only = makeStrategy("only", { capabilities: ["cheap"], hasPlatformKey: () => true, generate });
    const other = makeStrategy("other", { capabilities: ["reasoning"], hasPlatformKey: () => true });
    const registry = new LlmRegistry({ strategies: [only, other], platform: "only" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route([
      { id: "t1", description: "translate text", requiredCapabilities: ["multilingual"] },
    ]);

    expect(decisions[0]!.method).toBe("llm-fallback");
    expect(decisions[0]!.provider).toBe("only");
  });

  it("routes straight to LLM fallback when a subtask has no requiredCapabilities (2+ available providers)", async () => {
    const generate = vi.fn().mockResolvedValue(fallbackResponse('{"provider": "p"}'));
    const p = makeStrategy("p", { capabilities: ["code"], hasPlatformKey: () => true, generate });
    const q = makeStrategy("q", { capabilities: ["vision"], hasPlatformKey: () => true });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "p" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route([{ id: "t1", description: "generic task" }]);
    expect(decisions[0]!.method).toBe("llm-fallback");
  });

  it("routes directly to the only available provider without an LLM-fallback call", async () => {
    const generate = vi.fn();
    const solo = makeStrategy("solo", { hasPlatformKey: () => true, generate });
    const registry = new LlmRegistry({ strategies: [solo], platform: "solo" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route([{ id: "t1", description: "generic task" }]);

    expect(decisions[0]).toMatchObject({ provider: "solo", method: "rule" });
    expect(generate).not.toHaveBeenCalled();
  });

  it("treats a strategy without hasPlatformKey as unavailable by default", async () => {
    const noCheck = makeStrategy("no-check", { capabilities: ["code"] });
    const registry = new LlmRegistry({ strategies: [noCheck], platform: null });
    const router = new TaskRouter({ registry });

    await expect(router.route([{ id: "t1", description: "x" }])).rejects.toBeInstanceOf(
      NoAvailableProviderError,
    );
  });

  it("an apiKeys entry makes an otherwise-unchecked provider available", async () => {
    const generate = vi.fn().mockResolvedValue(fallbackResponse('{"provider": "byok"}'));
    const byok = makeStrategy("byok", { generate });
    const registry = new LlmRegistry({ strategies: [byok], platform: null });
    const router = new TaskRouter({ registry, metaProvider: "byok" });

    const decisions = await router.route([{ id: "t1", description: "x" }], {
      apiKeys: { byok: "user-key" },
    });

    expect(decisions[0]!.provider).toBe("byok");
  });

  it("providerOverrides win over the strategy's own reported availability and capabilities", async () => {
    const p = makeStrategy("p", { hasPlatformKey: () => false });
    const q = makeStrategy("q", { hasPlatformKey: () => true, capabilities: ["code"] });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "q" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route(
      [{ id: "t1", description: "x", requiredCapabilities: ["vision"] }],
      { providerOverrides: [{ provider: "p", available: true, capabilities: ["vision"] }] },
    );

    expect(decisions[0]).toMatchObject({ method: "rule", provider: "p" });
  });

  it("throws NoAvailableProviderError when no provider is available at all", async () => {
    const p = makeStrategy("p", { hasPlatformKey: () => false });
    const registry = new LlmRegistry({ strategies: [p], platform: null });
    const router = new TaskRouter({ registry });

    await expect(router.route([{ id: "t1", description: "x" }])).rejects.toBeInstanceOf(
      NoAvailableProviderError,
    );
  });

  it("throws NoAvailableProviderError when the LLM fallback names a provider outside the available set", async () => {
    const generate = vi.fn().mockResolvedValue(fallbackResponse('{"provider": "ghost"}'));
    const p = makeStrategy("p", { hasPlatformKey: () => true, generate });
    const q = makeStrategy("q", { hasPlatformKey: () => true });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "p" });
    const router = new TaskRouter({ registry });

    await expect(router.route([{ id: "t1", description: "x" }])).rejects.toBeInstanceOf(
      NoAvailableProviderError,
    );
  });

  it("strips a fenced code block from the LLM-fallback response before parsing", async () => {
    const generate = vi
      .fn()
      .mockResolvedValue(fallbackResponse('```json\n{"provider": "p", "rationale": "ok"}\n```'));
    const p = makeStrategy("p", { hasPlatformKey: () => true, generate });
    const q = makeStrategy("q", { hasPlatformKey: () => true });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "p" });
    const router = new TaskRouter({ registry });

    const decisions = await router.route([{ id: "t1", description: "x" }]);
    expect(decisions[0]!.provider).toBe("p");
  });
});

describe("TaskRouter.listAvailableProviders", () => {
  it("returns only available providers with their capabilities", async () => {
    const yes = makeStrategy("yes", { capabilities: ["code"], hasPlatformKey: () => true });
    const no = makeStrategy("no", { capabilities: ["vision"], hasPlatformKey: () => false });
    const registry = new LlmRegistry({ strategies: [yes, no], platform: "yes" });
    const router = new TaskRouter({ registry });

    const available = await router.listAvailableProviders();
    expect(available).toEqual([{ name: "yes", capabilities: ["code"] }]);
  });

  it("honors apiKeys and providerOverrides the same way route() does", async () => {
    const p = makeStrategy("p", { hasPlatformKey: () => false });
    const registry = new LlmRegistry({ strategies: [p], platform: null });
    const router = new TaskRouter({ registry });

    const available = await router.listAvailableProviders({ apiKeys: { p: "k" } });
    expect(available).toEqual([{ name: "p", capabilities: [] }]);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- task-router.test.ts`
Expected: FAIL — `../task-router` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/task-router.ts`:

```ts
import { NoAvailableProviderError } from "./errors";
import type { LlmRegistry } from "./registry";
import type { LlmStrategy } from "./types";

export const KNOWN_CAPABILITY_TAGS = [
  "vision",
  "code",
  "long-context",
  "cheap",
  "reasoning",
  "multilingual",
] as const;

export type CapabilityTag = (typeof KNOWN_CAPABILITY_TAGS)[number];

export interface Subtask {
  id: string;
  description: string;
  requiredCapabilities?: readonly string[];
}

export interface RoutingDecision {
  subtaskId: string;
  provider: string;
  model?: string;
  method: "rule" | "llm-fallback";
  rationale?: string;
}

export interface ProviderDescriptor {
  provider: string;
  capabilities?: readonly string[];
  available?: boolean;
}

export interface TaskRouterOptions {
  registry: LlmRegistry;
  /** Defaults to `registry.getPlatform()`'s provider name. */
  metaProvider?: string;
}

export interface RouteOptions {
  apiKeys?: Record<string, string>;
  providerOverrides?: readonly ProviderDescriptor[];
}

interface ResolvedProvider {
  name: string;
  strategy: LlmStrategy;
  capabilities: readonly string[];
  available: boolean;
}

export class TaskRouter {
  private readonly registry: LlmRegistry;
  private readonly metaProviderName: string | undefined;

  constructor(opts: TaskRouterOptions) {
    this.registry = opts.registry;
    this.metaProviderName = opts.metaProvider;
  }

  async route(subtasks: readonly Subtask[], opts: RouteOptions = {}): Promise<RoutingDecision[]> {
    const providers = this.resolveProviders(opts);
    const available = providers.filter((p) => p.available);

    const decisions: RoutingDecision[] = [];
    for (const subtask of subtasks) {
      if (available.length === 0) {
        throw new NoAvailableProviderError(subtask.id);
      }
      decisions.push(await this.routeOne(subtask, available));
    }
    return decisions;
  }

  async listAvailableProviders(
    opts: RouteOptions = {},
  ): Promise<Array<{ name: string; capabilities: readonly string[] }>> {
    return this.resolveProviders(opts)
      .filter((p) => p.available)
      .map((p) => ({ name: p.name, capabilities: p.capabilities }));
  }

  private resolveProviders(opts: RouteOptions): ResolvedProvider[] {
    const overrideByName = new Map((opts.providerOverrides ?? []).map((o) => [o.provider, o]));
    return this.registry.listProviderNames().map((name) => {
      const strategy = this.registry.get(name);
      const override = overrideByName.get(name);
      const capabilities = override?.capabilities ?? strategy.capabilities ?? [];

      let available: boolean;
      if (override?.available !== undefined) {
        available = override.available;
      } else if (opts.apiKeys?.[name]) {
        available = true;
      } else if (strategy.hasPlatformKey) {
        available = strategy.hasPlatformKey();
      } else {
        available = false;
      }

      return { name, strategy, capabilities, available };
    });
  }

  private async routeOne(subtask: Subtask, available: ResolvedProvider[]): Promise<RoutingDecision> {
    if (available.length === 1) {
      return {
        subtaskId: subtask.id,
        provider: available[0]!.name,
        method: "rule",
        rationale: "Only available provider",
      };
    }

    const required = subtask.requiredCapabilities ?? [];

    if (required.length > 0) {
      const scored = available
        .map((p) => ({
          provider: p,
          score: p.capabilities.filter((c) => required.includes(c)).length,
        }))
        .filter((s) => s.score > 0)
        .sort((a, b) => b.score - a.score);

      if (scored.length > 0 && (scored.length === 1 || scored[0]!.score > scored[1]!.score)) {
        const winner = scored[0]!.provider;
        const matched = winner.capabilities.filter((c) => required.includes(c));
        return {
          subtaskId: subtask.id,
          provider: winner.name,
          method: "rule",
          rationale: `Tag match: ${matched.join(", ")}`,
        };
      }
    }

    return this.llmFallback(subtask, available);
  }

  private getMetaProviderName(): string {
    return this.metaProviderName ?? this.registry.getPlatform().providerName;
  }

  private async llmFallback(subtask: Subtask, available: ResolvedProvider[]): Promise<RoutingDecision> {
    const metaStrategy = this.registry.get(this.getMetaProviderName());
    const candidateList = available
      .map((p) => `- ${p.name} (capabilities: ${p.capabilities.join(", ") || "none listed"})`)
      .join("\n");

    const prompt =
      `Pick the best provider for this subtask from the list below. ` +
      `Respond with ONLY a JSON object: {"provider": string, "model"?: string, "rationale"?: string}.\n\n` +
      `Subtask: ${subtask.description}\n\nAvailable providers:\n${candidateList}`;

    const response = await metaStrategy.generate({ prompt });
    const stripped = response.text.trim().replace(/^```(?:json)?\n?/, "").replace(/```$/, "");
    const parsed = JSON.parse(stripped) as { provider: string; model?: string; rationale?: string };

    if (!available.some((p) => p.name === parsed.provider)) {
      throw new NoAvailableProviderError(subtask.id);
    }

    return {
      subtaskId: subtask.id,
      provider: parsed.provider,
      model: parsed.model,
      method: "llm-fallback",
      rationale: parsed.rationale,
    };
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- task-router.test.ts && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/task-router.ts src/__tests__/task-router.test.ts
git commit -m "feat: add TaskRouter (capability-based provider routing)"
```

---

### Task 6: `Orchestrator` — decompose, route, execute-with-critique (sequential)

**Files:**
- Create: `src/orchestrator.ts`
- Test: `src/__tests__/orchestrator.test.ts`

**Interfaces:**
- Consumes: `TaskRouter` (Task 5) — `route()`, `listAvailableProviders()`; `TaskDecompositionError` (Task 1).
- Produces: `RunOptions`, `SubtaskResult`, `OrchestratorResult`, `Orchestrator` with `run(taskText, opts)` — Task 7 and Task 8 extend this same class.

- [ ] **Step 1: Write the failing tests**

Create `src/__tests__/orchestrator.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { LlmRegistry } from "../registry";
import { Orchestrator } from "../orchestrator";
import type { LlmResponse, LlmStrategy } from "../types";

function ok(text: string, finish: "stop" | "length" = "stop"): LlmResponse {
  return {
    text,
    model: "m",
    usage: { inputTokens: 1, outputTokens: 1 },
    truncated: finish === "length",
  };
}

function makeStrategy(
  name: string,
  opts: {
    capabilities?: readonly string[];
    hasPlatformKey?: () => boolean;
    generate: ReturnType<typeof vi.fn>;
  },
): LlmStrategy {
  return {
    providerName: name,
    defaultModel: `${name}-default`,
    capabilities: opts.capabilities,
    generate: opts.generate,
    validateKey: vi.fn(),
    hasPlatformKey: opts.hasPlatformKey ?? (() => true),
  };
}

const ONE_SUBTASK_JSON = '[{"id": "t1", "description": "do the thing"}]';

describe("Orchestrator.run — decompose", () => {
  it("parses a valid JSON subtask array and executes it", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON)) // decompose
      .mockResolvedValueOnce(ok("done")) // subtask generate
      .mockResolvedValueOnce(ok('{"approved": true}')); // self-critique
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("build a widget");

    expect(result.subtasks).toHaveLength(1);
    expect(result.subtasks[0]).toMatchObject({ result: "done", rounds: 0, unresolved: false });
  });

  it("strips a fenced code block before parsing", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(`\`\`\`json\n${ONE_SUBTASK_JSON}\n\`\`\``))
      .mockResolvedValueOnce(ok("done"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("build a widget");
    expect(result.subtasks).toHaveLength(1);
  });

  it("retries decompose once on invalid JSON, then throws TaskDecompositionError", async () => {
    const generate = vi.fn().mockResolvedValue(ok("not json at all"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    await expect(orchestrator.run("build a widget")).rejects.toThrow(
      /Failed to decompose the task/,
    );
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("truncates subtasks beyond maxSubtasks and reports the count", async () => {
    const threeSubtasks = JSON.stringify([
      { id: "t1", description: "a" },
      { id: "t2", description: "b" },
      { id: "t3", description: "c" },
    ]);
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(threeSubtasks))
      .mockResolvedValue(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("build many things", { maxSubtasks: 2 });
    expect(result.subtasks).toHaveLength(2);
    expect(result.truncatedSubtaskCount).toBe(1);
  });
});

describe("Orchestrator.run — critique loop", () => {
  it("returns unresolved:false when approved on the first critique", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("first answer"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");
    expect(result.subtasks[0]).toMatchObject({ result: "first answer", rounds: 0, unresolved: false });
  });

  it("retries once and resolves within a maxRounds=2 budget", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("draft"))
      .mockResolvedValueOnce(ok('{"approved": false, "feedback": "too short"}'))
      .mockResolvedValueOnce(ok("revised"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { maxRounds: 2 });
    expect(result.subtasks[0]).toMatchObject({ result: "revised", rounds: 1, unresolved: false });
  });

  it("marks unresolved:true when the critique loop exhausts maxRounds", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("draft"))
      .mockResolvedValueOnce(ok('{"approved": false, "feedback": "nope"}'))
      .mockResolvedValueOnce(ok("revised once"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { maxRounds: 1 });
    expect(result.subtasks[0]).toMatchObject({ result: "revised once", rounds: 1, unresolved: true });
  });

  it("skips the critique step entirely when maxRounds is 0", async () => {
    const generate = vi.fn().mockResolvedValueOnce(ok(ONE_SUBTASK_JSON)).mockResolvedValueOnce(ok("done"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { maxRounds: 0 });
    expect(result.subtasks[0]).toMatchObject({ result: "done", rounds: 0, unresolved: false });
    expect(generate).toHaveBeenCalledTimes(2); // decompose + one generate, no critique call
  });

  it("uses the next-highest-scoring available provider for cross critique on a rule-routed subtask", async () => {
    const subtaskJson =
      '[{"id": "t1", "description": "x", "requiredCapabilities": ["code", "reasoning"]}]';
    const bestGenerate = vi
      .fn()
      .mockResolvedValueOnce(ok(subtaskJson)) // decompose (via platform = best)
      .mockResolvedValueOnce(ok("answer")); // subtask generate (routed to best)
    const criticGenerate = vi.fn().mockResolvedValueOnce(ok('{"approved": true}'));
    const best = makeStrategy("best", { capabilities: ["code", "reasoning"], generate: bestGenerate });
    const second = makeStrategy("second", { capabilities: ["code"], generate: criticGenerate });
    const registry = new LlmRegistry({ strategies: [best, second], platform: "best" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { critique: "cross" });

    expect(criticGenerate).toHaveBeenCalledOnce();
    expect(result.subtasks[0]).toMatchObject({ unresolved: false });
  });

  it("falls back to self-critique when no second available provider exists", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("answer"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { critique: "cross" });
    expect(result.subtasks[0]).toMatchObject({ unresolved: false });
    expect(generate).toHaveBeenCalledTimes(3); // decompose, generate, self-critique (only one provider exists)
  });
});

describe("Orchestrator.run — partial failure isolation", () => {
  it("isolates one subtask's routing failure without affecting others", async () => {
    // Two AVAILABLE providers so the single-provider shortcut doesn't apply —
    // t1 uniquely tag-matches "p"; t2 has no tags, so it goes through
    // LLM-fallback and gets an invalid answer. Both routing calls go through
    // the platform provider "p", so its generate() must handle three
    // different prompt shapes (decompose, t2's routing fallback, t1's
    // execution) plus t1's critique — matched by content, not call order,
    // since t1 and t2 execute concurrently (default maxConcurrency).
    const twoSubtasks = JSON.stringify([
      { id: "t1", description: "codey thing", requiredCapabilities: ["code"] },
      { id: "t2", description: "generic thing" },
    ]);
    const generate = vi.fn().mockImplementation(({ prompt }: { prompt: string }) => {
      if (prompt.includes("Split the following task")) return Promise.resolve(ok(twoSubtasks));
      if (prompt.includes("Pick the best provider")) return Promise.resolve(ok('{"provider": "ghost"}'));
      if (prompt.includes("Does this fully satisfy")) return Promise.resolve(ok('{"approved": true}'));
      return Promise.resolve(ok("t1 done"));
    });
    const p = makeStrategy("p", { capabilities: ["code"], generate });
    const q = makeStrategy("q", { capabilities: ["vision"], generate: vi.fn().mockResolvedValue(ok("unused")) });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");

    const t1 = result.subtasks.find((s) => s.subtask.id === "t1")!;
    const t2 = result.subtasks.find((s) => s.subtask.id === "t2")!;
    expect(t1).toMatchObject({ result: "t1 done", unresolved: false });
    expect(t2.error).toMatch(/No available provider/);
    expect(t2.decision).toBeUndefined();
    expect(t2.unresolved).toBe(true);
  });

  it("isolates one subtask's generate() failure without affecting others", async () => {
    // Single provider — the shortcut routes both subtasks to it directly
    // (no fallback call), so the only prompts "p".generate() ever sees are
    // decompose, t1's execution (rejected), t2's execution, and t2's
    // critique. Matched by content since t1/t2 run concurrently.
    const twoSubtasks = JSON.stringify([
      { id: "t1", description: "task one" },
      { id: "t2", description: "task two" },
    ]);
    const generate = vi.fn().mockImplementation(({ prompt }: { prompt: string }) => {
      if (prompt.includes("Split the following task")) return Promise.resolve(ok(twoSubtasks));
      if (prompt === "task one") return Promise.reject(new Error("network down"));
      if (prompt === "task two") return Promise.resolve(ok("t2 done"));
      return Promise.resolve(ok('{"approved": true}')); // t2's self-critique
    });
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");

    const t1 = result.subtasks.find((s) => s.subtask.id === "t1")!;
    const t2 = result.subtasks.find((s) => s.subtask.id === "t2")!;
    expect(t1.error).toBe("network down");
    expect(t1.unresolved).toBe(true);
    expect(t2).toMatchObject({ result: "t2 done", unresolved: false });
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- orchestrator.test.ts`
Expected: FAIL — `../orchestrator` doesn't exist yet.

- [ ] **Step 3: Implement**

Create `src/orchestrator.ts`:

```ts
import { TaskDecompositionError } from "./errors";
import type { LlmRegistry } from "./registry";
import { KNOWN_CAPABILITY_TAGS, TaskRouter } from "./task-router";
import type { ProviderDescriptor, RoutingDecision, Subtask } from "./task-router";

export interface RunOptions {
  apiKeys?: Record<string, string>;
  providerOverrides?: readonly ProviderDescriptor[];
  /** Critique/retry rounds per subtask. Default 1. */
  maxRounds?: number;
  /** Default "self". */
  critique?: "self" | "cross";
  /** Default false. */
  synthesize?: boolean;
  /** Defaults to `registry.getPlatform()`'s provider name. */
  metaProvider?: string;
  /** Independent subtasks run concurrently, capped. Default 3. */
  maxConcurrency?: number;
  /** Upper bound on decompose() fan-out. Default 20. */
  maxSubtasks?: number;
}

export interface SubtaskResult {
  subtask: Subtask;
  /** Absent when routing itself failed — see `error`. */
  decision?: RoutingDecision;
  result: string;
  rounds: number;
  unresolved: boolean;
  error?: string;
}

export interface OrchestratorResult {
  subtasks: SubtaskResult[];
  final?: string;
  truncatedSubtaskCount?: number;
}

const DEFAULT_MAX_ROUNDS = 1;
const DEFAULT_MAX_SUBTASKS = 20;

interface AvailableProvider {
  name: string;
  capabilities: readonly string[];
}

function stripFence(text: string): string {
  return text.trim().replace(/^```(?:json)?\n?/, "").replace(/```$/, "");
}

export class Orchestrator {
  protected readonly registry: LlmRegistry;
  protected readonly taskRouter: TaskRouter;

  constructor(opts: { registry: LlmRegistry; taskRouter?: TaskRouter }) {
    this.registry = opts.registry;
    this.taskRouter = opts.taskRouter ?? new TaskRouter({ registry: opts.registry });
  }

  async run(taskText: string, opts: RunOptions = {}): Promise<OrchestratorResult> {
    const maxSubtasks = opts.maxSubtasks ?? DEFAULT_MAX_SUBTASKS;
    const { subtasks, truncatedSubtaskCount } = await this.decompose(
      taskText,
      opts.metaProvider,
      maxSubtasks,
    );

    const availableProviders = await this.taskRouter.listAvailableProviders({
      apiKeys: opts.apiKeys,
      providerOverrides: opts.providerOverrides,
    });

    const results: SubtaskResult[] = [];
    for (const subtask of subtasks) {
      results.push(await this.routeAndExecute(subtask, opts, availableProviders));
    }

    const result: OrchestratorResult = { subtasks: results };
    if (truncatedSubtaskCount > 0) result.truncatedSubtaskCount = truncatedSubtaskCount;
    return result;
  }

  private getMetaProviderName(metaProvider?: string): string {
    return metaProvider ?? this.registry.getPlatform().providerName;
  }

  private async decompose(
    taskText: string,
    metaProvider: string | undefined,
    maxSubtasks: number,
  ): Promise<{ subtasks: Subtask[]; truncatedSubtaskCount: number }> {
    const strategy = this.registry.get(this.getMetaProviderName(metaProvider));
    const basePrompt =
      `Split the following task into independent subtasks. Respond with ONLY a JSON array ` +
      `of objects: {"id": string, "description": string, "requiredCapabilities"?: string[]}. ` +
      `requiredCapabilities may only use these tags: ${KNOWN_CAPABILITY_TAGS.join(", ")}.\n\n` +
      `Task: ${taskText}`;

    let subtasks = await this.tryParseSubtasks(strategy, basePrompt);
    if (!subtasks) {
      subtasks = await this.tryParseSubtasks(
        strategy,
        `${basePrompt}\n\nYour previous response was not valid JSON. Return only the JSON array, no prose.`,
      );
    }
    if (!subtasks) {
      throw new TaskDecompositionError(
        new Error("Model did not return a parseable JSON array after one retry."),
      );
    }

    const truncatedSubtaskCount = Math.max(0, subtasks.length - maxSubtasks);
    return { subtasks: subtasks.slice(0, maxSubtasks), truncatedSubtaskCount };
  }

  private async tryParseSubtasks(
    strategy: { generate: (opts: { prompt: string }) => Promise<{ text: string }> },
    prompt: string,
  ): Promise<Subtask[] | null> {
    const response = await strategy.generate({ prompt });
    try {
      const parsed = JSON.parse(stripFence(response.text));
      if (!Array.isArray(parsed)) return null;
      return parsed.map(
        (item: { id: string; description: string; requiredCapabilities?: string[] }) => ({
          id: item.id,
          description: item.description,
          requiredCapabilities: item.requiredCapabilities,
        }),
      );
    } catch {
      return null;
    }
  }

  protected async routeAndExecute(
    subtask: Subtask,
    opts: RunOptions,
    availableProviders: AvailableProvider[],
  ): Promise<SubtaskResult> {
    let decision: RoutingDecision;
    try {
      const decisions = await this.taskRouter.route([subtask], {
        apiKeys: opts.apiKeys,
        providerOverrides: opts.providerOverrides,
      });
      decision = decisions[0]!;
    } catch (cause) {
      return {
        subtask,
        result: "",
        rounds: 0,
        unresolved: true,
        error: cause instanceof Error ? cause.message : String(cause),
      };
    }

    try {
      return await this.executeSubtask(subtask, decision, opts, availableProviders);
    } catch (cause) {
      return {
        subtask,
        decision,
        result: "",
        rounds: 0,
        unresolved: true,
        error: cause instanceof Error ? cause.message : String(cause),
      };
    }
  }

  private async executeSubtask(
    subtask: Subtask,
    decision: RoutingDecision,
    opts: RunOptions,
    availableProviders: AvailableProvider[],
  ): Promise<SubtaskResult> {
    const maxRounds = opts.maxRounds ?? DEFAULT_MAX_ROUNDS;
    const strategy = this.registry.get(decision.provider);
    const apiKey = opts.apiKeys?.[decision.provider];

    let output = await this.generateForSubtask(strategy, subtask, decision, apiKey);
    if (maxRounds === 0) {
      return { subtask, decision, result: output, rounds: 0, unresolved: false };
    }

    let rounds = 0;
    while (rounds < maxRounds) {
      const verdict = await this.critique(subtask, output, decision, opts, availableProviders);
      if (verdict.approved) {
        return { subtask, decision, result: output, rounds, unresolved: false };
      }
      rounds += 1;
      output = await this.generateForSubtask(strategy, subtask, decision, apiKey, verdict.feedback);
    }

    return { subtask, decision, result: output, rounds, unresolved: true };
  }

  private async generateForSubtask(
    strategy: ReturnType<LlmRegistry["get"]>,
    subtask: Subtask,
    decision: RoutingDecision,
    apiKey: string | undefined,
    feedback?: string,
  ): Promise<string> {
    const prompt = feedback
      ? `${subtask.description}\n\nA reviewer rejected your previous attempt with this feedback: ${feedback}\nTry again, addressing the feedback.`
      : subtask.description;
    const response = await strategy.generate({ prompt, model: decision.model, apiKey });
    return response.text;
  }

  private async critique(
    subtask: Subtask,
    output: string,
    decision: RoutingDecision,
    opts: RunOptions,
    availableProviders: AvailableProvider[],
  ): Promise<{ approved: boolean; feedback?: string }> {
    const mode = opts.critique ?? "self";
    const criticName =
      mode === "cross" ? this.pickCrossCritic(subtask, decision, availableProviders) : decision.provider;
    const criticStrategy = this.registry.get(criticName);
    const criticApiKey = opts.apiKeys?.[criticName];

    const prompt =
      `Subtask: ${subtask.description}\n\nCandidate answer:\n${output}\n\n` +
      `Does this fully satisfy the subtask? Respond with ONLY JSON: {"approved": boolean, "feedback"?: string}.`;

    const response = await criticStrategy.generate({ prompt, apiKey: criticApiKey });
    try {
      const parsed = JSON.parse(stripFence(response.text));
      return { approved: Boolean(parsed.approved), feedback: parsed.feedback };
    } catch {
      // An unparseable critique verdict is not evidence of approval —
      // force a retry rather than silently reporting success.
      return { approved: false, feedback: "The reviewer's response was not valid JSON." };
    }
  }

  private pickCrossCritic(
    subtask: Subtask,
    decision: RoutingDecision,
    availableProviders: AvailableProvider[],
  ): string {
    const others = availableProviders.filter((p) => p.name !== decision.provider);
    if (others.length === 0) return decision.provider;

    if (decision.method === "rule") {
      const required = subtask.requiredCapabilities ?? [];
      const scored = others
        .map((p) => ({ name: p.name, score: p.capabilities.filter((c) => required.includes(c)).length }))
        .sort((a, b) => b.score - a.score);
      return scored[0]!.name;
    }

    return others[0]!.name;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- orchestrator.test.ts && npm run typecheck`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.ts src/__tests__/orchestrator.test.ts
git commit -m "feat: add Orchestrator (decompose, route, execute-with-critique loop)"
```

---

### Task 7: `Orchestrator` — concurrency cap

**Files:**
- Modify: `src/orchestrator.ts`
- Modify: `src/__tests__/orchestrator.test.ts`

**Interfaces:**
- Consumes: `Orchestrator.routeAndExecute` (Task 6, now called through a concurrency-limited runner instead of a plain `for` loop).
- Produces: no new public names — `run()`'s behavior changes from sequential to concurrency-capped, same signature.

- [ ] **Step 1: Write the failing test**

Add to `src/__tests__/orchestrator.test.ts` (new top-level `describe`):

```ts
describe("Orchestrator.run — concurrency", () => {
  it("never runs more than maxConcurrency subtasks at once", async () => {
    // Single registered provider -> TaskRouter's single-available-provider
    // shortcut routes every subtask directly, with zero extra generate()
    // calls for routing. Combined with maxRounds: 0 (no critique calls),
    // the only generate() calls are: one decompose call, then exactly one
    // execution call per subtask -- which is what this test measures.
    let inFlight = 0;
    let maxInFlight = 0;
    const releases: Array<() => void> = [];

    const fourSubtasks = JSON.stringify([
      { id: "t1", description: "a" },
      { id: "t2", description: "b" },
      { id: "t3", description: "c" },
      { id: "t4", description: "d" },
    ]);

    const generate = vi.fn().mockImplementation(() => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<LlmResponse>((resolve) => {
        releases.push(() => {
          inFlight--;
          resolve(ok("done"));
        });
      });
    });
    generate.mockImplementationOnce(() => Promise.resolve(ok(fourSubtasks)));

    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const runPromise = orchestrator.run("task", { maxConcurrency: 2, maxRounds: 0 });

    // Let the decompose call resolve and the first wave of subtask
    // generate() calls start (each hangs until its release() fires).
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(inFlight).toBe(2);

    releases.splice(0).forEach((release) => release());
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(inFlight).toBe(2);

    releases.splice(0).forEach((release) => release());
    await runPromise;

    expect(maxInFlight).toBe(2);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- orchestrator.test.ts -t concurrency`
Expected: FAIL — today's `run()` uses a plain sequential `for` loop, so each
subtask's `generate()` call is started and fully awaited before the next
one starts. `inFlight` never exceeds 1, so the test hangs waiting for
`inFlight` to reach 2 (the first `await Promise.resolve()` flush only ever
sees one call in flight) until the test's timeout fails it. Confirm the
failure, then move to Step 3.

- [ ] **Step 3: Implement**

In `src/orchestrator.ts`, add this helper above the `Orchestrator` class:

```ts
async function runWithConcurrency<T, R>(
  items: readonly T[],
  limit: number,
  fn: (item: T, index: number) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;

  async function worker(): Promise<void> {
    while (next < items.length) {
      const index = next++;
      results[index] = await fn(items[index]!, index);
    }
  }

  const workerCount = Math.min(Math.max(limit, 1), items.length || 1);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}
```

Then change `run()`'s sequential loop from:

```ts
    const results: SubtaskResult[] = [];
    for (const subtask of subtasks) {
      results.push(await this.routeAndExecute(subtask, opts, availableProviders));
    }
```

to:

```ts
    const maxConcurrency = opts.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY;
    const results = await runWithConcurrency(subtasks, maxConcurrency, (subtask) =>
      this.routeAndExecute(subtask, opts, availableProviders),
    );
```

Add the constant next to the other defaults:

```ts
const DEFAULT_MAX_CONCURRENCY = 3;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- orchestrator.test.ts && npm run typecheck`
Expected: PASS, including all Task 6 tests (concurrency=1 in most of them since they use one subtask, or the default cap of 3 which never throttles a 1-2 subtask run) and the new concurrency test showing `maxInFlight === 2`.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.ts src/__tests__/orchestrator.test.ts
git commit -m "feat: cap Orchestrator subtask concurrency with maxConcurrency"
```

---

### Task 8: `Orchestrator` — opt-in synthesis

**Files:**
- Modify: `src/orchestrator.ts`
- Modify: `src/__tests__/orchestrator.test.ts`

**Interfaces:**
- Consumes: `OrchestratorResult.final` (already declared in Task 6).
- Produces: `run()` now populates `final` when `opts.synthesize === true`.

- [ ] **Step 1: Write the failing tests**

Add to `src/__tests__/orchestrator.test.ts`:

```ts
describe("Orchestrator.run — synthesis", () => {
  it("does not call synthesis or set `final` when synthesize is false (default)", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("done"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");
    expect(result.final).toBeUndefined();
    expect(generate).toHaveBeenCalledTimes(3);
  });

  it("makes one extra generate() call and sets `final` when synthesize is true", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("done"))
      .mockResolvedValueOnce(ok('{"approved": true}'))
      .mockResolvedValueOnce(ok("Here is the combined answer."));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { synthesize: true });
    expect(result.final).toBe("Here is the combined answer.");
    expect(generate).toHaveBeenCalledTimes(4);

    const synthesisCall = generate.mock.calls[3]![0];
    expect(synthesisCall.prompt).toContain("do the thing");
    expect(synthesisCall.prompt).toContain("unresolved: false");
  });

  it("notes unresolved and errored subtasks explicitly in the synthesis prompt", async () => {
    const twoSubtasks = JSON.stringify([
      { id: "t1", description: "a" },
      { id: "t2", description: "b" },
    ]);
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(twoSubtasks)) // decompose
      .mockResolvedValueOnce(ok("draft")) // t1 generate
      .mockResolvedValueOnce(ok('{"approved": false, "feedback": "no"}')) // t1 critique rejects
      .mockResolvedValueOnce(ok("still bad")) // t1 retry (maxRounds=1 default -> exhausted)
      .mockRejectedValueOnce(new Error("boom")) // t2 generate fails
      .mockResolvedValueOnce(ok("synthesis text")); // synthesis
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });
    // maxConcurrency: 1 forces the single-worker sequential path, so the
    // mock's call order below is deterministic (t1 fully settles before t2 starts).
    const result = await orchestrator.run("task", { synthesize: true, maxConcurrency: 1 });

    expect(result.final).toBe("synthesis text");
    const synthesisCall = generate.mock.calls[generate.mock.calls.length - 1]![0];
    expect(synthesisCall.prompt).toContain("unresolved: true");
    expect(synthesisCall.prompt).toContain("error: boom");
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- orchestrator.test.ts -t synthesis`
Expected: FAIL — `result.final` is always `undefined` today.

- [ ] **Step 3: Implement**

In `src/orchestrator.ts`, change the end of `run()` from:

```ts
    const result: OrchestratorResult = { subtasks: results };
    if (truncatedSubtaskCount > 0) result.truncatedSubtaskCount = truncatedSubtaskCount;
    return result;
```

to:

```ts
    const result: OrchestratorResult = { subtasks: results };
    if (truncatedSubtaskCount > 0) result.truncatedSubtaskCount = truncatedSubtaskCount;
    if (opts.synthesize) {
      result.final = await this.synthesize(taskText, results, opts.metaProvider);
    }
    return result;
```

Then add this method to the class (after `routeAndExecute`):

```ts
  private async synthesize(
    taskText: string,
    results: SubtaskResult[],
    metaProvider: string | undefined,
  ): Promise<string> {
    const strategy = this.registry.get(this.getMetaProviderName(metaProvider));
    const summary = results
      .map((r) => {
        const status = r.error
          ? `error: ${r.error}`
          : `unresolved: ${r.unresolved}`;
        return `- ${r.subtask.id} (${r.subtask.description}) [${status}]:\n${r.result || "(no output)"}`;
      })
      .join("\n\n");

    const prompt =
      `Original task: ${taskText}\n\nSubtask results:\n${summary}\n\n` +
      `Write the final combined answer. If any subtask above is marked "unresolved: true" ` +
      `or has an "error", explicitly note that gap instead of presenting a fully confident answer.`;

    const response = await strategy.generate({ prompt });
    return response.text;
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test && npm run typecheck && npm run lint && npm run build`
Expected: PASS across the whole suite.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator.ts src/__tests__/orchestrator.test.ts
git commit -m "feat: add opt-in synthesis step to Orchestrator"
```

---

### Task 9: Public exports

**Files:**
- Modify: `src/index.ts`
- Test: `src/__tests__/index-exports.test.ts`

**Interfaces:**
- Consumes: everything from Tasks 1, 5, 6.
- Produces: the package's full public surface for this feature.

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/index-exports.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  KNOWN_CAPABILITY_TAGS,
  NoAvailableProviderError,
  Orchestrator,
  TaskDecompositionError,
  TaskRouter,
} from "../index";

describe("public exports", () => {
  it("exports the task-router and orchestrator surface", () => {
    expect(typeof TaskRouter).toBe("function");
    expect(typeof Orchestrator).toBe("function");
    expect(typeof TaskDecompositionError).toBe("function");
    expect(typeof NoAvailableProviderError).toBe("function");
    expect(KNOWN_CAPABILITY_TAGS).toEqual([
      "vision",
      "code",
      "long-context",
      "cheap",
      "reasoning",
      "multilingual",
    ]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- index-exports.test.ts`
Expected: FAIL — none of these names are exported from `../index` yet.

- [ ] **Step 3: Implement**

Replace the full contents of `src/index.ts` with:

```ts
export { LlmRegistry } from "./registry";
export {
  InvalidPlatformProviderError,
  LlmKeyValidationError,
  NoAvailableProviderError,
  NoPlatformProviderError,
  TaskDecompositionError,
  UnknownProviderError,
  UnsupportedAttachmentError,
} from "./errors";
export type {
  LlmAttachment,
  LlmGenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResponse,
  LlmStrategy,
  LlmUsage,
} from "./types";
export { KNOWN_CAPABILITY_TAGS, TaskRouter } from "./task-router";
export type {
  CapabilityTag,
  ProviderDescriptor,
  RouteOptions,
  RoutingDecision,
  Subtask,
  TaskRouterOptions,
} from "./task-router";
export { Orchestrator } from "./orchestrator";
export type { OrchestratorResult, RunOptions, SubtaskResult } from "./orchestrator";
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npm test && npm run typecheck && npm run lint && npm run build`
Expected: PASS across the whole suite; `npm run build` succeeds with all five subpath entries plus the main entry.

- [ ] **Step 5: Commit**

```bash
git add src/index.ts src/__tests__/index-exports.test.ts
git commit -m "feat: export TaskRouter and Orchestrator from the main entry"
```

---

### Task 10: README + changeset

**Files:**
- Modify: `README.md`
- Create: `.changeset/task-orchestrator.md`

**Interfaces:**
- Consumes: nothing new — documentation and release metadata only.

- [ ] **Step 1: Update the README**

In `README.md`, change the intro paragraph from:

```
Library-agnostic LLM router. Provider-neutral `LlmStrategy` interface + `LlmRegistry` with env-driven platform selection, BYOK support, and boot-time env-key audit. Opt-in adapters for Gemini, Claude, and Grok via subpath exports — install only the SDKs you actually use.
```

to:

```
Library-agnostic LLM router. Provider-neutral `LlmStrategy` interface + `LlmRegistry` with env-driven platform selection, BYOK support, and boot-time env-key audit. Opt-in adapters for Gemini, Claude, Grok, ChatGPT, and DeepSeek via subpath exports — install only the SDKs you actually use. A `TaskRouter` + `Orchestrator` on top can split a free-text task into subtasks and run each on whichever registered provider fits best.
```

Update the "Install" section's adapter list to add ChatGPT and DeepSeek (both use the `openai` package, so no new `npm install` line is needed — just extend the comment):

```
npm install @google/generative-ai   # for Gemini
npm install @anthropic-ai/sdk       # for Claude
npm install openai                  # for Grok, ChatGPT, and DeepSeek (all OpenAI-compatible)
```

Add a new section after "Adding a custom provider" and before "Error mapping":

````markdown
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
subtask's failure never aborts the run — see
`docs/superpowers/specs/2026-08-25-task-orchestrator-design.md` for the
full design.
````

- [ ] **Step 2: Add a changeset**

Create `.changeset/task-orchestrator.md`:

```markdown
---
"@idevconn/llm-router": minor
---

Add `TaskRouter` and `Orchestrator` for capability-based multi-provider task routing, plus `ChatGptStrategy` and `DeepSeekStrategy` adapters. `LlmStrategy` gains two optional members, `hasPlatformKey?()` and `capabilities?`, which existing custom strategies can ignore without breaking.
```

- [ ] **Step 3: Verify the whole suite one last time**

Run: `npm run typecheck && npm run lint && npm test && npm run build`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add README.md .changeset/task-orchestrator.md
git commit -m "docs: document TaskRouter/Orchestrator and add release changeset"
```

---

## Self-Review

**Spec coverage:** §5 (interface change) → Task 1. §6 (capability vocabulary) → Task 5. §7 (`TaskRouter`, incl. `listAvailableProviders`) → Task 5. §8.1 (decompose) → Task 6. §8.2 (route, per-subtask isolation) → Task 6. §8.3 (critique loop, corrected rule) → Task 6. §8.4 (partial-failure) → Task 6. §8.5 (synthesis) → Task 8. §9 (errors) → Task 1. §10 (chatgpt/deepseek) → Tasks 3-4. Concurrency (`maxConcurrency`, not in the original §7/§8 body text but declared on `RunOptions`) → Task 7.

**Type consistency:** `Subtask`, `RoutingDecision`, `ProviderDescriptor` (Task 5) are imported by name into `orchestrator.ts` (Task 6) without renaming. `SubtaskResult.decision` is optional everywhere it's declared and consumed (Task 6's own type, its `routeAndExecute`, and Task 8's `synthesize` reading `r.error`/`r.unresolved`). `KNOWN_CAPABILITY_TAGS` is defined once in Task 5 and only ever imported (Task 6's decompose prompt, Task 9's re-export) — never redefined.

**No placeholders:** every step above has literal file contents or literal diffs, not descriptions of what to write.

**Mock/scheduling correctness (found by hand-simulating the test suite before finalizing this plan):**
- Added a single-available-provider routing shortcut to `TaskRouter` (spec §7 step 2) — without it, every single-provider test setup would silently make an extra, untested LLM-fallback call, and several Task 5 tests that specifically exercise the fallback path had to move to two-provider setups so the shortcut doesn't preempt them (`falls back... no matching tag`, `routes straight to LLM fallback...`, `providerOverrides win...`, both `NoAvailableProviderError`-from-fallback tests, and the fenced-JSON-stripping test).
- Task 6's cross-critique test originally gave both candidate providers an equal tag-overlap score (an accidental tie, which routes to LLM-fallback instead of the rule stage the test means to exercise) — fixed by requiring two tags so the intended winner scores strictly higher.
- Task 6/7/8 tests with 2+ subtasks cannot rely on `mockResolvedValueOnce` call-order chains once Task 7's concurrency lands (subtasks race under `Promise.all`-style scheduling) — switched those to prompt-content-keyed `mockImplementation`, or pinned `maxConcurrency: 1` where deterministic single-worker ordering was the simpler fix (Task 8's third synthesis test).

---

Plan complete and saved to `docs/superpowers/plans/2026-08-25-task-orchestrator.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
