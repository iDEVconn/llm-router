# Multi-turn `messages[]` Support (All Providers) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native multi-turn conversation history (`messages[]`) support to every provider adapter (Claude, Gemini, ChatGPT, Grok, DeepSeek) in `@idevconn/llm-router`, while keeping every existing single-turn (`prompt`-only) call byte-for-byte unchanged.

**Architecture:** Add a shared `LlmMessage` type and an optional `messages` field to `LlmGenerateOptions` (mutually exclusive with `prompt`, enforced by a single shared validator). Each of the 5 strategies grows an `if (opts.messages) { ... } else { /* existing code, untouched */ }` branch at the point where it currently builds its provider-shaped message/content array. No shared "message builder" abstraction across providers — each SDK's shape is different enough (Anthropic content-blocks vs OpenAI string|array content vs Gemini `Content[]` with `role: "model"`) that a shared builder would just be a leaky abstraction; the shared logic is only the prompt/messages exclusivity check.

**Tech Stack:** TypeScript, vitest, tsup, `@anthropic-ai/sdk`, `openai` (used directly by ChatGPT/Grok/DeepSeek), `@google/generative-ai` + `@google/genai` (Gemini direct/Vertex).

**Spec:** This plan's task list is the spec — see the task-by-task description below; each task states exactly what changes and what stays byte-identical.

## Global Constraints

- Existing single-turn behavior (prompt + attachments, no `messages`) MUST remain byte-for-byte identical — every existing test in `src/__tests__/{claude,gemini,chatgpt,grok,deepseek}.test.ts` must pass with zero edits.
- `messages` and `prompt` are mutually exclusive and one is required: both set → `InvalidGenerateOptionsError`; neither set → `InvalidGenerateOptionsError`.
- `attachments` apply only to the **last** turn when `messages` is used (matches how attachments+prompt combine today for single-turn).
- Gemini role mapping: `"user"` → `"user"`, `"assistant"` → `"model"` (Gemini's own vocabulary, not OpenAI's/Anthropic's `"assistant"`).
- `systemPrompt` handling is unchanged in both single-turn and multi-turn modes for every provider.
- `UnsupportedMultiTurnError` is added to `errors.ts` for future providers that can't implement multi-turn, but none of the 5 current adapters throw it — all 5 implement `messages` natively.
- No new runtime dependencies. No changes to `TaskRouter`/`Orchestrator`/`Retriever`/injection-defense.

---

## File Structure

- Modify `src/types.ts` — add `LlmMessage`, make `prompt` optional, add `messages`.
- Modify `src/errors.ts` — add `InvalidGenerateOptionsError`, `UnsupportedMultiTurnError`.
- Create `src/validate-generate-options.ts` — shared `assertExactlyOnePromptSource(opts)` helper, used by all 5 `generate()` methods.
- Create `src/__tests__/validate-generate-options.test.ts` — unit tests for the helper.
- Modify `src/claude/index.ts` + `src/__tests__/claude.test.ts`.
- Modify `src/chatgpt/index.ts` + `src/__tests__/chatgpt.test.ts`.
- Modify `src/grok/index.ts` + `src/__tests__/grok.test.ts`.
- Modify `src/deepseek/index.ts` + `src/__tests__/deepseek.test.ts`.
- Modify `src/gemini/index.ts` + `src/__tests__/gemini.test.ts`.
- Modify `README.md`.
- Add a changeset via `npx changeset`.

---

### Task 1: `LlmMessage` type + `LlmGenerateOptions.messages`

**Files:**
- Modify: `src/types.ts:11-71` (the `LlmGenerateOptions` interface and the block right above it)
- Test: none (pure type addition — covered indirectly by every strategy test below; TS compiler is the check)

**Interfaces:**
- Produces: `export interface LlmMessage { role: "user" | "assistant"; content: string }`, and `LlmGenerateOptions.messages?: LlmMessage[]`, `LlmGenerateOptions.prompt?: string` (now optional).

- [ ] **Step 1: Edit `src/types.ts`**

Replace:

```ts
/** Per-call options. */
export interface LlmGenerateOptions {
  prompt: string;
  /**
   * Stable instructions shared across many calls (e.g. a system/role
   * prompt). Kept separate from `prompt` so strategies that support
   * prompt caching (currently Claude, via `cache_control`) can cache it
   * independently of the per-call dynamic content, cutting the cost of
   * repeated calls that reuse the same instructions. Strategies without
   * caching support still use it correctly (as a system-role message or
   * `systemInstruction`) — they just don't get the cost benefit.
   */
  systemPrompt?: string;
```

with:

```ts
/** One turn of multi-turn conversation history. See `LlmGenerateOptions.messages`. */
export interface LlmMessage {
  role: "user" | "assistant";
  content: string;
}

/** Per-call options. */
export interface LlmGenerateOptions {
  /**
   * Single-turn prompt text. Mutually exclusive with `messages` — for
   * multi-turn conversation history, use `messages` instead. Exactly one
   * of `prompt` / `messages` must be set; both or neither throws
   * `InvalidGenerateOptionsError`.
   */
  prompt?: string;
  /**
   * Full multi-turn conversation history. When provided, this replaces
   * `prompt` as the content sent to the provider — exactly one of
   * `prompt` / `messages` must be set, never both, never neither
   * (`InvalidGenerateOptionsError` otherwise). Roles must alternate
   * starting with "user" (provider-enforced, not validated here).
   * `attachments` apply only to the final turn.
   */
  messages?: LlmMessage[];
  /**
   * Stable instructions shared across many calls (e.g. a system/role
   * prompt). Kept separate from `prompt`/`messages` so strategies that
   * support prompt caching (currently Claude, via `cache_control`) can
   * cache it independently of the per-call dynamic content, cutting the
   * cost of repeated calls that reuse the same instructions. Strategies
   * without caching support still use it correctly (as a system-role
   * message or `systemInstruction`) — they just don't get the cost
   * benefit.
   */
  systemPrompt?: string;
```

- [ ] **Step 2: Typecheck**

Run: `npm run typecheck`
Expected: FAILS — every strategy's `opts.prompt` usage (e.g. `content.push({ type: "text", text: opts.prompt })`) now sees `string | undefined` where `string` is required. This is expected; Tasks 3–8 fix each call site as part of adding the `messages` branch (they narrow `opts.prompt` inside the `else` branch, where the shared validator has already guaranteed it's defined).

- [ ] **Step 3: Commit**

```bash
git add src/types.ts
git commit -m "feat(types): add LlmMessage and optional messages field to LlmGenerateOptions"
```

---

### Task 2: `InvalidGenerateOptionsError` + `UnsupportedMultiTurnError`

**Files:**
- Modify: `src/errors.ts` (doc-comment table at top, plus new classes appended at the end)
- Test: `src/__tests__/errors.test.ts` (append)

**Interfaces:**
- Produces: `new InvalidGenerateOptionsError(reason: string)`, `new UnsupportedMultiTurnError(providerName: string)`.

- [ ] **Step 1: Write the failing tests**

Append to `src/__tests__/errors.test.ts`:

```ts
import { InvalidGenerateOptionsError, UnsupportedMultiTurnError } from "../errors";

describe("InvalidGenerateOptionsError", () => {
  it("names the problem when both prompt and messages are set", () => {
    const err = new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but both were provided.",
    );
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("InvalidGenerateOptionsError");
    expect(err.message).toMatch(/both were provided/);
  });

  it("names the problem when neither prompt nor messages are set", () => {
    const err = new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but neither was provided.",
    );
    expect(err.message).toMatch(/neither was provided/);
  });
});

describe("UnsupportedMultiTurnError", () => {
  it("names the offending provider", () => {
    const err = new UnsupportedMultiTurnError("acme-llm");
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("UnsupportedMultiTurnError");
    expect(err.providerName).toBe("acme-llm");
    expect(err.message).toMatch(/acme-llm/);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/errors.test.ts`
Expected: FAIL with `"../errors" has no exported member 'InvalidGenerateOptionsError'` (or similar TS/import error).

- [ ] **Step 3: Implement in `src/errors.ts`**

Update the doc-comment table at the top of the file — replace:

```
 *   - `UnsupportedThinkingModeError` → 400
 *   - `InvalidThinkingConfigError`   → 400
 */
```

with:

```
 *   - `UnsupportedThinkingModeError` → 400
 *   - `InvalidThinkingConfigError`   → 400
 *   - `InvalidGenerateOptionsError`  → 400
 *   - `UnsupportedMultiTurnError`    → 400
 */
```

Append at the end of the file:

```ts
export class InvalidGenerateOptionsError extends Error {
  constructor(reason: string) {
    super(`Invalid LlmGenerateOptions: ${reason}`);
    this.name = "InvalidGenerateOptionsError";
  }
}

export class UnsupportedMultiTurnError extends Error {
  constructor(public readonly providerName: string) {
    super(
      `${providerName} does not support multi-turn \`messages\`. Pass a single-turn \`prompt\` instead, or switch to a provider that supports multi-turn.`,
    );
    this.name = "UnsupportedMultiTurnError";
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/errors.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/errors.ts src/__tests__/errors.test.ts
git commit -m "feat(errors): add InvalidGenerateOptionsError and UnsupportedMultiTurnError"
```

---

### Task 3: Shared `assertExactlyOnePromptSource` validator

**Files:**
- Create: `src/validate-generate-options.ts`
- Create: `src/__tests__/validate-generate-options.test.ts`

**Interfaces:**
- Consumes: `LlmGenerateOptions` (from Task 1), `InvalidGenerateOptionsError` (from Task 2).
- Produces: `export function assertExactlyOnePromptSource(opts: LlmGenerateOptions): void` — throws `InvalidGenerateOptionsError` on violation, otherwise returns `void`. Called at the very top of every strategy's `generate()`, before any network/SDK-client work.

- [ ] **Step 1: Write the failing test**

Create `src/__tests__/validate-generate-options.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { InvalidGenerateOptionsError } from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";

describe("assertExactlyOnePromptSource", () => {
  it("passes when only prompt is set", () => {
    expect(() => assertExactlyOnePromptSource({ prompt: "hi" })).not.toThrow();
  });

  it("passes when only messages is set", () => {
    expect(() =>
      assertExactlyOnePromptSource({ messages: [{ role: "user", content: "hi" }] }),
    ).not.toThrow();
  });

  it("throws InvalidGenerateOptionsError when both prompt and messages are set", () => {
    expect(() =>
      assertExactlyOnePromptSource({
        prompt: "hi",
        messages: [{ role: "user", content: "hi" }],
      }),
    ).toThrow(InvalidGenerateOptionsError);
  });

  it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", () => {
    expect(() => assertExactlyOnePromptSource({})).toThrow(InvalidGenerateOptionsError);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/__tests__/validate-generate-options.test.ts`
Expected: FAIL — `Cannot find module '../validate-generate-options'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/validate-generate-options.ts`:

```ts
import { InvalidGenerateOptionsError } from "./errors";
import type { LlmGenerateOptions } from "./types";

/**
 * Enforces that a `generate()` call supplies exactly one content source.
 * Called first in every strategy's `generate()`, before any SDK client
 * or network call, so a malformed call fails fast and cheaply.
 */
export function assertExactlyOnePromptSource(
  opts: Pick<LlmGenerateOptions, "prompt" | "messages">,
): void {
  const hasPrompt = opts.prompt !== undefined;
  const hasMessages = opts.messages !== undefined;

  if (hasPrompt && hasMessages) {
    throw new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but both were provided.",
    );
  }
  if (!hasPrompt && !hasMessages) {
    throw new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but neither was provided.",
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run src/__tests__/validate-generate-options.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/validate-generate-options.ts src/__tests__/validate-generate-options.test.ts
git commit -m "feat: add shared prompt/messages exclusivity validator"
```

---

### Task 4: Claude multi-turn

**Files:**
- Modify: `src/claude/index.ts:68-147` (`generate()` method)
- Test: `src/__tests__/claude.test.ts` (append a `describe("messages")` block)

**Interfaces:**
- Consumes: `assertExactlyOnePromptSource` (Task 3), `LlmMessage` (Task 1).
- Produces: no new exports; internal behavior only.

- [ ] **Step 1: Write the failing tests**

Append to `src/__tests__/claude.test.ts` (inside the top-level `describe("ClaudeStrategy", ...)`, e.g. right after the `describe("signal", ...)` block, before the final closing `});`):

```ts
  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [{ type: "text", text: "ok" }],
        model: "claude-haiku-4-5",
        usage: { input_tokens: 1, output_tokens: 1 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockMessagesCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "user", content: [{ type: "text", text: "first" }] },
        { role: "assistant", content: [{ type: "text", text: "second" }] },
        { role: "user", content: [{ type: "text", text: "third" }] },
      ]);
    });

    it("puts attachments on the last turn's content, not the first", async () => {
      mockMessagesCreate.mockResolvedValueOnce({
        content: [],
        model: "claude-haiku-4-5",
        usage: { input_tokens: 0, output_tokens: 0 },
      });
      const strategy = new ClaudeStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      });

      const call = mockMessagesCreate.mock.calls[0]![0];
      expect(call.messages[0].content).toEqual([{ type: "text", text: "first" }]);
      expect(call.messages[1].content).toEqual([{ type: "text", text: "second" }]);
      expect(call.messages[2].content).toEqual([
        { type: "image", source: { type: "base64", media_type: "image/png", data: "aW1n" } },
        { type: "text", text: "third" },
      ]);
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new ClaudeStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockMessagesCreate).not.toHaveBeenCalled();
    });
  });
```

Add `InvalidGenerateOptionsError` to the existing `import { ... } from "../errors";` at the top of the file.

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/claude.test.ts`
Expected: FAIL — the multi-turn tests fail because `generate()` doesn't branch on `opts.messages` yet (it still does `content.push({ type: "text", text: opts.prompt })`, so with no `prompt` it sends `text: undefined`); the two `InvalidGenerateOptionsError` tests fail because nothing throws yet.

- [ ] **Step 3: Implement in `src/claude/index.ts`**

Add to the top-level import:

```ts
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
```

→

```ts
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";
import type { LlmMessage } from "../types";
```

Replace the body of `generate()` from its start through the `content.push({ type: "text", text: opts.prompt });` line and the `params` construction, i.e. replace:

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    const client = opts.apiKey
      ? new Anthropic({ apiKey: opts.apiKey })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;
    const maxTokens = opts.maxTokens ?? DEFAULT_MAX_TOKENS;
    const thinking = this.resolveThinking(opts.thinking, maxTokens);

    type ContentBlock =
      | { type: "text"; text: string }
      | {
          type: "image";
          source: { type: "base64"; media_type: string; data: string };
        }
      | {
          type: "document";
          source: { type: "base64"; media_type: "application/pdf"; data: string };
        };

    const content: ContentBlock[] = [];

    for (const attachment of opts.attachments ?? []) {
      const data = toBase64(attachment.data);
      if (SUPPORTED_IMAGE_TYPES.has(attachment.mimetype)) {
        content.push({
          type: "image",
          source: { type: "base64", media_type: attachment.mimetype, data },
        });
      } else {
        content.push({
          type: "document",
          source: { type: "base64", media_type: "application/pdf", data },
        });
      }
    }

    content.push({ type: "text", text: opts.prompt });

    const params = {
      model: modelName,
      max_tokens: maxTokens,
      // `cache_control: ephemeral` on the system block lets Anthropic cache
      // it server-side, so repeated calls that reuse the same systemPrompt
      // (the common case — a report-generation instruction set called many
      // times) are billed at the much cheaper cache-read rate instead of
      // full input-token price on every call.
      ...(opts.systemPrompt
        ? {
            system: [
              {
                type: "text" as const,
                text: opts.systemPrompt,
                cache_control: { type: "ephemeral" as const },
              },
            ],
          }
        : {}),
      ...(thinking ? { thinking } : {}),
      // Anthropic's SDK types accept the broader union; cast here so the
      // pkg compiles without pulling in the entire Anthropic.Messages
      // type surface as a public dep.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: [{ role: "user" as const, content: content as any }],
    };
    const requestOptions = opts.signal ? { signal: opts.signal } : undefined;
```

with:

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    assertExactlyOnePromptSource(opts);

    const client = opts.apiKey
      ? new Anthropic({ apiKey: opts.apiKey })
      : this.getPlatformClient();
    const modelName = opts.model?.trim() || this.defaultModel;
    const maxTokens = opts.maxTokens ?? DEFAULT_MAX_TOKENS;
    const thinking = this.resolveThinking(opts.thinking, maxTokens);

    type ContentBlock =
      | { type: "text"; text: string }
      | {
          type: "image";
          source: { type: "base64"; media_type: string; data: string };
        }
      | {
          type: "document";
          source: { type: "base64"; media_type: "application/pdf"; data: string };
        };

    const attachmentBlocks: ContentBlock[] = [];
    for (const attachment of opts.attachments ?? []) {
      const data = toBase64(attachment.data);
      if (SUPPORTED_IMAGE_TYPES.has(attachment.mimetype)) {
        attachmentBlocks.push({
          type: "image",
          source: { type: "base64", media_type: attachment.mimetype, data },
        });
      } else {
        attachmentBlocks.push({
          type: "document",
          source: { type: "base64", media_type: "application/pdf", data },
        });
      }
    }

    type AnthropicMessage = { role: "user" | "assistant"; content: ContentBlock[] };
    let anthropicMessages: AnthropicMessage[];

    if (opts.messages) {
      anthropicMessages = (opts.messages as LlmMessage[]).map((message) => ({
        role: message.role,
        content: [{ type: "text" as const, text: message.content }],
      }));
      const lastMessage = anthropicMessages[anthropicMessages.length - 1];
      if (lastMessage) lastMessage.content.unshift(...attachmentBlocks);
    } else {
      anthropicMessages = [
        { role: "user", content: [...attachmentBlocks, { type: "text", text: opts.prompt! }] },
      ];
    }

    const params = {
      model: modelName,
      max_tokens: maxTokens,
      // `cache_control: ephemeral` on the system block lets Anthropic cache
      // it server-side, so repeated calls that reuse the same systemPrompt
      // (the common case — a report-generation instruction set called many
      // times) are billed at the much cheaper cache-read rate instead of
      // full input-token price on every call.
      ...(opts.systemPrompt
        ? {
            system: [
              {
                type: "text" as const,
                text: opts.systemPrompt,
                cache_control: { type: "ephemeral" as const },
              },
            ],
          }
        : {}),
      ...(thinking ? { thinking } : {}),
      // Anthropic's SDK types accept the broader union; cast here so the
      // pkg compiles without pulling in the entire Anthropic.Messages
      // type surface as a public dep.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      messages: anthropicMessages as any,
    };
    const requestOptions = opts.signal ? { signal: opts.signal } : undefined;
```

Note the attachment ordering: the existing single-turn behavior puts attachment blocks *before* the text block (see the original code — attachments are pushed first, then `content.push({type:"text",...})` last). The rewrite above preserves that for both branches: single-turn spreads `[...attachmentBlocks, textBlock]`, and multi-turn unshifts attachment blocks onto the front of the last message's content array (which starts as `[textBlock]`), landing at the same `[attachments..., text]` order the existing tests assert (`content[0].type === "image"`, `content[last].type === "text"`).

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/claude.test.ts`
Expected: PASS — all pre-existing tests plus the 4 new ones.

- [ ] **Step 5: Run the full existing suite for a regression check**

Run: `npx vitest run src/__tests__/claude.test.ts`
Expected: every test in the file passes, including the pre-existing image/pdf/thinking/streaming/signal tests, unmodified.

- [ ] **Step 6: Commit**

```bash
git add src/claude/index.ts src/__tests__/claude.test.ts
git commit -m "feat(claude): support multi-turn messages[]"
```

---

### Task 5: ChatGPT multi-turn

**Files:**
- Modify: `src/chatgpt/index.ts:89-127` (the message-building section of `generate()`)
- Test: `src/__tests__/chatgpt.test.ts` (append a `describe("messages (multi-turn)")` block)

**Interfaces:**
- Consumes: `assertExactlyOnePromptSource` (Task 3), `LlmMessage` (Task 1).

- [ ] **Step 1: Write the failing tests**

Append to `src/__tests__/chatgpt.test.ts` (inside `describe("ChatGptStrategy", ...)`, before the final closing `});`):

```ts
  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "user", content: "first" },
        { role: "assistant", content: "second" },
        { role: "user", content: "third" },
      ]);
    });

    it("prepends the systemPrompt as a leading system message in multi-turn mode too", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [{ role: "user", content: "first" }],
        systemPrompt: "Be concise.",
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages[0]).toEqual({ role: "system", content: "Be concise." });
      expect(call.messages[1]).toEqual({ role: "user", content: "first" });
    });

    it("puts attachments on the last turn's content array, not the first", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "gpt-4.1-mini",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new ChatGptStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/jpeg" }],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages[0]).toEqual({ role: "user", content: "first" });
      expect(call.messages[1]).toEqual({ role: "assistant", content: "second" });
      expect(call.messages[2].role).toBe("user");
      expect(call.messages[2].content[0].type).toBe("image_url");
      expect(call.messages[2].content[0].image_url.url).toMatch(/^data:image\/jpeg;base64,/);
      expect(call.messages[2].content[1]).toEqual({ type: "text", text: "third" });
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new ChatGptStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });
  });
```

Add `InvalidGenerateOptionsError` to the existing error import at the top of the file.

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/chatgpt.test.ts`
Expected: FAIL — no `messages` branch exists yet, and nothing throws `InvalidGenerateOptionsError`.

- [ ] **Step 3: Implement in `src/chatgpt/index.ts`**

Update the import block:

```ts
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";
```

→

```ts
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";
import type { LlmGenerateOptions, LlmMessage, LlmResponse, LlmStrategy } from "../types";
```

(`InvalidGenerateOptionsError` is imported for re-export symmetry with the test file's import list — it's not referenced directly in this file, so TS/eslint's unused-import check would flag it; instead just import `assertExactlyOnePromptSource`, which throws it internally. Drop `InvalidGenerateOptionsError` from this file's own import — only add it in the test file.)

Add `assertExactlyOnePromptSource(opts);` as the very first line of `generate()`:

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    assertExactlyOnePromptSource(opts);

    for (const attachment of opts.attachments ?? []) {
```

Replace the message-building block:

```ts
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
```

with:

```ts
    type ChatContentPart =
      | { type: "text"; text: string }
      | { type: "image_url"; image_url: { url: string; detail: "high" | "low" | "auto" } };

    const attachmentParts: ChatContentPart[] = (opts.attachments ?? []).map((attachment) => {
      const data = toBase64(attachment.data);
      return {
        type: "image_url",
        image_url: { url: `data:${attachment.mimetype};base64,${data}`, detail: "high" },
      };
    });

    type ChatMessage = { role: "system" | "user" | "assistant"; content: string | ChatContentPart[] };
    const turns: ChatMessage[] = [];

    if (opts.messages) {
      const history = opts.messages as LlmMessage[];
      history.forEach((message, index) => {
        const isLast = index === history.length - 1;
        if (isLast && attachmentParts.length > 0) {
          turns.push({
            role: message.role,
            content: [...attachmentParts, { type: "text", text: message.content }],
          });
        } else {
          turns.push({ role: message.role, content: message.content });
        }
      });
    } else {
      turns.push({
        role: "user",
        content: [...attachmentParts, { type: "text", text: opts.prompt! }],
      });
    }

    const messages = opts.systemPrompt
      ? [{ role: "system" as const, content: opts.systemPrompt }, ...turns]
      : turns;
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/chatgpt.test.ts`
Expected: PASS — all pre-existing tests (image attachment, PDF rejection, systemPrompt, streaming, thinking, signal) plus the new multi-turn tests.

- [ ] **Step 5: Commit**

```bash
git add src/chatgpt/index.ts src/__tests__/chatgpt.test.ts
git commit -m "feat(chatgpt): support multi-turn messages[]"
```

---

### Task 6: Grok multi-turn

**Files:**
- Modify: `src/grok/index.ts:67-125` (the message-building section of `generate()`)
- Test: `src/__tests__/grok.test.ts` (append a `describe("messages (multi-turn)")` block)

**Interfaces:**
- Consumes: `assertExactlyOnePromptSource` (Task 3), `LlmMessage` (Task 1).

Grok's adapter builds messages the same shape as ChatGPT's (same `openai` SDK, same OpenAI wire format) — apply the identical pattern from Task 5, adjusted to this file's structure (no `reasoningEffort`, has its own `generateStreaming`/`shapeResponse` helpers that this task does not need to touch since they only consume the already-built `messages`/`body`).

- [ ] **Step 1: Write the failing tests**

Append to `src/__tests__/grok.test.ts` (inside `describe("GrokStrategy", ...)`, before the final closing `});`):

```ts
  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "grok-4.3",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new GrokStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "user", content: "first" },
        { role: "assistant", content: "second" },
        { role: "user", content: "third" },
      ]);
    });

    it("prepends the systemPrompt as a leading system message in multi-turn mode too", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "grok-4.3",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new GrokStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [{ role: "user", content: "first" }],
        systemPrompt: "Be concise.",
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages[0]).toEqual({ role: "system", content: "Be concise." });
      expect(call.messages[1]).toEqual({ role: "user", content: "first" });
    });

    it("puts attachments on the last turn's content array, not the first", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "grok-4.3",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new GrokStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/jpeg" }],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages[0]).toEqual({ role: "user", content: "first" });
      expect(call.messages[1]).toEqual({ role: "assistant", content: "second" });
      expect(call.messages[2].role).toBe("user");
      expect(call.messages[2].content[0].type).toBe("image_url");
      expect(call.messages[2].content[1]).toEqual({ type: "text", text: "third" });
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new GrokStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new GrokStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });
  });
```

Add `InvalidGenerateOptionsError` to the existing error import at the top of the test file.

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/grok.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement in `src/grok/index.ts`**

Update the import block:

```ts
import {
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";
```

→

```ts
import {
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";
import type { LlmGenerateOptions, LlmMessage, LlmResponse, LlmStrategy } from "../types";
```

Add `assertExactlyOnePromptSource(opts);` as the first line of `generate()`:

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    assertExactlyOnePromptSource(opts);

    for (const attachment of opts.attachments ?? []) {
```

Replace:

```ts
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

    // A leading system-role message is the correct OpenAI-wire-format shape
    // for stable instructions. It also positions the request to benefit
    // from any automatic prefix-based caching xAI's backend may apply
    // (OpenAI-compatible APIs commonly cache repeated prompt prefixes
    // transparently) — no explicit cache API is documented for xAI, so
    // this is a structural best-effort, not a guaranteed cost saving.
    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: messageContent },
        ]
      : [{ role: "user" as const, content: messageContent }];
```

with:

```ts
    type ChatContentPart =
      | { type: "text"; text: string }
      | { type: "image_url"; image_url: { url: string; detail: "high" | "low" | "auto" } };

    const attachmentParts: ChatContentPart[] = (opts.attachments ?? []).map((attachment) => {
      const data = toBase64(attachment.data);
      return {
        type: "image_url",
        image_url: { url: `data:${attachment.mimetype};base64,${data}`, detail: "high" },
      };
    });

    type ChatMessage = { role: "system" | "user" | "assistant"; content: string | ChatContentPart[] };
    const turns: ChatMessage[] = [];

    if (opts.messages) {
      const history = opts.messages as LlmMessage[];
      history.forEach((message, index) => {
        const isLast = index === history.length - 1;
        if (isLast && attachmentParts.length > 0) {
          turns.push({
            role: message.role,
            content: [...attachmentParts, { type: "text", text: message.content }],
          });
        } else {
          turns.push({ role: message.role, content: message.content });
        }
      });
    } else {
      turns.push({
        role: "user",
        content: [...attachmentParts, { type: "text", text: opts.prompt! }],
      });
    }

    // A leading system-role message is the correct OpenAI-wire-format shape
    // for stable instructions. It also positions the request to benefit
    // from any automatic prefix-based caching xAI's backend may apply
    // (OpenAI-compatible APIs commonly cache repeated prompt prefixes
    // transparently) — no explicit cache API is documented for xAI, so
    // this is a structural best-effort, not a guaranteed cost saving.
    const messages = opts.systemPrompt
      ? [{ role: "system" as const, content: opts.systemPrompt }, ...turns]
      : turns;
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/grok.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/grok/index.ts src/__tests__/grok.test.ts
git commit -m "feat(grok): support multi-turn messages[]"
```

---

### Task 7: DeepSeek multi-turn

**Files:**
- Modify: `src/deepseek/index.ts:89-121` (the message-building section of `generate()`)
- Test: `src/__tests__/deepseek.test.ts` (append a `describe("messages (multi-turn)")` block)

**Interfaces:**
- Consumes: `assertExactlyOnePromptSource` (Task 3), `LlmMessage` (Task 1). DeepSeek has no attachment support at all (unconditionally throws `UnsupportedAttachmentError` on any attachment) — that check runs before the prompt/messages branch and is unaffected by this task.

- [ ] **Step 1: Write the failing tests**

Append to `src/__tests__/deepseek.test.ts` (inside `describe("DeepSeekStrategy", ...)`, before the final closing `});`):

```ts
  describe("messages (multi-turn)", () => {
    it("builds a multi-turn messages array from 2+ turns", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "deepseek-chat",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new DeepSeekStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "user", content: "first" },
        { role: "assistant", content: "second" },
        { role: "user", content: "third" },
      ]);
    });

    it("prepends the systemPrompt as a leading system message in multi-turn mode too", async () => {
      mockChatCompletionsCreate.mockResolvedValueOnce({
        choices: [{ message: { content: "ok" } }],
        model: "deepseek-chat",
        usage: { prompt_tokens: 1, completion_tokens: 1 },
      });
      const strategy = new DeepSeekStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [{ role: "user", content: "first" }],
        systemPrompt: "Be concise.",
      });

      const call = mockChatCompletionsCreate.mock.calls[0]![0];
      expect(call.messages).toEqual([
        { role: "system", content: "Be concise." },
        { role: "user", content: "first" },
      ]);
    });

    it("still rejects attachments in multi-turn mode (no vision endpoint)", async () => {
      const strategy = new DeepSeekStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({
          messages: [{ role: "user", content: "first" }],
          attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
        }),
      ).rejects.toBeInstanceOf(UnsupportedAttachmentError);
    });

    it("throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new DeepSeekStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });

    it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new DeepSeekStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockChatCompletionsCreate).not.toHaveBeenCalled();
    });
  });
```

Add `InvalidGenerateOptionsError` to the existing error import at the top of the test file.

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/deepseek.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement in `src/deepseek/index.ts`**

Update the import block:

```ts
import {
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";
```

→

```ts
import {
  LlmKeyValidationError,
  UnsupportedAttachmentError,
  UnsupportedThinkingModeError,
} from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";
import type { LlmGenerateOptions, LlmMessage, LlmResponse, LlmStrategy } from "../types";
```

Add `assertExactlyOnePromptSource(opts);` as the first line of `generate()`, before the attachments check (attachments must still be rejected before anything else, matching the new test above; `assertExactlyOnePromptSource` is cheap validation so ordering it first vs. after the attachment check doesn't change observable behavior for any existing test — put it first for consistency with the other 4 adapters):

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    assertExactlyOnePromptSource(opts);

    const attachments = opts.attachments ?? [];
```

Replace:

```ts
    const messages = opts.systemPrompt
      ? [
          { role: "system" as const, content: opts.systemPrompt },
          { role: "user" as const, content: opts.prompt },
        ]
      : [{ role: "user" as const, content: opts.prompt }];
```

with:

```ts
    type DeepSeekMessage = { role: "system" | "user" | "assistant"; content: string };
    const turns: DeepSeekMessage[] = opts.messages
      ? (opts.messages as LlmMessage[]).map((message) => ({
          role: message.role,
          content: message.content,
        }))
      : [{ role: "user", content: opts.prompt! }];

    const messages: DeepSeekMessage[] = opts.systemPrompt
      ? [{ role: "system", content: opts.systemPrompt }, ...turns]
      : turns;
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/deepseek.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/deepseek/index.ts src/__tests__/deepseek.test.ts
git commit -m "feat(deepseek): support multi-turn messages[]"
```

---

### Task 8: Gemini multi-turn (direct API + Vertex)

**Files:**
- Modify: `src/gemini/index.ts` (`buildParts`, `generateViaDirectApi`, `generateViaVertex`)
- Test: `src/__tests__/gemini.test.ts` (append a `describe("messages (multi-turn)")` block covering both connection modes)

**Interfaces:**
- Consumes: `assertExactlyOnePromptSource` (Task 3), `LlmMessage` (Task 1).
- Produces: `toGeminiRole(role: "user" | "assistant"): "user" | "model"` (internal, not exported).

Gemini's SDK type for a full request is `GenerateContentRequest` with field `contents: Content[]`, `Content = { role: string; parts: Part[] }` — confirmed against `node_modules/@google/generative-ai/dist/generative-ai.d.ts`. `generateContent`/`generateContentStream` accept either that request object or a bare `Part[]`/string shorthand; the existing single-turn code uses the bare-array shorthand and this task keeps that untouched, only adding a new `{contents}`-object call for the `messages` branch.

- [ ] **Step 1: Write the failing tests**

Append to `src/__tests__/gemini.test.ts` (inside `describe("GeminiStrategy", ...)`, before the final closing `});`):

```ts
  describe("messages (multi-turn)", () => {
    it("direct API: builds a contents array with assistant mapped to model role", async () => {
      mockGenerateContent.mockResolvedValueOnce({
        response: { text: () => "ok" },
      });
      const strategy = new GeminiStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockGenerateContent.mock.calls[0]![0];
      expect(call.contents).toEqual([
        { role: "user", parts: [{ text: "first" }] },
        { role: "model", parts: [{ text: "second" }] },
        { role: "user", parts: [{ text: "third" }] },
      ]);
    });

    it("direct API: puts attachments on the last turn's parts, not the first", async () => {
      mockGenerateContent.mockResolvedValueOnce({ response: { text: () => "ok" } });
      const strategy = new GeminiStrategy({ apiKey: "k" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      });

      const call = mockGenerateContent.mock.calls[0]![0];
      expect(call.contents[0]).toEqual({ role: "user", parts: [{ text: "first" }] });
      expect(call.contents[1].parts[0]).toEqual({ text: "third" });
      expect(call.contents[1].parts[1].inlineData.mimeType).toBe("image/png");
    });

    it("direct API: throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });

    it("direct API: throws InvalidGenerateOptionsError when neither prompt nor messages are set", async () => {
      const strategy = new GeminiStrategy({ apiKey: "k" });
      await expect(strategy.generate({})).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockGenerateContent).not.toHaveBeenCalled();
    });

    it("Vertex: builds a contents array with assistant mapped to model role", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "assistant", content: "second" },
          { role: "user", content: "third" },
        ],
      });

      const call = mockVertexGenerateContent.mock.calls[0]![0];
      expect(call.contents).toEqual([
        { role: "user", parts: [{ text: "first" }] },
        { role: "model", parts: [{ text: "second" }] },
        { role: "user", parts: [{ text: "third" }] },
      ]);
    });

    it("Vertex: puts attachments on the last turn's parts, not the first", async () => {
      mockVertexGenerateContent.mockResolvedValueOnce({ text: "ok" });
      const strategy = new GeminiStrategy({ connection: "vertex" });

      await strategy.generate({
        messages: [
          { role: "user", content: "first" },
          { role: "user", content: "third" },
        ],
        attachments: [{ data: Buffer.from("img"), mimetype: "image/png" }],
      });

      const call = mockVertexGenerateContent.mock.calls[0]![0];
      expect(call.contents[0]).toEqual({ role: "user", parts: [{ text: "first" }] });
      expect(call.contents[1].parts[0]).toEqual({ text: "third" });
      expect(call.contents[1].parts[1].inlineData.mimeType).toBe("image/png");
    });

    it("Vertex: throws InvalidGenerateOptionsError when both prompt and messages are set", async () => {
      const strategy = new GeminiStrategy({ connection: "vertex" });
      await expect(
        strategy.generate({ prompt: "p", messages: [{ role: "user", content: "m" }] }),
      ).rejects.toBeInstanceOf(InvalidGenerateOptionsError);
      expect(mockVertexGenerateContent).not.toHaveBeenCalled();
    });
  });
```

Add `InvalidGenerateOptionsError` to the existing error import at the top of the test file.

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/__tests__/gemini.test.ts`
Expected: FAIL.

- [ ] **Step 3: Implement in `src/gemini/index.ts`**

Update the import block:

```ts
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
import type {
  LlmAttachment,
  LlmGenerateOptions,
  LlmResponse,
  LlmStrategy,
} from "../types";
```

→

```ts
import {
  InvalidThinkingConfigError,
  LlmKeyValidationError,
  UnsupportedThinkingModeError,
} from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";
import type {
  LlmAttachment,
  LlmGenerateOptions,
  LlmMessage,
  LlmResponse,
  LlmStrategy,
} from "../types";
```

Add a role mapper and a shared `Content[]` builder right after the existing `buildParts` function (keep `buildParts` itself completely unchanged — the single-turn path keeps calling it):

```ts
type GeminiContent = { role: string; parts: GeminiPart[] };

function toGeminiRole(role: LlmMessage["role"]): "user" | "model" {
  return role === "assistant" ? "model" : "user";
}

function buildContents(opts: LlmGenerateOptions): GeminiContent[] {
  const history = opts.messages as LlmMessage[];
  const contents: GeminiContent[] = history.map((message) => ({
    role: toGeminiRole(message.role),
    parts: [{ text: message.content }],
  }));
  const last = contents[contents.length - 1];
  if (last) {
    for (const attachment of opts.attachments ?? []) {
      last.parts.push({
        inlineData: { mimeType: attachment.mimetype, data: toBase64(attachment.data) },
      });
    }
  }
  return contents;
}
```

In `generate()`, add the validator as the first check (before the existing `opts.signal?.aborted` check, so a malformed call fails before any abort-signal handling too):

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    if (opts.signal?.aborted) {
      throw opts.signal.reason ?? new Error("gemini request aborted before it started");
    }
```

→

```ts
  async generate(opts: LlmGenerateOptions): Promise<LlmResponse> {
    assertExactlyOnePromptSource(opts);

    if (opts.signal?.aborted) {
      throw opts.signal.reason ?? new Error("gemini request aborted before it started");
    }
```

In `generateViaDirectApi`, replace:

```ts
    const parts = buildParts(opts);
    const requestOptions = opts.signal ? { signal: opts.signal } : undefined;

    if (opts.onToken) {
      const result = await model.generateContentStream(parts, requestOptions);
      for await (const chunk of result.stream) {
        forwardDelta(opts.onToken, chunk.text());
      }
      const finalResponse = await result.response;
      return buildDirectResponse(finalResponse, modelName);
    }

    const result = await model.generateContent(parts, requestOptions);
```

with:

```ts
    const requestOptions = opts.signal ? { signal: opts.signal } : undefined;
    const requestPayload: GeminiPart[] | { contents: GeminiContent[] } = opts.messages
      ? { contents: buildContents(opts) }
      : buildParts(opts);

    if (opts.onToken) {
      const result = await model.generateContentStream(requestPayload, requestOptions);
      for await (const chunk of result.stream) {
        forwardDelta(opts.onToken, chunk.text());
      }
      const finalResponse = await result.response;
      return buildDirectResponse(finalResponse, modelName);
    }

    const result = await model.generateContent(requestPayload, requestOptions);
```

(The `GeminiPart[] | {contents: GeminiContent[]}` union matches what `generateContent`/`generateContentStream` already accept per the SDK's `GenerateContentRequest | string | Array<string|Part>` signature — passing a bare `Part[]` for single-turn is unchanged, passing `{contents}` for multi-turn is the new branch.)

In `generateViaVertex`, replace:

```ts
    const parts = buildParts(opts);
    const contents = [{ role: "user", parts }];
```

with:

```ts
    const contents: GeminiContent[] = opts.messages
      ? buildContents(opts)
      : [{ role: "user", parts: buildParts(opts) }];
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/__tests__/gemini.test.ts`
Expected: PASS — all pre-existing direct-API and Vertex tests plus the new multi-turn tests for both connection modes.

- [ ] **Step 5: Commit**

```bash
git add src/gemini/index.ts src/__tests__/gemini.test.ts
git commit -m "feat(gemini): support multi-turn messages[] (direct API + Vertex)"
```

---

### Task 9: README updates

**Files:**
- Modify: `README.md:78-125`

- [ ] **Step 1: Edit the "Streaming and reasoning/thinking" section header and field list**

Replace:

```md
## Streaming and reasoning/thinking

`LlmGenerateOptions` has three additive fields. A strategy that doesn't
implement one of them simply ignores it — every existing caller keeps
working unchanged.

- **`onToken?: (delta: string) => void`** — called with each incremental
```

with:

```md
## Multi-turn conversation history

`LlmGenerateOptions.prompt` is now optional. Pass `messages: LlmMessage[]`
instead for multi-turn conversation history:

```ts
interface LlmMessage {
  role: "user" | "assistant";
  content: string;
}
```

Exactly one of `prompt` / `messages` must be set — both, or neither,
throws `InvalidGenerateOptionsError` before any network call. When
`messages` is used, `attachments` apply only to the **last** turn (same
place they'd land if you were building that turn as a single-turn
`prompt` call). All 5 built-in strategies (Claude, Gemini, ChatGPT, Grok,
DeepSeek) implement `messages` natively — none of them throw
`UnsupportedMultiTurnError` (that error exists for custom third-party
strategies that choose not to support it).

Gemini maps roles to its own vocabulary: `"assistant"` → `"model"`,
`"user"` → `"user"` — an implementation detail only relevant if you're
reading or extending `GeminiStrategy` directly; callers use the same
`LlmMessage.role` values (`"user"`/`"assistant"`) for every provider.

```ts
const result = await strategy.generate({
  messages: [
    { role: "user", content: "What's the capital of France?" },
    { role: "assistant", content: "Paris." },
    { role: "user", content: "And its population?" },
  ],
});
```

## Streaming and reasoning/thinking

`LlmGenerateOptions` has three additive fields. A strategy that doesn't
implement one of them simply ignores it — every existing caller keeps
working unchanged.

- **`onToken?: (delta: string) => void`** — called with each incremental
```

- [ ] **Step 2: Add a Multi-turn column to the per-provider support table**

Replace:

```md
| Provider | Streaming | Thinking |
|---|---|---|
| Claude | ✅ `messages.stream()` | `adaptive`, `budget` (`budget_tokens` ≥1024 and < `maxTokens`). `effort` unsupported. |
| Gemini | ✅ both connection modes | **Vertex only** (`GeminiStrategy({ connection: "vertex" })`) — `adaptive`, `budget`. The direct API SDK has no thinking support at all; any `thinking` request on that connection throws `UnsupportedThinkingModeError`. |
| ChatGPT | ✅ `stream:true` | `effort` only (`reasoning_effort`). Chat Completions never returns a reasoning trace — `response.thinking` is always `undefined` for this strategy. |
| Grok | ✅ `stream:true` | **Not implemented.** xAI's reasoning-effort parameter contract could not be verified against primary documentation — any `thinking` request throws `UnsupportedThinkingModeError` until this is confirmed against docs.x.ai. |
| DeepSeek | ✅ `stream:true` | **Not implemented as a request option**, same reason as Grok. `deepseek-reasoner`'s `reasoning_content` is instead surfaced passively via `response.thinking` whenever the model returns it — inherent to the model, not caller-controlled. |
```

with:

```md
| Provider | Streaming | Thinking | Multi-turn |
|---|---|---|---|
| Claude | ✅ `messages.stream()` | `adaptive`, `budget` (`budget_tokens` ≥1024 and < `maxTokens`). `effort` unsupported. | ✅ |
| Gemini | ✅ both connection modes | **Vertex only** (`GeminiStrategy({ connection: "vertex" })`) — `adaptive`, `budget`. The direct API SDK has no thinking support at all; any `thinking` request on that connection throws `UnsupportedThinkingModeError`. | ✅ |
| ChatGPT | ✅ `stream:true` | `effort` only (`reasoning_effort`). Chat Completions never returns a reasoning trace — `response.thinking` is always `undefined` for this strategy. | ✅ |
| Grok | ✅ `stream:true` | **Not implemented.** xAI's reasoning-effort parameter contract could not be verified against primary documentation — any `thinking` request throws `UnsupportedThinkingModeError` until this is confirmed against docs.x.ai. | ✅ |
| DeepSeek | ✅ `stream:true` | **Not implemented as a request option**, same reason as Grok. `deepseek-reasoner`'s `reasoning_content` is instead surfaced passively via `response.thinking` whenever the model returns it — inherent to the model, not caller-controlled. | ✅ |
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document multi-turn messages[] support"
```

---

### Task 10: Changeset

**Files:**
- Create: `.changeset/<auto-generated-name>.md` via the CLI (do not hand-write it — the CLI's interactive prompts pick the package and bump type correctly).

- [ ] **Step 1: Run the changeset CLI**

Run: `npx changeset`
When prompted:
- Select `@idevconn/llm-router` (the only package in this repo).
- Bump type: **minor** (new, backward-compatible feature).
- Summary: `Add multi-turn messages[] support across all provider adapters (Claude, Gemini, ChatGPT, Grok, DeepSeek); prompt is now optional, InvalidGenerateOptionsError when both/neither of prompt/messages are set.`

- [ ] **Step 2: Verify the generated file**

Run: `cat .changeset/*.md` and confirm it references `@idevconn/llm-router` with a `minor` bump and the summary above.

- [ ] **Step 3: Commit**

```bash
git add .changeset/
git commit -m "chore: add changeset for multi-turn messages[] support"
```

---

### Task 11: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Typecheck**

Run: `npm run typecheck`
Expected: PASS, zero errors.

- [ ] **Step 2: Lint**

Run: `npm run lint`
Expected: PASS, zero errors.

- [ ] **Step 3: Full test suite**

Run: `npm test`
Expected: PASS — every test in `src/__tests__/`, including every pre-existing test in `claude.test.ts`, `gemini.test.ts`, `chatgpt.test.ts`, `grok.test.ts`, `deepseek.test.ts`, `index-exports.test.ts`, `task-router.test.ts`, `orchestrator.test.ts`, etc., unmodified and green.

- [ ] **Step 4: Build**

Run: `npm run build`
Expected: PASS — `tsup` rebuilds `dist/` with no errors. This matters because the compliance consumer repo links/installs from a local path, not the npm registry, so a stale `dist/` would silently ship the old behavior. `dist/` is gitignored in this repo, so there is nothing to commit for this step — it only needs to exist on disk for local linking.

- [ ] **Step 5: Final status check**

Run: `git status`
Expected: clean (everything already committed per-task).
