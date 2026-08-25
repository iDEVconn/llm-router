# Task Orchestrator — Design Spec

Date: 2026-08-25
Status: Approved for planning (see L99 review below)

## 1. Motivation

`llm-router` currently routes one `generate()` call to one explicitly-named
provider. There is no way to hand it a whole task, have it figure out which
registered provider is best suited for which part, and get results back —
callers do that decomposition and routing by hand today.

This spec adds a `TaskRouter` (pure routing decision) and an `Orchestrator`
(decompose → route → execute-with-critique → optional synthesis) on top of
the existing `LlmRegistry`. It also adds two new provider strategies
(`chatgpt`, `deepseek`) so the router has more than three candidates to
choose from.

**Explicitly out of scope:** this is not a code-editing / tool-use agent.
Every step is a `generate()` text call. There is no file access, no git, no
test execution. It's the text-only analog of subagent-driven-development's
implementer→reviewer→fix-loop shape, not a reimplementation of it.

## 2. Goals

- Given a free-text task, split it into subtasks and run each on the
  registered provider best suited for it, using only providers that are
  actually usable (platform key configured, or a BYOK key supplied for this
  call).
- Cheap, deterministic routing first (tag matching); fall back to an LLM
  call only when tags don't decide it.
- Self-correct low-quality output via a bounded critique/retry loop before
  giving up and returning a flagged best-effort result.
- Never let one bad run silently explode cost (unbounded subtask fan-out)
  or silently swallow a partial failure (one subtask's error hiding the
  others' results).
- Zero breaking impact on existing consumers who implement `LlmStrategy`
  themselves (the Ollama example in the README) unless they opt in.

## 3. Non-goals

- No real subagent dispatch, no tool use, no file/git operations.
- No adaptive re-routing mid-run based on a subtask's own output (a subtask
  routes once; only the critique/retry loop can change its content, not its
  provider).
- No cross-run learning/memory of which provider historically performed
  best (static tag-matching + one-shot LLM fallback only).

## 4. Architecture overview

New files:

- `src/task-router.ts` — `TaskRouter`. Pure decision logic: given subtasks
  and available providers, returns routing decisions. No network calls
  except the LLM-fallback path, which goes through the registry like any
  other `generate()` call — but the class has no execution loop, no retry
  logic, nothing stateful. Unit-testable with fake `LlmStrategy` stubs.
- `src/orchestrator.ts` — `Orchestrator`. Owns the full pipeline: decompose
  → `TaskRouter.route()` → per-subtask execute/critique/retry → optional
  synthesis. Depends on `LlmRegistry` and `TaskRouter`.
- `src/chatgpt/index.ts` — `ChatGptStrategy`, same shape as `GrokStrategy`
  (OpenAI SDK, default OpenAI base URL).
- `src/deepseek/index.ts` — `DeepSeekStrategy`, same shape as
  `GrokStrategy` (OpenAI SDK, `baseURL: "https://api.deepseek.com"`).

Both new strategies are new subpath exports (`@idevconn/llm-router/chatgpt`,
`/deepseek`), following the existing `/gemini`, `/claude`, `/grok` pattern.
They reuse the `openai` peer dependency already declared — no new peer
deps. `openai` moves from "peer dep used by grok only" to "peer dep used by
grok, chatgpt, and deepseek."

`task-router.ts` and `orchestrator.ts` import no provider SDK — both live
in the main entry, consistent with "pure router core, zero SDK
dependencies on the main entry."

## 5. `LlmStrategy` interface change — Hybrid (Option C)

Two **optional** additions to `LlmStrategy` (types.ts):

```ts
export interface LlmStrategy {
  // ...existing members unchanged...

  /**
   * True when this strategy has a usable platform-level key configured
   * (constructor-supplied, not per-call BYOK). Optional for backward
   * compatibility — a strategy that omits this is treated by TaskRouter
   * as "unknown," not "available." All three built-in strategies
   * (claude, gemini, grok) already have an equivalent private check;
   * this just promotes it to the interface, and chatgpt/deepseek
   * implement it too.
   */
  hasPlatformKey?(): boolean;

  /**
   * Free-text capability tags used by TaskRouter's rule stage (e.g.
   * `["vision", "code", "cheap", "long-context", "reasoning"]`). Optional;
   * a strategy without tags simply never wins the rule stage and routes
   * through the LLM-fallback stage instead.
   */
  readonly capabilities?: readonly string[];
}
```

Neither addition is required, so an existing consumer's custom
`LlmStrategy` (per the README's Ollama example) keeps compiling untouched
on upgrade. It just won't participate in rule-based availability/capability
matching until it adds one or both — or until the caller supplies an
explicit override (next section).

**Safe-default rule:** if `hasPlatformKey` is absent, `TaskRouter` treats
the provider as unavailable unless the caller's `apiKeys` map (or the
override list) says otherwise. Never assume a provider is usable when it
can't be asked.

**Override escape hatch:** `TaskRouter.route()` and `Orchestrator.run()`
accept an optional `providerOverrides` list:

```ts
interface ProviderDescriptor {
  provider: string;
  capabilities?: readonly string[];
  available?: boolean;
}
```

When a provider name appears in `providerOverrides`, its values win over
whatever the registered `LlmStrategy` reports (or fails to report). This
is how a consumer with a capability-less or `hasPlatformKey`-less custom
strategy opts into full routing participation without touching the
strategy class itself.

## 6. Capability vocabulary

Tags are free strings on `LlmStrategy.capabilities`, but the **decompose
step's prompt is constrained to a closed, exported list**
(`src/task-router.ts`'s `KNOWN_CAPABILITY_TAGS`, e.g. `vision`, `code`,
`long-context`, `cheap`, `reasoning`, `multilingual`). The decompose prompt
includes this exact list and instructs the model to only use tags from it
for `requiredCapabilities`. This keeps the two independently-produced tag
sets (hand-written on strategies, LLM-generated on subtasks) drawing from
the same vocabulary, so the rule stage actually has a chance to match
instead of degrading to LLM-fallback on every subtask. The list is
exported so consumers writing custom strategies know what to tag against.

## 7. `TaskRouter`

```ts
interface Subtask {
  id: string;
  description: string;
  requiredCapabilities?: readonly string[];
}

interface RoutingDecision {
  subtaskId: string;
  provider: string;
  model?: string;
  method: "rule" | "llm-fallback";
  rationale?: string;
}

interface TaskRouterOptions {
  registry: LlmRegistry;
  /** Defaults to `registry.getPlatform()`'s provider name. */
  metaProvider?: string;
}

class TaskRouter {
  constructor(opts: TaskRouterOptions);
  route(
    subtasks: readonly Subtask[],
    opts?: { apiKeys?: Record<string, string>; providerOverrides?: readonly ProviderDescriptor[] },
  ): Promise<RoutingDecision[]>;
}
```

Algorithm per subtask:

1. Compute the available-provider set: every registered provider name
   where (override says `available: true`) OR (no override, and
   `hasPlatformKey?.() === true`) OR (`apiKeys[name]` is set).
2. If `subtask.requiredCapabilities` is empty/absent, or no available
   provider's tags intersect it at all, go straight to LLM-fallback.
3. Otherwise score each available provider by tag-overlap count. A unique
   top scorer → `method: "rule"`. A tie → LLM-fallback (ties are exactly
   the case rules can't decide).
4. LLM-fallback: one `generate()` call through `metaProvider`, given the
   subtask description and the available-provider list (name +
   capabilities only — never a provider absent from the available set), asking
   for `{provider, model?, rationale}` as JSON. If the returned provider
   isn't in the available set, treat it as router error
   (`NoAvailableProviderError`) rather than silently using it.
5. If the available-provider set is empty for a subtask, throw
   `NoAvailableProviderError` for that subtask (caught by the Orchestrator
   per the partial-failure policy in §8.4 — it does not abort the whole run).

One LLM-fallback call per ambiguous subtask (not batched). Simpler; a
future optimization can batch all of a run's ambiguous subtasks into one
classifier call if per-subtask cost becomes a concern.

## 8. `Orchestrator`

```ts
interface RunOptions {
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

interface SubtaskResult {
  subtask: Subtask;
  decision: RoutingDecision;
  result: string;
  rounds: number;
  unresolved: boolean;
  /** Set when the subtask errored (network/provider failure or
   *  NoAvailableProviderError) instead of completing. `result` is empty
   *  and `unresolved` is true when this is set. */
  error?: string;
}

interface OrchestratorResult {
  subtasks: SubtaskResult[];
  /** Present only when `synthesize: true`. */
  final?: string;
}

class Orchestrator {
  constructor(opts: { registry: LlmRegistry; taskRouter?: TaskRouter });
  run(taskText: string, opts?: RunOptions): Promise<OrchestratorResult>;
}
```

### 8.1 Decompose

One `generate()` call through `metaProvider`, prompt asks for a JSON array
of `{id, description, requiredCapabilities?}` using only tags from
`KNOWN_CAPABILITY_TAGS` (§6).

Parsing: strip a leading/trailing ``` fenced block if present, then
`JSON.parse`. On failure, retry once with a follow-up message ("that
wasn't valid JSON, return only the JSON array, no prose"). A second
failure throws `TaskDecompositionError` — the whole `run()` rejects here,
since there's nothing to route yet.

If the parsed array exceeds `maxSubtasks`, truncate to the first
`maxSubtasks` entries and note the truncation on the returned
`OrchestratorResult` (a `truncatedSubtaskCount` field) rather than
silently dropping the rest.

### 8.2 Route

`taskRouter.route(subtasks, { apiKeys, providerOverrides })` — see §7.
A subtask whose routing throws `NoAvailableProviderError` gets a
`SubtaskResult` with `error` set and skips execution (§8.4); it does not
block routing/execution of the other subtasks.

### 8.3 Execute-with-critique loop (per subtask, run concurrently up to `maxConcurrency`)

State machine, precisely, for `maxRounds = R`:

1. `generate()` the subtask via its routed provider/model. `rounds = 0`.
2. If `R === 0`, stop here: return the output, `unresolved: false`
   (no critique requested means no verification requested — the caller
   opted out, so the raw output is definitionally the final answer, not
   flagged as unresolved).
3. Otherwise, critique the output:
   - `critique: "self"` — the same provider, one more `generate()` call
     asking "does this fully satisfy the subtask description? approved:
     bool, feedback?" against the subtask spec.
   - `critique: "cross"` — a second provider does the same critique call.
     For a rule-routed subtask, that's the available provider with the
     next-highest tag-overlap score. For an LLM-fallback-routed subtask
     (no score exists), it's the next available provider in registration
     order, excluding the one already chosen. Either way, ties break by
     registration order. If no second available provider exists at all,
     fall back to `"self"` for this subtask only.
4. `approved: true` → stop, return output, `unresolved: false`.
5. `approved: false` and `rounds < R` → one retry `generate()` call,
   passing the original description plus the critique feedback verbatim.
   `rounds += 1`. **The retry's output is not critiqued again** — one
   round is exactly one generate-critique-retry sequence, not a re-checked
   loop. Return the retry's output, `unresolved: false` if this was the
   last allowed round (retry output is unverified but it's what the round
   budget bought — flagging it `unresolved` would make `unresolved`
   ambiguous between "never checked" and "checked and rejected twice").
6. `approved: false` and `rounds === R` already spent with no rounds left
   → return the last generated output (pre- or post-retry, whichever is
   most recent), `unresolved: true`.

In short: `unresolved` means "the critique step explicitly rejected this
and no budget was left to react to that rejection" — never "we didn't
bother checking."

### 8.4 Partial-failure policy

A subtask can fail two ways: routing (`NoAvailableProviderError`) or
execution (provider `generate()` throws — network, 5xx, auth). Either way:
that subtask's `SubtaskResult` gets `error` set, `unresolved: true`,
`result: ""`, and the run continues for every other subtask.
`Orchestrator.run()` itself only rejects on `TaskDecompositionError`
(nothing to route yet) — once subtasks exist, the run always resolves with
a per-subtask breakdown of what worked and what didn't.

### 8.5 Synthesis (opt-in)

When `synthesize: true`, one final `generate()` call through
`metaProvider` after all subtasks settle. The prompt includes every
subtask's result **and its `unresolved`/`error` status** explicitly, and
is instructed to note gaps rather than paper over them — the synthesis
must not present a confident final answer built silently on a failed or
unresolved part.

## 9. New errors (`src/errors.ts`)

```ts
export class TaskDecompositionError extends Error { /* provider, cause */ }
export class NoAvailableProviderError extends Error { /* subtaskId */ }
```

Both follow the existing plain-`Error`-subclass convention (framework
agnostic, named `.name`, no HTTP coupling).

## 10. New provider strategies

`ChatGptStrategy` and `DeepSeekStrategy` copy `GrokStrategy`'s shape
exactly: `openai` SDK, constructor-configurable `baseURL` (default
`https://api.openai.com/v1` and `https://api.deepseek.com` respectively),
`hasPlatformKey()`, the same `generate()`/`validateKey()` contract. Vision
attachment support (which MIME types each accepts) and default model
naming are implementation details for the plan/implementation phase, not
architectural — they follow whatever each strategy's own vision endpoint
actually supports, same as Grok's image-only restriction today.

## 11. Testing approach

- `TaskRouter`: unit tests with fake `LlmStrategy` objects (no network),
  covering rule-stage tag matching, ties, unknown-availability defaults,
  override precedence, and `NoAvailableProviderError`.
- `Orchestrator`: unit tests with fake strategies whose `generate()`
  returns scripted responses per call in sequence, covering the critique
  state machine (approved-first-try, rejected-then-retry,
  rejected-with-no-budget, cross-critique fallback-to-self,
  partial-failure isolation, `maxSubtasks` truncation).
- `ChatGptStrategy` / `DeepSeekStrategy`: mirror the existing
  `grok.test.ts` structure.

## 12. Open questions / deferred to future work

- Batching LLM-fallback routing calls across a run's ambiguous subtasks
  into one classifier call (cost optimization, not correctness).
- Adaptive re-routing mid-run based on a subtask's own critique history.
- Default `capabilities` tag sets for the built-in strategies — decided at
  implementation time per provider's actual documented strengths, not
  architectural.

## Appendix: L99 review summary

An architectural review (senior-architect deep-analysis pass) surfaced
eight gaps in the first draft of this design before it was written down:
an unannounced breaking change on `hasPlatformKey`, an ungoverned
capability-tag vocabulary, an unspecified critique-loop state machine, no
subtask fan-out cap, no concurrency cap, no partial-failure policy,
synthesis blind to unresolved/errored subtasks, and a fragile
non-retrying JSON parse on decompose. All eight are addressed above (§5,
§6, §8.1, §8.3, §8.4, §8.5 respectively) rather than deferred.
