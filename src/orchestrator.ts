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
  /**
   * Set only when `synthesize: true` was requested and the synthesis call
   * failed. Lets a caller tell "synthesis wasn't asked for" apart from
   * "synthesis was attempted and failed" — both leave `final` unset.
   */
  synthesisError?: string;
}

const DEFAULT_MAX_ROUNDS = 1;
const DEFAULT_MAX_SUBTASKS = 20;
const DEFAULT_MAX_CONCURRENCY = 3;

interface AvailableProvider {
  name: string;
  capabilities: readonly string[];
}

/** One decompose attempt: parsed subtasks, or why it failed. */
interface DecomposeAttempt {
  subtasks: Subtask[] | null;
  /**
   * Set only when `strategy.generate()` itself rejected (network, 401, rate
   * limit) — i.e. a provider failure, not a "model returned garbage" failure.
   */
  generateError?: unknown;
}

function errorMessage(cause: unknown): string {
  return cause instanceof Error ? cause.message : String(cause);
}

function stripFence(text: string): string {
  return text.trim().replace(/^```(?:json)?\n?/, "").replace(/```$/, "");
}

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
      opts.apiKeys,
    );

    const availableProviders = await this.taskRouter.listAvailableProviders({
      apiKeys: opts.apiKeys,
      providerOverrides: opts.providerOverrides,
      metaProvider: opts.metaProvider,
    });

    const maxConcurrency = opts.maxConcurrency ?? DEFAULT_MAX_CONCURRENCY;
    const results = await runWithConcurrency(subtasks, maxConcurrency, (subtask) =>
      this.routeAndExecute(subtask, opts, availableProviders),
    );

    const result: OrchestratorResult = { subtasks: results };
    if (truncatedSubtaskCount > 0) result.truncatedSubtaskCount = truncatedSubtaskCount;
    if (opts.synthesize) {
      try {
        result.final = await this.synthesize(
          taskText,
          results,
          opts.metaProvider,
          opts.apiKeys,
          truncatedSubtaskCount,
        );
      } catch (cause) {
        // Synthesis is a best-effort final step layered on top of subtasks
        // that already exist — a BYOK key error or a transient provider
        // failure here must not discard the completed subtask breakdown.
        // Leave `final` unset, but record why so the caller can tell this
        // apart from the synthesize: false case.
        result.synthesisError = errorMessage(cause);
      }
    }
    return result;
  }

  private getMetaProviderName(metaProvider?: string): string {
    return metaProvider ?? this.registry.getPlatform().providerName;
  }

  private async decompose(
    taskText: string,
    metaProvider: string | undefined,
    maxSubtasks: number,
    apiKeys: Record<string, string> | undefined,
  ): Promise<{ subtasks: Subtask[]; truncatedSubtaskCount: number }> {
    const metaProviderName = this.getMetaProviderName(metaProvider);
    const strategy = this.registry.get(metaProviderName);
    const apiKey = apiKeys?.[metaProviderName];
    const basePrompt =
      `Split the following task into independent subtasks. Respond with ONLY a JSON array ` +
      `of objects: {"id": string, "description": string, "requiredCapabilities"?: string[]}. ` +
      `requiredCapabilities may only use these tags: ${KNOWN_CAPABILITY_TAGS.join(", ")}.\n\n` +
      `Task: ${taskText}`;

    let attempt = await this.tryParseSubtasks(strategy, basePrompt, apiKey);
    if (!attempt.subtasks) {
      attempt = await this.tryParseSubtasks(
        strategy,
        `${basePrompt}\n\nYour previous response was not valid JSON. Return only the JSON array, no prose.`,
        apiKey,
      );
    }
    if (!attempt.subtasks) {
      // Prefer the real provider error when the last attempt failed because
      // generate() rejected; only fall back to the synthetic message when the
      // model actually answered but the payload was unusable.
      throw new TaskDecompositionError(
        attempt.generateError ??
          new Error("Model did not return a parseable JSON array after one retry."),
      );
    }

    const subtasks = attempt.subtasks;
    const truncatedSubtaskCount = Math.max(0, subtasks.length - maxSubtasks);
    return { subtasks: subtasks.slice(0, maxSubtasks), truncatedSubtaskCount };
  }

  private async tryParseSubtasks(
    strategy: {
      generate: (opts: { prompt: string; apiKey?: string }) => Promise<{ text: string }>;
    },
    prompt: string,
    apiKey: string | undefined,
  ): Promise<DecomposeAttempt> {
    let response: { text: string };
    try {
      response = await strategy.generate({ prompt, apiKey });
    } catch (generateError) {
      // A rejected generate() (network/auth/rate-limit) is a provider failure,
      // not a "model returned garbage" failure. Keep the real error so
      // decompose() can surface it as the TaskDecompositionError cause instead
      // of misattributing it to unparseable JSON.
      return { subtasks: null, generateError };
    }

    try {
      const parsed = JSON.parse(stripFence(response.text));
      if (!Array.isArray(parsed)) return { subtasks: null };
      if (
        !parsed.every(
          (item) => typeof item?.id === "string" && typeof item?.description === "string",
        )
      ) {
        return { subtasks: null };
      }
      return {
        subtasks: parsed.map(
          (item: { id: string; description: string; requiredCapabilities?: string[] }) => ({
            id: item.id,
            description: item.description,
            requiredCapabilities: item.requiredCapabilities,
          }),
        ),
      };
    } catch {
      // The model answered, but not with a parseable JSON array — a genuine
      // decomposition failure that decompose()'s retry path handles.
      return { subtasks: null };
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
        // Without this the router's own LLM-fallback stage would fall back to
        // registry.getPlatform(), which throws in BYOK-only mode even when the
        // caller supplied both a metaProvider and its apiKey.
        metaProvider: opts.metaProvider,
      });
      decision = decisions[0]!;
    } catch (cause) {
      return { subtask, result: "", rounds: 0, unresolved: true, error: errorMessage(cause) };
    }

    try {
      return await this.executeSubtask(subtask, decision, opts, availableProviders);
    } catch (cause) {
      // Reached only when the very first generateForSubtask() call fails —
      // there is no prior output to preserve in that case.
      return {
        subtask,
        decision,
        result: "",
        rounds: 0,
        unresolved: true,
        error: errorMessage(cause),
      };
    }
  }

  private async synthesize(
    taskText: string,
    results: SubtaskResult[],
    metaProvider: string | undefined,
    apiKeys: Record<string, string> | undefined,
    truncatedSubtaskCount: number,
  ): Promise<string> {
    const metaProviderName = this.getMetaProviderName(metaProvider);
    const strategy = this.registry.get(metaProviderName);
    const apiKey = apiKeys?.[metaProviderName];
    const summary = results
      .map((r) => {
        const status = r.error
          ? `error: ${r.error}`
          : `unresolved: ${r.unresolved}`;
        return `- ${r.subtask.id} (${r.subtask.description}) [${status}]:\n${r.result || "(no output)"}`;
      })
      .join("\n\n");

    // Subtasks dropped before routing are a gap too — synthesis can't note
    // what it never sees, so tell it explicitly (spec §8.5).
    const truncationNote =
      truncatedSubtaskCount > 0
        ? `${truncatedSubtaskCount} additional subtask${truncatedSubtaskCount === 1 ? "" : "s"} ` +
          `were dropped before routing because of the maxSubtasks limit and were never attempted. ` +
          `Note that gap as well.\n\n`
        : "";

    const prompt =
      `Original task: ${taskText}\n\nSubtask results:\n${summary}\n\n${truncationNote}` +
      `Write the final combined answer. If any subtask above is marked "unresolved: true" ` +
      `or has an "error", explicitly note that gap instead of presenting a fully confident answer.`;

    const response = await strategy.generate({ prompt, apiKey });
    return response.text;
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

    // Once a first answer exists, a failure in the critique/retry machinery
    // must not throw it away: return that answer flagged best-effort rather
    // than letting routeAndExecute's catch blank the result out (spec §8.3).
    let rounds = 0;
    while (rounds < maxRounds) {
      let verdict: { approved: boolean; feedback?: string };
      try {
        verdict = await this.critique(subtask, output, decision, opts, availableProviders);
      } catch (cause) {
        return {
          subtask,
          decision,
          result: output,
          rounds,
          unresolved: true,
          error: errorMessage(cause),
        };
      }

      if (verdict.approved) {
        return { subtask, decision, result: output, rounds, unresolved: false };
      }

      rounds += 1;
      try {
        output = await this.generateForSubtask(
          strategy,
          subtask,
          decision,
          apiKey,
          verdict.feedback,
        );
      } catch (cause) {
        return {
          subtask,
          decision,
          result: output,
          rounds,
          unresolved: true,
          error: errorMessage(cause),
        };
      }
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
