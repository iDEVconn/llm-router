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
      opts.apiKeys,
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

    let subtasks = await this.tryParseSubtasks(strategy, basePrompt, apiKey);
    if (!subtasks) {
      subtasks = await this.tryParseSubtasks(
        strategy,
        `${basePrompt}\n\nYour previous response was not valid JSON. Return only the JSON array, no prose.`,
        apiKey,
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
    strategy: {
      generate: (opts: { prompt: string; apiKey?: string }) => Promise<{ text: string }>;
    },
    prompt: string,
    apiKey: string | undefined,
  ): Promise<Subtask[] | null> {
    try {
      const response = await strategy.generate({ prompt, apiKey });
      const parsed = JSON.parse(stripFence(response.text));
      if (!Array.isArray(parsed)) return null;
      if (
        !parsed.every(
          (item) => typeof item?.id === "string" && typeof item?.description === "string",
        )
      ) {
        return null;
      }
      return parsed.map(
        (item: { id: string; description: string; requiredCapabilities?: string[] }) => ({
          id: item.id,
          description: item.description,
          requiredCapabilities: item.requiredCapabilities,
        }),
      );
    } catch {
      // Covers both a rejected generate() call (network/auth/rate-limit) and
      // JSON-parse/shape failures — either way this attempt didn't produce a
      // usable subtask array, so decompose()'s retry-then-error path handles it.
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
