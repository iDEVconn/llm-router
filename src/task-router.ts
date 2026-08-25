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
      decisions.push(await this.routeOne(subtask, available, opts.apiKeys));
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

  private async routeOne(
    subtask: Subtask,
    available: ResolvedProvider[],
    apiKeys?: Record<string, string>,
  ): Promise<RoutingDecision> {
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

    return this.llmFallback(subtask, available, apiKeys);
  }

  private getMetaProviderName(): string {
    return this.metaProviderName ?? this.registry.getPlatform().providerName;
  }

  private async llmFallback(
    subtask: Subtask,
    available: ResolvedProvider[],
    apiKeys?: Record<string, string>,
  ): Promise<RoutingDecision> {
    const metaProviderName = this.getMetaProviderName();
    const metaStrategy = this.registry.get(metaProviderName);
    const candidateList = available
      .map((p) => `- ${p.name} (capabilities: ${p.capabilities.join(", ") || "none listed"})`)
      .join("\n");

    const prompt =
      `Pick the best provider for this subtask from the list below. ` +
      `Respond with ONLY a JSON object: {"provider": string, "model"?: string, "rationale"?: string}.\n\n` +
      `Subtask: ${subtask.description}\n\nAvailable providers:\n${candidateList}`;

    const response = await metaStrategy.generate({ prompt, apiKey: apiKeys?.[metaProviderName] });
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
