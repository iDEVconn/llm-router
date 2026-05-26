import {
  InvalidPlatformProviderError,
  NoPlatformProviderError,
  UnknownProviderError,
} from "./errors";
import type { LlmStrategy } from "./types";

interface LlmRegistryOptions<TName extends string> {
  /** Concrete strategy instances. Order doesn't matter; lookup is by `providerName`. */
  strategies: LlmStrategy[];
  /**
   * Provider name returned by `getPlatform()`. Pass `null` (or omit) to run
   * in BYOK-only mode — every caller must supply their own key.
   */
  platform?: TName | null;
  /**
   * Optional env-key audit. Maps each provider name to the env var that
   * holds its platform-level API key (e.g. `{ gemini: "GEMINI_API_KEY" }`).
   * The registry checks `env[envKey]` at construction time and logs a
   * warning per missing key.
   */
  providerEnvKeys?: Partial<Record<TName, string>>;
  /** Env source for the audit. Defaults to `process.env`. */
  env?: Record<string, string | undefined>;
  /** Optional logger. Defaults to `console`. */
  logger?: { warn: (msg: string) => void };
  /**
   * Optional label map used in audit messages (e.g. `{ gemini: "Gemini" }`).
   * Falls back to the provider name when not provided.
   */
  providerLabels?: Partial<Record<TName, string>>;
}

/**
 * Routes LLM calls to one of N registered strategies. The platform-default
 * strategy is chosen at construction time (typically from an env var);
 * callers can override it per request by passing a different provider name
 * to `get(name)`. Keep one instance per process — stateless, cheap, safe
 * to share across concurrent requests.
 */
export class LlmRegistry<TName extends string = string> {
  private readonly byName: ReadonlyMap<string, LlmStrategy>;
  private readonly platformStrategy: LlmStrategy | null;
  private readonly platformName: TName | null;

  constructor(opts: LlmRegistryOptions<TName>) {
    const entries: [string, LlmStrategy][] = opts.strategies.map((s) => [s.providerName, s]);
    const map = new Map<string, LlmStrategy>(entries);
    if (map.size !== entries.length) {
      throw new Error("Duplicate strategy providerName values are not allowed.");
    }
    this.byName = map;

    if (opts.platform === null || opts.platform === undefined) {
      this.platformStrategy = null;
      this.platformName = null;
      opts.logger?.warn(
        "[llm-router] No platform provider configured — running in BYOK-only mode.",
      );
    } else {
      const platformStrategy = map.get(opts.platform);
      if (!platformStrategy) {
        throw new InvalidPlatformProviderError(opts.platform, [...map.keys()]);
      }
      this.platformStrategy = platformStrategy;
      this.platformName = opts.platform;
    }

    this.auditEnvKeys(opts);
  }

  /** Lookup by provider name. Throws `UnknownProviderError` when unknown. */
  get(name: TName | string): LlmStrategy {
    const strategy = this.byName.get(name);
    if (!strategy) throw new UnknownProviderError(name);
    return strategy;
  }

  /** Platform-default strategy. Throws `NoPlatformProviderError` in BYOK-only mode. */
  getPlatform(): LlmStrategy {
    if (!this.platformStrategy) throw new NoPlatformProviderError();
    return this.platformStrategy;
  }

  /** Platform-default provider name. `null` in BYOK-only mode. */
  getPlatformProviderName(): TName | null {
    return this.platformName;
  }

  /** Type guard for raw input strings (e.g. from a DB column or query param). */
  isProviderName(value: string): value is TName {
    return this.byName.has(value);
  }

  /** All registered provider names (cheap enumeration helper). */
  listProviderNames(): readonly string[] {
    return [...this.byName.keys()];
  }

  private auditEnvKeys(opts: LlmRegistryOptions<TName>): void {
    if (!opts.providerEnvKeys) return;
    const env = opts.env ?? (typeof process !== "undefined" ? process.env : {});
    const logger = opts.logger ?? console;

    for (const [name, envKey] of Object.entries(opts.providerEnvKeys) as [
      TName,
      string,
    ][]) {
      if (!envKey) continue;
      if ((env[envKey] ?? "").trim()) continue;

      const label = opts.providerLabels?.[name] ?? name;
      if (name === this.platformName) {
        logger.warn(
          `[llm-router] ${envKey} is not set — ${label} is the active platform strategy but has no key, so platform calls will fail.`,
        );
      } else {
        logger.warn(
          `[llm-router] ${envKey} is not set — ${label} cannot be used as a platform fallback. Users on ${label} must supply a BYOK key.`,
        );
      }
    }
  }
}
