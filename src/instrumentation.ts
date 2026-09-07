import type { LlmGenerateOptions, LlmResponse, LlmStrategy, LlmUsage } from "./types";

export interface LlmCallEvent {
  provider: string;
  model: string;
  usage: LlmUsage;
  truncated: boolean;
  latencyMs: number;
  timestamp: string;
  error?: string;
}

export interface WithInstrumentationOptions {
  onCall: (event: LlmCallEvent) => void;
}

/**
 * No bundled logger — callers decide where `onCall` events go (pino,
 * winston, a NestJS Logger, a test spy, ...).
 */
export function withInstrumentation(
  strategy: LlmStrategy,
  opts: WithInstrumentationOptions,
): LlmStrategy {
  return {
    ...strategy,
    async generate(genOpts: LlmGenerateOptions): Promise<LlmResponse> {
      const start = Date.now();
      try {
        const response = await strategy.generate(genOpts);
        opts.onCall({
          provider: strategy.providerName,
          model: response.model,
          usage: response.usage,
          truncated: response.truncated,
          latencyMs: Date.now() - start,
          timestamp: new Date().toISOString(),
        });
        return response;
      } catch (err) {
        opts.onCall({
          provider: strategy.providerName,
          model: genOpts.model ?? strategy.defaultModel,
          usage: { inputTokens: 0, outputTokens: 0 },
          truncated: false,
          latencyMs: Date.now() - start,
          timestamp: new Date().toISOString(),
          error: err instanceof Error ? err.message : String(err),
        });
        throw err;
      }
    },
  };
}

/** Applies decorators left-to-right: `compose(s, a, b)` behaves like `b(a(s))`. */
export function compose(
  strategy: LlmStrategy,
  ...decorators: Array<(s: LlmStrategy) => LlmStrategy>
): LlmStrategy {
  return decorators.reduce((acc, decorator) => decorator(acc), strategy);
}
