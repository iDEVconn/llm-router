import { beforeEach, describe, expect, it, vi } from "vitest";

const mockMessagesCreate = vi.fn();
const mockMessagesStream = vi.fn();

vi.mock("@anthropic-ai/sdk", () => {
  class Anthropic {
    public readonly messages: { create: typeof mockMessagesCreate; stream: typeof mockMessagesStream };
    constructor(public readonly opts: { apiKey: string }) {
      this.messages = { create: mockMessagesCreate, stream: mockMessagesStream };
    }
  }
  return { default: Anthropic };
});

import { CircuitBreakerOpenError, LlmAbortedError } from "../errors";
import * as pkg from "../index";
import { ClaudeStrategy } from "../claude/index";
import { withCircuitBreaker } from "../circuit-breaker";
import { withRateLimit } from "../rate-limit";
import { withRetry } from "../retry";
import { isAbortError } from "../resilience-errors";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy } from "../types";

function makeAborted(cause: unknown = new Error("Request was aborted.")): LlmAbortedError {
  return new LlmAbortedError({
    providerName: "claude",
    partialText: "Hello wor",
    partialThinking: "hmm",
    usage: { inputTokens: 12, outputTokens: 3 },
    usageEstimated: true,
    cause,
  });
}

/** A Claude-SDK-shaped stream that yields `deltas`, then aborts `controller` and throws `abortErr`. */
function abortingClaudeStream(deltas: string[], controller: AbortController, abortErr: unknown) {
  return {
    [Symbol.asyncIterator]: async function* () {
      for (const text of deltas) {
        yield { type: "content_block_delta", delta: { type: "text_delta", text } };
      }
      controller.abort();
      throw abortErr;
    },
    finalMessage: () => new Promise(() => {}),
  };
}

function okClaudeStream() {
  return {
    [Symbol.asyncIterator]: async function* () {
      yield { type: "content_block_delta", delta: { type: "text_delta", text: "ok" } };
    },
    finalMessage: () =>
      Promise.resolve({
        content: [{ type: "text", text: "ok" }],
        model: "claude-haiku-4-5",
        usage: { input_tokens: 1, output_tokens: 1 },
        stop_reason: "end_turn",
      }),
  };
}

describe("LlmAbortedError", () => {
  it("carries partial text, thinking, usage, estimate flag and cause", () => {
    const cause = new Error("Request was aborted.");
    const err = makeAborted(cause);
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect(err.name).toBe("AbortError");
    expect(err.providerName).toBe("claude");
    expect(err.partialText).toBe("Hello wor");
    expect(err.partialThinking).toBe("hmm");
    expect(err.usage).toEqual({ inputTokens: 12, outputTokens: 3 });
    expect(err.usageEstimated).toBe(true);
    expect(err.cause).toBe(cause);
    expect(err.message).toBe("claude request aborted (9 chars received)");
  });

  it("leaves partialThinking undefined when none was given", () => {
    const err = new LlmAbortedError({
      providerName: "grok",
      partialText: "",
      usage: { inputTokens: 1, outputTokens: 0 },
      usageEstimated: true,
      cause: "raw reason",
    });
    expect(err.partialThinking).toBeUndefined();
    expect(err.cause).toBe("raw reason");
    expect(err.message).toBe("grok request aborted (0 chars received)");
  });

  it("instanceof works across separately-bundled copies of the class (CJS subpath bundles)", async () => {
    vi.resetModules();
    const copy = await import("../errors");
    expect(copy.LlmAbortedError).not.toBe(LlmAbortedError);
    const fromCopy = new copy.LlmAbortedError({
      providerName: "claude",
      partialText: "x",
      usage: { inputTokens: 1, outputTokens: 1 },
      usageEstimated: true,
      cause: null,
    });
    expect(fromCopy).toBeInstanceOf(LlmAbortedError);
    expect(makeAborted()).toBeInstanceOf(copy.LlmAbortedError);
    expect(new Error("x")).not.toBeInstanceOf(LlmAbortedError);
    expect(Object.assign(new Error("x"), { name: "AbortError" })).not.toBeInstanceOf(
      LlmAbortedError,
    );
    expect(null).not.toBeInstanceOf(LlmAbortedError);
  });

  it("subclasses inherit the brand-based instanceof", () => {
    class ToolAbortedError extends LlmAbortedError {}
    const sub = new ToolAbortedError({
      providerName: "claude",
      partialText: "",
      usage: { inputTokens: 0, outputTokens: 0 },
      usageEstimated: true,
      cause: null,
    });
    expect(sub).toBeInstanceOf(LlmAbortedError);
    expect(sub).toBeInstanceOf(Error);
    expect(sub.name).toBe("AbortError");
  });

  it("is exported from the package entry point", () => {
    expect(pkg.LlmAbortedError).toBe(LlmAbortedError);
  });

  it("is recognised by isAbortError once the signal is aborted (and only then)", () => {
    const controller = new AbortController();
    expect(isAbortError(makeAborted(), controller.signal)).toBe(false);
    controller.abort();
    expect(isAbortError(makeAborted(), controller.signal)).toBe(true);
  });
});

describe("LlmAbortedError + resilience decorators", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  function fakeStrategy(): LlmStrategy & {
    generate: ReturnType<typeof vi.fn<(opts: LlmGenerateOptions) => Promise<LlmResponse>>>;
  } {
    return {
      providerName: "claude",
      defaultModel: "claude-haiku-4-5",
      generate: vi.fn<(opts: LlmGenerateOptions) => Promise<LlmResponse>>(),
      validateKey: vi.fn(),
    };
  }

  it("withRetry never retries an LlmAbortedError on an aborted signal", async () => {
    const strategy = fakeStrategy();
    const controller = new AbortController();
    controller.abort();
    const aborted = makeAborted();
    strategy.generate.mockRejectedValue(aborted);
    const onRetry = vi.fn();

    const retried = withRetry(strategy, {
      maxAttempts: 5,
      initialBackoffMs: 1,
      maxBackoffMs: 1,
      multiplier: 1,
      jitter: false,
      onRetry,
    });

    await expect(retried.generate({ prompt: "p", signal: controller.signal })).rejects.toBe(
      aborted,
    );
    expect(strategy.generate).toHaveBeenCalledTimes(1);
    expect(onRetry).not.toHaveBeenCalled();
  });

  it("an aborted streaming ClaudeStrategy call does not trip the circuit breaker, while the same raw error without an abort does", async () => {
    // The Anthropic SDK's APIUserAbortError has name "Error", not
    // "AbortError" — wrapping it is what lets isAbortError recognise it.
    const rawAbort = new Error("Request was aborted.");
    const breaker = withCircuitBreaker(new ClaudeStrategy({ apiKey: "k" }), {
      threshold: 1,
      samplingWindowMs: 60_000,
      resetTimeoutMs: 60_000,
    });

    const controller = new AbortController();
    mockMessagesStream.mockReturnValueOnce(abortingClaudeStream(["Hel", "lo"], controller, rawAbort));
    const err = await breaker
      .generate({ prompt: "p", onToken: () => {}, signal: controller.signal })
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(LlmAbortedError);
    expect((err as LlmAbortedError).partialText).toBe("Hello");
    expect((err as LlmAbortedError).cause).toBe(rawAbort);

    // Breaker still closed: the next call reaches the provider.
    mockMessagesStream.mockReturnValueOnce(okClaudeStream());
    const ok = await breaker.generate({ prompt: "p", onToken: () => {} });
    expect(ok.text).toBe("ok");
    expect(mockMessagesStream).toHaveBeenCalledTimes(2);

    // Control: the same raw error with no abort is a provider failure.
    mockMessagesStream.mockReturnValueOnce({
      [Symbol.asyncIterator]: async function* () {
        throw rawAbort;
      },
      finalMessage: () => new Promise(() => {}),
    });
    await expect(breaker.generate({ prompt: "p", onToken: () => {} })).rejects.toBe(rawAbort);
    await expect(breaker.generate({ prompt: "p", onToken: () => {} })).rejects.toBeInstanceOf(
      CircuitBreakerOpenError,
    );
    expect(mockMessagesStream).toHaveBeenCalledTimes(3);
  });

  it("withRetry passes an aborted streaming call through exactly once", async () => {
    const controller = new AbortController();
    mockMessagesStream.mockReturnValueOnce(
      abortingClaudeStream(["a"], controller, new Error("Request was aborted.")),
    );
    const retried = withRetry(new ClaudeStrategy({ apiKey: "k" }), {
      maxAttempts: 3,
      initialBackoffMs: 1,
      maxBackoffMs: 1,
      multiplier: 1,
      jitter: false,
    });

    await expect(
      retried.generate({ prompt: "p", onToken: () => {}, signal: controller.signal }),
    ).rejects.toBeInstanceOf(LlmAbortedError);
    expect(mockMessagesStream).toHaveBeenCalledTimes(1);
  });

  it("withRateLimit releases its concurrency slot after an aborted streaming call", async () => {
    const limited = withRateLimit(new ClaudeStrategy({ apiKey: "k" }), {
      tokensPerSecond: 1000,
      maxConcurrent: 1,
      maxWaitMs: 50,
    });
    const controller = new AbortController();
    mockMessagesStream.mockReturnValueOnce(
      abortingClaudeStream(["a"], controller, new Error("Request was aborted.")),
    );
    await expect(
      limited.generate({ prompt: "p", onToken: () => {}, signal: controller.signal }),
    ).rejects.toBeInstanceOf(LlmAbortedError);

    mockMessagesStream.mockReturnValueOnce(okClaudeStream());
    await expect(limited.generate({ prompt: "p", onToken: () => {} })).resolves.toMatchObject({
      text: "ok",
    });
  });
});
