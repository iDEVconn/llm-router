import { describe, expect, it, vi } from "vitest";
import { compose, withInstrumentation } from "../instrumentation";
import type { LlmStrategy } from "../types";

function makeStrategy(): LlmStrategy {
  return {
    providerName: "gemini",
    defaultModel: "gemini-2.5-flash",
    generate: vi.fn().mockResolvedValue({
      text: "hi",
      model: "gemini-2.5-flash",
      usage: { inputTokens: 10, outputTokens: 20 },
      truncated: true,
    }),
    validateKey: vi.fn(),
  };
}

function makeFailingStrategy(error: Error): LlmStrategy {
  return {
    providerName: "gemini",
    defaultModel: "gemini-2.5-flash",
    generate: vi.fn().mockRejectedValue(error),
    validateKey: vi.fn(),
  };
}

describe("withInstrumentation", () => {
  it("emits an onCall event with provider, model, usage and truncated on success", async () => {
    const strategy = makeStrategy();
    const onCall = vi.fn();
    const wrapped = withInstrumentation(strategy, { onCall });

    await wrapped.generate({ prompt: "hi" });

    expect(onCall).toHaveBeenCalledTimes(1);
    const event = onCall.mock.calls[0]![0];
    expect(event.provider).toBe("gemini");
    expect(event.model).toBe("gemini-2.5-flash");
    expect(event.usage).toEqual({ inputTokens: 10, outputTokens: 20 });
    expect(event.truncated).toBe(true);
    expect(typeof event.latencyMs).toBe("number");
    expect(typeof event.timestamp).toBe("string");
    expect(event.error).toBeUndefined();
  });

  it("emits an onCall event with the error message and no usage on failure", async () => {
    const strategy = makeFailingStrategy(new Error("boom"));
    const onCall = vi.fn();
    const wrapped = withInstrumentation(strategy, { onCall });

    await expect(wrapped.generate({ prompt: "hi" })).rejects.toThrow("boom");

    expect(onCall).toHaveBeenCalledTimes(1);
    const event = onCall.mock.calls[0]![0];
    expect(event.provider).toBe("gemini");
    expect(event.error).toBe("boom");
    expect(typeof event.latencyMs).toBe("number");
  });

  it("forwards onToken, thinking, and signal to the wrapped strategy unmodified", async () => {
    const strategy = makeStrategy();
    const wrapped = withInstrumentation(strategy, { onCall: vi.fn() });
    const onToken = vi.fn();
    const controller = new AbortController();
    const thinking = { type: "adaptive" as const };

    await wrapped.generate({ prompt: "hi", onToken, thinking, signal: controller.signal });

    const passedOpts = (strategy.generate as ReturnType<typeof vi.fn>).mock.calls[0]![0];
    expect(passedOpts.onToken).toBe(onToken);
    expect(passedOpts.thinking).toBe(thinking);
    expect(passedOpts.signal).toBe(controller.signal);
  });
});

describe("compose", () => {
  it("applies decorators so the resulting strategy behaves as if manually wrapped", async () => {
    const strategy = makeStrategy();
    const calls: string[] = [];
    const decoratorA = (s: LlmStrategy): LlmStrategy => ({
      ...s,
      generate: async (opts) => {
        calls.push("a");
        return s.generate(opts);
      },
    });
    const decoratorB = (s: LlmStrategy): LlmStrategy => ({
      ...s,
      generate: async (opts) => {
        calls.push("b");
        return s.generate(opts);
      },
    });

    const composed = compose(strategy, decoratorA, decoratorB);
    await composed.generate({ prompt: "hi" });

    // compose(s, a, b) === b(a(s)) — b wraps a's result, so b's logic runs
    // first at call time, then delegates inward to a, then to s.
    expect(calls).toEqual(["b", "a"]);
  });
});
