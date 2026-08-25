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
    generate?: LlmStrategy["generate"];
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

  it("passes the meta provider's apiKey from opts.apiKeys to the LLM-fallback generate() call", async () => {
    const generate = vi.fn().mockResolvedValue(fallbackResponse('{"provider": "p"}'));
    const p = makeStrategy("p", { hasPlatformKey: () => false, generate });
    const q = makeStrategy("q", { hasPlatformKey: () => true });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "q" });
    const router = new TaskRouter({ registry, metaProvider: "p" });

    await router.route([{ id: "t1", description: "x" }], { apiKeys: { p: "user-key" } });

    expect(generate.mock.calls[0]![0]).toMatchObject({ apiKey: "user-key" });
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
