import { describe, expect, it, vi } from "vitest";
import {
  InvalidPlatformProviderError,
  LlmRegistry,
  NoPlatformProviderError,
  UnknownProviderError,
  type LlmStrategy,
} from "../index";

function makeStrategy(name: string): LlmStrategy {
  return {
    providerName: name,
    defaultModel: `${name}-default`,
    generate: vi.fn(),
    validateKey: vi.fn(),
  };
}

describe("LlmRegistry", () => {
  it("looks up strategies by providerName", () => {
    const gemini = makeStrategy("gemini");
    const claude = makeStrategy("claude");
    const registry = new LlmRegistry<"gemini" | "claude">({
      strategies: [gemini, claude],
      platform: "gemini",
    });
    expect(registry.get("gemini")).toBe(gemini);
    expect(registry.get("claude")).toBe(claude);
  });

  it("throws UnknownProviderError for unknown names", () => {
    const registry = new LlmRegistry({
      strategies: [makeStrategy("gemini")],
      platform: "gemini",
    });
    expect(() => registry.get("nope")).toThrow(UnknownProviderError);
  });

  it("getPlatform returns the platform strategy when configured", () => {
    const gemini = makeStrategy("gemini");
    const registry = new LlmRegistry({
      strategies: [gemini],
      platform: "gemini",
    });
    expect(registry.getPlatform()).toBe(gemini);
    expect(registry.getPlatformProviderName()).toBe("gemini");
  });

  it("getPlatform throws NoPlatformProviderError in BYOK-only mode", () => {
    const registry = new LlmRegistry({
      strategies: [makeStrategy("gemini")],
      platform: null,
    });
    expect(() => registry.getPlatform()).toThrow(NoPlatformProviderError);
    expect(registry.getPlatformProviderName()).toBeNull();
  });

  it("rejects an unknown platform name at construction time", () => {
    expect(
      () =>
        new LlmRegistry({
          strategies: [makeStrategy("gemini")],
          platform: "missing" as never,
        }),
    ).toThrow(InvalidPlatformProviderError);
  });

  it("rejects duplicate providerName values", () => {
    expect(
      () =>
        new LlmRegistry({
          strategies: [makeStrategy("gemini"), makeStrategy("gemini")],
        }),
    ).toThrow(/Duplicate strategy providerName/);
  });

  it("isProviderName narrows raw strings to the union", () => {
    const registry = new LlmRegistry<"gemini" | "claude">({
      strategies: [makeStrategy("gemini"), makeStrategy("claude")],
    });
    expect(registry.isProviderName("gemini")).toBe(true);
    expect(registry.isProviderName("anthropic")).toBe(false);
  });

  it("listProviderNames returns every registered name", () => {
    const registry = new LlmRegistry({
      strategies: [makeStrategy("a"), makeStrategy("b")],
    });
    expect([...registry.listProviderNames()].sort()).toEqual(["a", "b"]);
  });

  it("warns when the active platform's env key is missing", () => {
    const warn = vi.fn();
    new LlmRegistry<"gemini">({
      strategies: [makeStrategy("gemini")],
      platform: "gemini",
      providerEnvKeys: { gemini: "GEMINI_API_KEY" },
      env: {},
      logger: { warn },
    });
    expect(warn).toHaveBeenCalled();
    const msg = warn.mock.calls.flat().join("\n");
    expect(msg).toMatch(/active platform/i);
    expect(msg).toMatch(/GEMINI_API_KEY/);
  });

  it("warns about inactive provider keys but with BYOK-aware copy", () => {
    const warn = vi.fn();
    new LlmRegistry<"gemini" | "claude">({
      strategies: [makeStrategy("gemini"), makeStrategy("claude")],
      platform: "gemini",
      providerEnvKeys: { gemini: "GEMINI_API_KEY", claude: "CLAUDE_API_KEY" },
      env: { GEMINI_API_KEY: "set", CLAUDE_API_KEY: "" },
      logger: { warn },
    });
    const msg = warn.mock.calls.flat().join("\n");
    expect(msg).toMatch(/CLAUDE_API_KEY/);
    expect(msg).toMatch(/BYOK/);
    expect(msg).not.toMatch(/GEMINI_API_KEY is not set/);
  });

  it("warns once on BYOK-only mode", () => {
    const warn = vi.fn();
    new LlmRegistry({
      strategies: [makeStrategy("gemini")],
      platform: null,
      logger: { warn },
    });
    expect(warn).toHaveBeenCalled();
    expect(warn.mock.calls.some((c) => /BYOK-only mode/.test(c.join(" ")))).toBe(true);
  });
});
