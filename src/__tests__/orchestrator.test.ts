import { describe, expect, it, vi } from "vitest";
import { LlmRegistry } from "../registry";
import { Orchestrator } from "../orchestrator";
import type { LlmResponse, LlmStrategy } from "../types";

async function flushUntil(predicate: () => boolean, maxTicks = 50): Promise<void> {
  for (let i = 0; i < maxTicks; i++) {
    if (predicate()) return;
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  throw new Error(`flushUntil: condition not met after ${maxTicks} ticks`);
}

function ok(text: string, finish: "stop" | "length" = "stop"): LlmResponse {
  return {
    text,
    model: "m",
    usage: { inputTokens: 1, outputTokens: 1 },
    truncated: finish === "length",
  };
}

function makeStrategy(
  name: string,
  opts: {
    capabilities?: readonly string[];
    hasPlatformKey?: () => boolean;
    generate: LlmStrategy["generate"];
  },
): LlmStrategy {
  return {
    providerName: name,
    defaultModel: `${name}-default`,
    capabilities: opts.capabilities,
    generate: opts.generate,
    validateKey: vi.fn(),
    hasPlatformKey: opts.hasPlatformKey ?? (() => true),
  };
}

const ONE_SUBTASK_JSON = '[{"id": "t1", "description": "do the thing"}]';

describe("Orchestrator.run — decompose", () => {
  it("parses a valid JSON subtask array and executes it", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON)) // decompose
      .mockResolvedValueOnce(ok("done")) // subtask generate
      .mockResolvedValueOnce(ok('{"approved": true}')); // self-critique
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("build a widget");

    expect(result.subtasks).toHaveLength(1);
    expect(result.subtasks[0]).toMatchObject({ result: "done", rounds: 0, unresolved: false });
  });

  it("strips a fenced code block before parsing", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(`\`\`\`json\n${ONE_SUBTASK_JSON}\n\`\`\``))
      .mockResolvedValueOnce(ok("done"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("build a widget");
    expect(result.subtasks).toHaveLength(1);
  });

  it("retries decompose once on invalid JSON, then throws TaskDecompositionError", async () => {
    const generate = vi.fn().mockResolvedValue(ok("not json at all"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    await expect(orchestrator.run("build a widget")).rejects.toThrow(
      /Failed to decompose the task/,
    );
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("truncates subtasks beyond maxSubtasks and reports the count", async () => {
    const threeSubtasks = JSON.stringify([
      { id: "t1", description: "a" },
      { id: "t2", description: "b" },
      { id: "t3", description: "c" },
    ]);
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(threeSubtasks))
      .mockResolvedValue(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("build many things", { maxSubtasks: 2 });
    expect(result.subtasks).toHaveLength(2);
    expect(result.truncatedSubtaskCount).toBe(1);
  });

  it("treats a rejected decompose generate() call as unparseable, retries, then throws TaskDecompositionError", async () => {
    const generate = vi
      .fn()
      .mockRejectedValueOnce(new Error("network down"))
      .mockRejectedValueOnce(new Error("still down"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    await expect(orchestrator.run("build a widget")).rejects.toThrow(
      /Failed to decompose the task/,
    );
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("retries decompose once when parsed items have the wrong shape, then throws TaskDecompositionError", async () => {
    const generate = vi.fn().mockResolvedValue(ok('[{"task": "not the right shape"}]'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    await expect(orchestrator.run("build a widget")).rejects.toThrow(
      /Failed to decompose the task/,
    );
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("passes the meta provider's apiKey from opts.apiKeys to the decompose generate() call", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("done"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    await orchestrator.run("build a widget", { apiKeys: { p: "secret-key" } });

    expect(generate.mock.calls[0]![0]).toMatchObject({ apiKey: "secret-key" });
  });
});

describe("Orchestrator.run — critique loop", () => {
  it("returns unresolved:false when approved on the first critique", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("first answer"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");
    expect(result.subtasks[0]).toMatchObject({ result: "first answer", rounds: 0, unresolved: false });
  });

  it("retries once and resolves within a maxRounds=2 budget", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("draft"))
      .mockResolvedValueOnce(ok('{"approved": false, "feedback": "too short"}'))
      .mockResolvedValueOnce(ok("revised"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { maxRounds: 2 });
    expect(result.subtasks[0]).toMatchObject({ result: "revised", rounds: 1, unresolved: false });
  });

  it("marks unresolved:true when the critique loop exhausts maxRounds", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("draft"))
      .mockResolvedValueOnce(ok('{"approved": false, "feedback": "nope"}'))
      .mockResolvedValueOnce(ok("revised once"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { maxRounds: 1 });
    expect(result.subtasks[0]).toMatchObject({ result: "revised once", rounds: 1, unresolved: true });
  });

  it("skips the critique step entirely when maxRounds is 0", async () => {
    const generate = vi.fn().mockResolvedValueOnce(ok(ONE_SUBTASK_JSON)).mockResolvedValueOnce(ok("done"));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { maxRounds: 0 });
    expect(result.subtasks[0]).toMatchObject({ result: "done", rounds: 0, unresolved: false });
    expect(generate).toHaveBeenCalledTimes(2); // decompose + one generate, no critique call
  });

  it("uses the next-highest-scoring available provider for cross critique on a rule-routed subtask", async () => {
    const subtaskJson =
      '[{"id": "t1", "description": "x", "requiredCapabilities": ["code", "reasoning"]}]';
    const bestGenerate = vi
      .fn()
      .mockResolvedValueOnce(ok(subtaskJson)) // decompose (via platform = best)
      .mockResolvedValueOnce(ok("answer")); // subtask generate (routed to best)
    const criticGenerate = vi.fn().mockResolvedValueOnce(ok('{"approved": true}'));
    const best = makeStrategy("best", { capabilities: ["code", "reasoning"], generate: bestGenerate });
    const second = makeStrategy("second", { capabilities: ["code"], generate: criticGenerate });
    const registry = new LlmRegistry({ strategies: [best, second], platform: "best" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { critique: "cross" });

    expect(criticGenerate).toHaveBeenCalledOnce();
    expect(result.subtasks[0]).toMatchObject({ unresolved: false });
  });

  it("falls back to self-critique when no second available provider exists", async () => {
    const generate = vi
      .fn()
      .mockResolvedValueOnce(ok(ONE_SUBTASK_JSON))
      .mockResolvedValueOnce(ok("answer"))
      .mockResolvedValueOnce(ok('{"approved": true}'));
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task", { critique: "cross" });
    expect(result.subtasks[0]).toMatchObject({ unresolved: false });
    expect(generate).toHaveBeenCalledTimes(3); // decompose, generate, self-critique (only one provider exists)
  });
});

describe("Orchestrator.run — partial failure isolation", () => {
  it("isolates one subtask's routing failure without affecting others", async () => {
    // Two AVAILABLE providers so the single-provider shortcut doesn't apply —
    // t1 uniquely tag-matches "p"; t2 has no tags, so it goes through
    // LLM-fallback and gets an invalid answer. Both routing calls go through
    // the platform provider "p", so its generate() must handle three
    // different prompt shapes (decompose, t2's routing fallback, t1's
    // execution) plus t1's critique — matched by content, not call order,
    // since t1 and t2 execute concurrently (default maxConcurrency).
    const twoSubtasks = JSON.stringify([
      { id: "t1", description: "codey thing", requiredCapabilities: ["code"] },
      { id: "t2", description: "generic thing" },
    ]);
    const generate = vi.fn().mockImplementation(({ prompt }: { prompt: string }) => {
      if (prompt.includes("Split the following task")) return Promise.resolve(ok(twoSubtasks));
      if (prompt.includes("Pick the best provider")) return Promise.resolve(ok('{"provider": "ghost"}'));
      if (prompt.includes("Does this fully satisfy")) return Promise.resolve(ok('{"approved": true}'));
      return Promise.resolve(ok("t1 done"));
    });
    const p = makeStrategy("p", { capabilities: ["code"], generate });
    const q = makeStrategy("q", { capabilities: ["vision"], generate: vi.fn().mockResolvedValue(ok("unused")) });
    const registry = new LlmRegistry({ strategies: [p, q], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");

    const t1 = result.subtasks.find((s) => s.subtask.id === "t1")!;
    const t2 = result.subtasks.find((s) => s.subtask.id === "t2")!;
    expect(t1).toMatchObject({ result: "t1 done", unresolved: false });
    expect(t2.error).toMatch(/No available provider/);
    expect(t2.decision).toBeUndefined();
    expect(t2.unresolved).toBe(true);
  });

  it("isolates one subtask's generate() failure without affecting others", async () => {
    // Single provider — the shortcut routes both subtasks to it directly
    // (no fallback call), so the only prompts "p".generate() ever sees are
    // decompose, t1's execution (rejected), t2's execution, and t2's
    // critique. Matched by content since t1/t2 run concurrently.
    const twoSubtasks = JSON.stringify([
      { id: "t1", description: "task one" },
      { id: "t2", description: "task two" },
    ]);
    const generate = vi.fn().mockImplementation(({ prompt }: { prompt: string }) => {
      if (prompt.includes("Split the following task")) return Promise.resolve(ok(twoSubtasks));
      if (prompt === "task one") return Promise.reject(new Error("network down"));
      if (prompt === "task two") return Promise.resolve(ok("t2 done"));
      return Promise.resolve(ok('{"approved": true}')); // t2's self-critique
    });
    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const result = await orchestrator.run("task");

    const t1 = result.subtasks.find((s) => s.subtask.id === "t1")!;
    const t2 = result.subtasks.find((s) => s.subtask.id === "t2")!;
    expect(t1.error).toBe("network down");
    expect(t1.unresolved).toBe(true);
    expect(t2).toMatchObject({ result: "t2 done", unresolved: false });
  });
});

describe("Orchestrator.run — concurrency", () => {
  it("never runs more than maxConcurrency subtasks at once", async () => {
    // Single registered provider -> TaskRouter's single-available-provider
    // shortcut routes every subtask directly, with zero extra generate()
    // calls for routing. Combined with maxRounds: 0 (no critique calls),
    // the only generate() calls are: one decompose call, then exactly one
    // execution call per subtask -- which is what this test measures.
    let inFlight = 0;
    let maxInFlight = 0;
    const releases: Array<() => void> = [];

    const fourSubtasks = JSON.stringify([
      { id: "t1", description: "a" },
      { id: "t2", description: "b" },
      { id: "t3", description: "c" },
      { id: "t4", description: "d" },
    ]);

    const generate = vi.fn().mockImplementation(() => {
      inFlight++;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise<LlmResponse>((resolve) => {
        releases.push(() => {
          inFlight--;
          resolve(ok("done"));
        });
      });
    });
    generate.mockImplementationOnce(() => Promise.resolve(ok(fourSubtasks)));

    const p = makeStrategy("p", { generate });
    const registry = new LlmRegistry({ strategies: [p], platform: "p" });
    const orchestrator = new Orchestrator({ registry });

    const runPromise = orchestrator.run("task", { maxConcurrency: 2, maxRounds: 0 });

    // Let the decompose call resolve and the first wave of subtask
    // generate() calls start (each hangs until its release() fires). Poll
    // on the actual condition rather than counting microtask flushes, since
    // the number of `await` hops between run() and this mock is an
    // implementation detail of the call chain (decompose ->
    // listAvailableProviders -> runWithConcurrency -> routeAndExecute ->
    // route/routeOne -> executeSubtask -> generateForSubtask -> generate)
    // that can silently grow or shrink under unrelated future refactors.
    await flushUntil(() => inFlight === 2);
    expect(inFlight).toBe(2);

    releases.splice(0).forEach((release) => release());
    await flushUntil(() => inFlight === 2);
    expect(inFlight).toBe(2);

    releases.splice(0).forEach((release) => release());
    await runPromise;

    expect(maxInFlight).toBe(2);
  });
});
