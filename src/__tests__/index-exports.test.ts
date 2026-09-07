import { describe, expect, it } from "vitest";
import {
  BudgetExceededError,
  KNOWN_CAPABILITY_TAGS,
  NoAvailableProviderError,
  Orchestrator,
  TaskDecompositionError,
  TaskRouter,
  calculateCost,
  compose,
  withBudget,
  withInstrumentation,
} from "../index";

describe("public exports", () => {
  it("exports the task-router and orchestrator surface", () => {
    expect(typeof TaskRouter).toBe("function");
    expect(typeof Orchestrator).toBe("function");
    expect(typeof TaskDecompositionError).toBe("function");
    expect(typeof NoAvailableProviderError).toBe("function");
    expect(KNOWN_CAPABILITY_TAGS).toEqual([
      "vision",
      "code",
      "long-context",
      "cheap",
      "reasoning",
      "multilingual",
    ]);
  });

  it("exports the cost-control surface", () => {
    expect(typeof calculateCost).toBe("function");
    expect(typeof withBudget).toBe("function");
    expect(typeof BudgetExceededError).toBe("function");
  });

  it("exports the instrumentation surface", () => {
    expect(typeof withInstrumentation).toBe("function");
    expect(typeof compose).toBe("function");
  });
});
