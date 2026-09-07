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
  Retriever,
  detectPromptInjection,
  detectPromptInjectionWithModel,
  sanitizeUntrustedContent,
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

  it("exports the prompt-injection defense surface", () => {
    expect(typeof sanitizeUntrustedContent).toBe("function");
    expect(typeof detectPromptInjection).toBe("function");
    expect(typeof detectPromptInjectionWithModel).toBe("function");
  });

  it("exports the RAG surface", () => {
    expect(typeof Retriever).toBe("function");
  });
});
