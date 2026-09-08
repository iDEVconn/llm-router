import { describe, expect, it } from "vitest";
import { BudgetExceededError, NoAvailableProviderError, TaskDecompositionError } from "../errors";

describe("TaskDecompositionError", () => {
  it("wraps an Error cause with its message", () => {
    const err = new TaskDecompositionError(new Error("bad json"));
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("TaskDecompositionError");
    expect(err.message).toBe("Failed to decompose the task into subtasks: bad json");
    expect(err.cause).toBeInstanceOf(Error);
  });

  it("stringifies a non-Error cause", () => {
    const err = new TaskDecompositionError("not json");
    expect(err.message).toBe("Failed to decompose the task into subtasks: not json");
  });
});

describe("NoAvailableProviderError", () => {
  it("names the unroutable subtask", () => {
    const err = new NoAvailableProviderError("subtask-2");
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("NoAvailableProviderError");
    expect(err.subtaskId).toBe("subtask-2");
    expect(err.message).toMatch(/subtask-2/);
  });
});

describe("BudgetExceededError", () => {
  it("reports the per-call limit and the offending cost", () => {
    const err = new BudgetExceededError("perCall", 2.5, 1);
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("BudgetExceededError");
    expect(err.kind).toBe("perCall");
    expect(err.cost).toBe(2.5);
    expect(err.limit).toBe(1);
    expect(err.message).toMatch(/2\.5/);
  });

  it("reports the accumulated total limit", () => {
    const err = new BudgetExceededError("total", 10, 10);
    expect(err.kind).toBe("total");
    expect(err.message).toMatch(/10/);
  });
});

import { InvalidGenerateOptionsError, UnsupportedMultiTurnError } from "../errors";

describe("InvalidGenerateOptionsError", () => {
  it("names the problem when both prompt and messages are set", () => {
    const err = new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but both were provided.",
    );
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("InvalidGenerateOptionsError");
    expect(err.message).toMatch(/both were provided/);
  });

  it("names the problem when neither prompt nor messages are set", () => {
    const err = new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but neither was provided.",
    );
    expect(err.message).toMatch(/neither was provided/);
  });
});

describe("UnsupportedMultiTurnError", () => {
  it("names the offending provider", () => {
    const err = new UnsupportedMultiTurnError("acme-llm");
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe("UnsupportedMultiTurnError");
    expect(err.providerName).toBe("acme-llm");
    expect(err.message).toMatch(/acme-llm/);
  });
});
