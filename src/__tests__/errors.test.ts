import { describe, expect, it } from "vitest";
import { NoAvailableProviderError, TaskDecompositionError } from "../errors";

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
