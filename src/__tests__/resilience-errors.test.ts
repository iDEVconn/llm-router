import { describe, expect, it } from "vitest";
import {
  BudgetExceededError,
  InvalidGenerateOptionsError,
  InvalidThinkingConfigError,
  UnsupportedAttachmentError,
  UnsupportedMultiTurnError,
  UnsupportedThinkingModeError,
} from "../errors";
import { isAbortError, isCallerFaultError } from "../resilience-errors";

describe("isCallerFaultError", () => {
  it("returns true for each caller-fault error type", () => {
    expect(isCallerFaultError(new InvalidGenerateOptionsError("x"))).toBe(true);
    expect(isCallerFaultError(new InvalidThinkingConfigError("x"))).toBe(true);
    expect(isCallerFaultError(new UnsupportedAttachmentError("p", "image/x"))).toBe(true);
    expect(isCallerFaultError(new UnsupportedThinkingModeError("p", "adaptive", []))).toBe(true);
    expect(isCallerFaultError(new UnsupportedMultiTurnError("p"))).toBe(true);
    expect(isCallerFaultError(new BudgetExceededError("perCall", 5, 1))).toBe(true);
  });

  it("returns false for a generic error", () => {
    expect(isCallerFaultError(new Error("network blip"))).toBe(false);
  });

  it("returns false for non-Error values", () => {
    expect(isCallerFaultError("just a string")).toBe(false);
    expect(isCallerFaultError(undefined)).toBe(false);
  });
});

describe("isAbortError", () => {
  it("returns false when the signal is not aborted", () => {
    const controller = new AbortController();
    expect(isAbortError(new Error("x"), controller.signal)).toBe(false);
  });

  it("returns false when signal is undefined", () => {
    expect(isAbortError(new Error("x"), undefined)).toBe(false);
  });

  it("returns true when the thrown error is the signal's abort reason", () => {
    const controller = new AbortController();
    const reason = new Error("aborted by caller");
    controller.abort(reason);
    expect(isAbortError(reason, controller.signal)).toBe(true);
  });

  it("returns true when the thrown error is a DOMException-style AbortError, even if not the exact reason object", () => {
    const controller = new AbortController();
    controller.abort();
    const abortErr = new Error("The operation was aborted");
    abortErr.name = "AbortError";
    expect(isAbortError(abortErr, controller.signal)).toBe(true);
  });

  it("returns false when the signal is aborted but the thrown error is unrelated", () => {
    const controller = new AbortController();
    controller.abort(new Error("aborted"));
    expect(isAbortError(new Error("unrelated failure"), controller.signal)).toBe(false);
  });
});
