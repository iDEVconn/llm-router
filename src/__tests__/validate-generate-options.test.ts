import { describe, expect, it } from "vitest";
import { InvalidGenerateOptionsError } from "../errors";
import { assertExactlyOnePromptSource } from "../validate-generate-options";

describe("assertExactlyOnePromptSource", () => {
  it("passes when only prompt is set", () => {
    expect(() => assertExactlyOnePromptSource({ prompt: "hi" })).not.toThrow();
  });

  it("passes when only messages is set", () => {
    expect(() =>
      assertExactlyOnePromptSource({ messages: [{ role: "user", content: "hi" }] }),
    ).not.toThrow();
  });

  it("throws InvalidGenerateOptionsError when both prompt and messages are set", () => {
    expect(() =>
      assertExactlyOnePromptSource({
        prompt: "hi",
        messages: [{ role: "user", content: "hi" }],
      }),
    ).toThrow(InvalidGenerateOptionsError);
  });

  it("throws InvalidGenerateOptionsError when neither prompt nor messages are set", () => {
    expect(() => assertExactlyOnePromptSource({})).toThrow(InvalidGenerateOptionsError);
  });
});
