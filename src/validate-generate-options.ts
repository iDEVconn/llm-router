import { InvalidGenerateOptionsError } from "./errors";
import type { LlmGenerateOptions } from "./types";

/**
 * Enforces that a `generate()` call supplies exactly one content source.
 * Called first in every strategy's `generate()`, before any SDK client
 * or network call, so a malformed call fails fast and cheaply.
 */
export function assertExactlyOnePromptSource(
  opts: Pick<LlmGenerateOptions, "prompt" | "messages">,
): void {
  const hasPrompt = opts.prompt !== undefined;
  const hasMessages = opts.messages !== undefined;

  if (hasPrompt && hasMessages) {
    throw new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but both were provided.",
    );
  }
  if (!hasPrompt && !hasMessages) {
    throw new InvalidGenerateOptionsError(
      "Exactly one of `prompt` or `messages` must be set, but neither was provided.",
    );
  }
  if (hasMessages && opts.messages!.length === 0) {
    throw new InvalidGenerateOptionsError(
      "`messages` must contain at least one turn.",
    );
  }
}
