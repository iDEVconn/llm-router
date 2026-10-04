import type { LlmGenerateOptions, LlmUsage } from "./types";

/**
 * Deliberately rough token estimate: `Math.ceil(chars / 4)`, counting
 * UTF-16 code units (`string.length`). No tokenizer, no language-specific
 * logic — it under/over-counts for non-Latin scripts, code, and so on.
 * Only used to fill in usage the provider never reported (e.g. a streaming
 * call aborted before the final usage chunk arrived); callers see
 * `usageEstimated: true` whenever any side came from here.
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Rough input-side estimate for a call: `systemPrompt` plus `prompt`, or
 * plus every `messages[].content` for a multi-turn call. Attachments are
 * not counted.
 */
export function estimateInputTokens(
  opts: Pick<LlmGenerateOptions, "prompt" | "messages" | "systemPrompt">,
): number {
  let chars = (opts.systemPrompt ?? "").length + (opts.prompt ?? "").length;
  for (const message of opts.messages ?? []) chars += message.content.length;
  return Math.ceil(chars / 4);
}

/** Usage counts the provider actually reported before the stream ended; absent side = not reported. */
export interface ReportedUsage {
  inputTokens?: number | null;
  outputTokens?: number | null;
}

/**
 * Combines whatever usage the provider reported mid-stream with rough
 * estimates for the missing side(s). `usageEstimated` is false only when
 * BOTH sides were provider-reported.
 */
export function resolveAbortUsage(
  reported: ReportedUsage,
  opts: Pick<LlmGenerateOptions, "prompt" | "messages" | "systemPrompt">,
  partialText: string,
  partialThinking?: string,
): { usage: LlmUsage; usageEstimated: boolean } {
  const hasInput = typeof reported.inputTokens === "number";
  const hasOutput = typeof reported.outputTokens === "number";
  return {
    usage: {
      inputTokens: hasInput ? (reported.inputTokens as number) : estimateInputTokens(opts),
      outputTokens: hasOutput
        ? (reported.outputTokens as number)
        : estimateTokens(partialText + (partialThinking ?? "")),
    },
    usageEstimated: !(hasInput && hasOutput),
  };
}
