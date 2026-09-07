import type { LlmUsage } from "./types";

/**
 * Prices are not baked in as a constant — they drift independently of
 * this package's release cycle, so the caller owns and passes the table.
 */
export type PricingTable = Record<
  string /* provider */,
  Record<string /* model */, { inputPer1M: number; outputPer1M: number }>
>;

export function calculateCost(
  usage: LlmUsage,
  provider: string,
  model: string,
  pricing: PricingTable,
): number {
  const rates = pricing[provider]?.[model];
  if (!rates) {
    throw new Error(`No pricing entry for provider "${provider}", model "${model}".`);
  }
  return (
    (usage.inputTokens / 1_000_000) * rates.inputPer1M +
    (usage.outputTokens / 1_000_000) * rates.outputPer1M
  );
}
