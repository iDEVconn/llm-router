import { BudgetExceededError } from "./errors";
import { calculateCost, type PricingTable } from "./pricing";
import type { LlmGenerateOptions, LlmResponse, LlmStrategy, LlmUsage } from "./types";

export interface WithBudgetOptions {
  pricing: PricingTable;
  maxCostPerCall?: number;
  maxCostTotal?: number;
  onCost?: (event: { provider: string; model: string; cost: number; usage: LlmUsage }) => void;
}

/**
 * Wraps a strategy so cost is tracked across calls. Cost can only be known
 * after a call returns (token counts come from the response), so
 * `maxCostPerCall` is enforced post-hoc on the call that just finished,
 * while `maxCostTotal` is enforced up front against the accumulated total
 * from prior calls, blocking the next call before it starts.
 */
export function withBudget(strategy: LlmStrategy, opts: WithBudgetOptions): LlmStrategy {
  let totalCost = 0;

  return {
    ...strategy,
    async generate(genOpts: LlmGenerateOptions): Promise<LlmResponse> {
      if (opts.maxCostTotal !== undefined && totalCost >= opts.maxCostTotal) {
        throw new BudgetExceededError("total", totalCost, opts.maxCostTotal);
      }

      const response = await strategy.generate(genOpts);
      const cost = calculateCost(response.usage, strategy.providerName, response.model, opts.pricing);

      if (opts.maxCostPerCall !== undefined && cost > opts.maxCostPerCall) {
        throw new BudgetExceededError("perCall", cost, opts.maxCostPerCall);
      }

      totalCost += cost;
      opts.onCost?.({
        provider: strategy.providerName,
        model: response.model,
        cost,
        usage: response.usage,
      });
      return response;
    },
  };
}
