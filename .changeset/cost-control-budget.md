---
"@idevconn/llm-router": minor
---

Add cost control: `calculateCost(usage, provider, model, pricing)` in `src/pricing.ts` computes cost from a caller-supplied `PricingTable` (prices are not baked in, since they drift independently of this package), and `withBudget(strategy, opts)` in `src/budget.ts` wraps any `LlmStrategy` to track spend across calls, throwing the new `BudgetExceededError` when `maxCostPerCall` is exceeded by a call's actual cost, or when `maxCostTotal` has already been reached before the next call starts. `opts.onCost` fires with `{ provider, model, cost, usage }` after each successful call.
