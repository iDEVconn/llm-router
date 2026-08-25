---
"@idevconn/llm-router": minor
---

Add `TaskRouter` and `Orchestrator` for capability-based multi-provider task routing, plus `ChatGptStrategy` and `DeepSeekStrategy` adapters. `LlmStrategy` gains two optional members, `hasPlatformKey?()` and `capabilities?`, which existing custom strategies can ignore without breaking.
