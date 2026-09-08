---
"@idevconn/llm-router": patch
---

Fix GeminiStrategy.generate() dropping opts.maxTokens — now forwarded as generationConfig.maxOutputTokens (direct API) and config.maxOutputTokens (Vertex).
