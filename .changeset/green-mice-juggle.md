---
"@idevconn/llm-router": minor
---

`GeminiStrategy` accepts `connection: "vertex"` to run platform-funded calls through Vertex AI with Application Default Credentials, using `@google/genai`. Per-call BYOK still routes through the direct Gemini API. No automatic failover.