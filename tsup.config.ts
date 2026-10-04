import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    gemini: "src/gemini/index.ts",
    claude: "src/claude/index.ts",
    grok: "src/grok/index.ts",
    chatgpt: "src/chatgpt/index.ts",
    deepseek: "src/deepseek/index.ts",
    "embeddings/pgvector": "src/embeddings/pgvector/index.ts",
  },
  format: ["esm", "cjs"],
  // tsup injects a deprecated `baseUrl` into its DTS pass; TypeScript 6 rejects it without this.
  dts: { compilerOptions: { ignoreDeprecations: "6.0" } },
  sourcemap: true,
  clean: true,
  target: "es2022",
  external: ["@google/generative-ai", "@google/genai", "@anthropic-ai/sdk", "openai", "pg"],
});
