import { defineConfig } from "tsup";

export default defineConfig({
  entry: {
    index: "src/index.ts",
    gemini: "src/gemini/index.ts",
    claude: "src/claude/index.ts",
    grok: "src/grok/index.ts",
    chatgpt: "src/chatgpt/index.ts",
    deepseek: "src/deepseek/index.ts",
  },
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: true,
  clean: true,
  target: "es2022",
  external: ["@google/generative-ai", "@google/genai", "@anthropic-ai/sdk", "openai"],
});
