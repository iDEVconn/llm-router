# @idevconn/llm-router

## 0.2.0

### Minor Changes

- d736bdc: Initial release.

  Library-agnostic LLM router. Main entry ships pure types + `LlmRegistry`
  with env-driven platform selection, BYOK support, and boot-time env-key
  audit — zero SDK dependencies. Concrete adapters for Gemini, Claude,
  and Grok live behind subpath exports (`@idevconn/llm-router/gemini`,
  `/claude`, `/grok`) with their SDKs declared as **optional** peer
  dependencies, so consumers install only what they actually use.

  Errors are plain `Error` subclasses (`UnknownProviderError`,
  `NoPlatformProviderError`, `InvalidPlatformProviderError`,
  `LlmKeyValidationError`, `UnsupportedAttachmentError`) — callers map
  them to framework-specific exceptions at their controller boundary.
