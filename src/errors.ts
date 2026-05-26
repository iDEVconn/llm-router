/**
 * Plain `Error` subclasses so the pkg stays framework-agnostic.
 * Consumers (NestJS, Express, Fastify, …) wrap these at the controller
 * boundary to map onto the right HTTP status:
 *   - `UnknownProviderError`         → 404
 *   - `NoPlatformProviderError`      → 400
 *   - `InvalidPlatformProviderError` → throw at boot (config error)
 *   - `LlmKeyValidationError`        → 400 / 401 depending on caller intent
 *   - `UnsupportedAttachmentError`   → 400
 */

export class UnknownProviderError extends Error {
  constructor(public readonly providerName: string) {
    super(`Unknown LLM provider: "${providerName}"`);
    this.name = "UnknownProviderError";
  }
}

export class NoPlatformProviderError extends Error {
  constructor() {
    super(
      "No platform LLM provider is configured. Either set the platform name explicitly or have callers supply a BYOK key.",
    );
    this.name = "NoPlatformProviderError";
  }
}

export class InvalidPlatformProviderError extends Error {
  constructor(
    public readonly providerName: string,
    public readonly knownProviders: readonly string[],
  ) {
    super(
      `Invalid platform LLM provider "${providerName}". Expected one of: ${knownProviders.join(", ")}.`,
    );
    this.name = "InvalidPlatformProviderError";
  }
}

export class LlmKeyValidationError extends Error {
  constructor(
    public readonly providerName: string,
    public override readonly cause: unknown,
  ) {
    const causeMessage = cause instanceof Error ? cause.message : String(cause);
    super(`${providerName} rejected the API key: ${causeMessage}`);
    this.name = "LlmKeyValidationError";
  }
}

export class UnsupportedAttachmentError extends Error {
  constructor(
    public readonly providerName: string,
    public readonly mimetype: string,
    public readonly hint?: string,
  ) {
    super(
      `${providerName} does not accept "${mimetype}".${hint ? ` ${hint}` : ""}`,
    );
    this.name = "UnsupportedAttachmentError";
  }
}
