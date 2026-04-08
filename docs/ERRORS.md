# Errors

## Normalized Error Classes

- `AuthError`
- `RateLimitError`
- `TimeoutError`
- `InvalidRequestError`
- `ServerError`
- `UnsupportedFeatureError`
- `UnsupportedModelError`
- `NotConfiguredError`
- `ProviderError` (fallback)

## Mapping

`map_http_error(status, body)` maps HTTP status to normalized class.

Adapters may override `normalize_error()` for provider-specific body parsing while preserving normalized class hierarchy.
