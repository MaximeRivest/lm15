from __future__ import annotations


class ULMError(Exception):
    pass


class TransportError(ULMError):
    pass


class ProviderError(ULMError):
    pass


class AuthError(ProviderError):
    pass


class RateLimitError(ProviderError):
    pass


class TimeoutError(ProviderError):
    pass


class InvalidRequestError(ProviderError):
    pass


class ServerError(ProviderError):
    pass


class UnsupportedModelError(ProviderError):
    pass


class UnsupportedFeatureError(ProviderError):
    pass


class NotConfiguredError(ProviderError):
    pass


def map_http_error(status: int, message: str) -> ProviderError:
    if status in (401, 403):
        return AuthError(message)
    if status == 408:
        return TimeoutError(message)
    if status == 429:
        return RateLimitError(message)
    if status in (400, 404, 409, 422):
        return InvalidRequestError(message)
    if 500 <= status <= 599:
        return ServerError(message)
    return ProviderError(message)
