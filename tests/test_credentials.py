"""Credential-provider seam.

``api_key`` accepts a zero-argument callable everywhere a string is
accepted.  The callable is invoked at request-build time, once per
request, so rotating credentials (Azure Entra token providers, OAuth
refreshers) stay fresh in long-lived clients.  A static string is the
degenerate constant provider — nothing changes for existing callers.

Credential material must never appear in adapter reprs, whether the
credential is a string or a callable.
"""

from __future__ import annotations

import pytest

from lm15.providers import (
    AnthropicLM,
    AsyncAnthropicLM,
    AsyncGeminiLM,
    AsyncOpenAIChatLM,
    AsyncOpenAILM,
    GeminiLM,
    OpenAIChatLM,
    OpenAILM,
)
from lm15.types import Message, Request

REQUEST = Request(model="m", messages=(Message.user("hi"),))

# (adapter class, header name, header value template)
SYNC_ADAPTERS = [
    (OpenAILM, "Authorization", "Bearer {token}"),
    (OpenAIChatLM, "Authorization", "Bearer {token}"),
    (AnthropicLM, "x-api-key", "{token}"),
    (GeminiLM, "x-goog-api-key", "{token}"),
]

ASYNC_ADAPTERS = [AsyncOpenAILM, AsyncOpenAIChatLM, AsyncAnthropicLM, AsyncGeminiLM]


class CountingCredential:
    """A credential provider that counts calls and can rotate its token."""

    def __init__(self, token: str = "tok-first") -> None:
        self.calls = 0
        self.token = token

    def __call__(self) -> str:
        self.calls += 1
        return self.token


def _header(http, name: str) -> str | None:
    for key, value in http.headers:
        if key.lower() == name.lower():
            return value
    return None


class TestCallableApiKey:
    @pytest.mark.parametrize("cls,header,template", SYNC_ADAPTERS)
    def test_callable_is_resolved_per_request(self, cls, header, template) -> None:
        credential = CountingCredential()
        lm = cls(api_key=credential)

        first = lm.build_request(REQUEST, stream=False)
        assert _header(first, header) == template.format(token="tok-first")

        credential.token = "tok-rotated"
        second = lm.build_request(REQUEST, stream=False)
        assert _header(second, header) == template.format(token="tok-rotated")
        assert credential.calls == 2

    @pytest.mark.parametrize("cls,header,template", SYNC_ADAPTERS)
    def test_static_string_still_works(self, cls, header, template) -> None:
        lm = cls(api_key="sk-static")
        try:
            http = lm.build_request(REQUEST, stream=False)
        finally:
            lm.close()
        assert _header(http, header) == template.format(token="sk-static")

    @pytest.mark.parametrize("cls", ASYNC_ADAPTERS)
    def test_async_mirror_accepts_callable(self, cls) -> None:
        credential = CountingCredential()
        lm = cls(api_key=credential)
        # The inner sync adapter owns all pure mapping, including headers.
        lm._inner.build_request(REQUEST, stream=False)
        credential.token = "tok-rotated"
        lm._inner.build_request(REQUEST, stream=False)
        assert credential.calls == 2


class TestSubscriptionFreshness:
    """Subscription adapters re-resolve the local OAuth credential per
    request: a long-lived client picks up tokens refreshed on disk (by the
    provider CLI, another process, or lm15's own refresh) without being
    rebuilt."""

    def test_claude_code_picks_up_rotated_token(self, tmp_path) -> None:
        from lm15.providers import ClaudeCodeLM
        from .test_subscription_auth import _write_claude_creds

        path = tmp_path / "credentials.json"
        _write_claude_creds(path, access="sk-ant-oat-FIRST")
        lm = ClaudeCodeLM(credentials_path=path)

        first = lm.build_request(REQUEST, stream=False)
        assert _header(first, "Authorization") == "Bearer sk-ant-oat-FIRST"

        _write_claude_creds(path, access="sk-ant-oat-SECOND")
        second = lm.build_request(REQUEST, stream=False)
        assert _header(second, "Authorization") == "Bearer sk-ant-oat-SECOND"

    def test_codex_picks_up_rotated_token(self, tmp_path) -> None:
        from lm15.providers import OpenAICodexLM
        from .test_subscription_auth import _fake_codex_jwt, _write_codex_auth

        path = tmp_path / "auth.json"
        jwt_first = _fake_codex_jwt(exp=int(__import__("time").time()) + 3600)
        _write_codex_auth(path, access=jwt_first)
        lm = OpenAICodexLM(auth_path=path)

        first = lm.build_request(REQUEST, stream=False)
        assert _header(first, "Authorization") == f"Bearer {jwt_first}"

        jwt_second = _fake_codex_jwt(exp=int(__import__("time").time()) + 7200)
        _write_codex_auth(path, access=jwt_second)
        second = lm.build_request(REQUEST, stream=False)
        assert _header(second, "Authorization") == f"Bearer {jwt_second}"

    def test_async_claude_code_mirror_stays_fresh(self, tmp_path) -> None:
        from lm15.providers import AsyncClaudeCodeLM
        from .test_subscription_auth import _write_claude_creds

        path = tmp_path / "credentials.json"
        _write_claude_creds(path, access="sk-ant-oat-FIRST")
        lm = AsyncClaudeCodeLM(credentials_path=path)

        _write_claude_creds(path, access="sk-ant-oat-SECOND")
        http = lm._inner.build_request(REQUEST, stream=False)
        assert _header(http, "Authorization") == "Bearer sk-ant-oat-SECOND"

    def test_explicit_api_key_stays_static(self, tmp_path) -> None:
        from lm15.providers import ClaudeCodeLM

        lm = ClaudeCodeLM(api_key="sk-ant-explicit")
        http = lm.build_request(REQUEST, stream=False)
        assert _header(http, "Authorization") == "Bearer sk-ant-explicit"


class TestCredentialHygiene:
    @pytest.mark.parametrize("cls,_header_,_template_", SYNC_ADAPTERS)
    def test_repr_never_contains_string_credential(self, cls, _header_, _template_) -> None:
        lm = cls(api_key="sk-SUPERSECRET")
        try:
            assert "SUPERSECRET" not in repr(lm)
        finally:
            lm.close()

    @pytest.mark.parametrize("cls", ASYNC_ADAPTERS)
    def test_async_repr_never_contains_string_credential(self, cls) -> None:
        lm = cls(api_key="sk-SUPERSECRET")
        assert "SUPERSECRET" not in repr(lm)
