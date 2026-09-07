"""MAP-9 on the complete path (changes/2026-09-07-complete-tool-call-no-guess.md):
a non-streaming reply whose tool call has no name is refused with
ProviderError; lm15 never substitutes a name."""

import json

import pytest

from lm15 import Message, ProviderError, Request
from lm15.providers.base import HttpResponse
from lm15.vet import adapter_for_provider


def _lm(provider: str):
    return adapter_for_provider(provider, "test-key")


def _http(body: dict) -> HttpResponse:
    return HttpResponse(status=200, reason="OK", headers=[], body=json.dumps(body).encode())


REQUEST = Request(model="m", messages=[Message.user("hi")])

BODIES = {
    "openai": {"output": [{"type": "function_call", "call_id": "c", "arguments": "{}"}]},
    "openai_chat": {"choices": [{"message": {"tool_calls": [{"id": "c", "function": {"arguments": "{}"}}]}}]},
    "anthropic": {"content": [{"type": "tool_use", "id": "t", "input": {}}]},
    "gemini": {"candidates": [{"content": {"parts": [{"functionCall": {"args": {}}}]}}]},
}


@pytest.mark.parametrize("provider", sorted(BODIES))
def test_nameless_tool_call_is_refused(provider: str) -> None:
    with pytest.raises(ProviderError) as info:
        _lm(provider).parse_response(REQUEST, _http(BODIES[provider]))
    assert info.value.code == "provider"
    assert "no name" in str(info.value)


def test_empty_name_is_no_name() -> None:
    body = {"content": [{"type": "tool_use", "id": "t", "name": "", "input": {}}]}
    with pytest.raises(ProviderError):
        _lm("anthropic").parse_response(REQUEST, _http(body))


def test_named_calls_still_parse() -> None:
    body = {"content": [{"type": "tool_use", "id": "t", "name": "get_weather", "input": {"city": "x"}}]}
    response = _lm("anthropic").parse_response(REQUEST, _http(body))
    assert response.tool_calls[0].name == "get_weather"
