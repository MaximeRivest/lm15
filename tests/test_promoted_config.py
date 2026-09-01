"""Promoted config knobs (the 2026-09-01 extensions burn-down).

service_tier / user_id / store graduated from provider-syntax smuggling
in config.extensions to canonical Config fields. Wire equality for the
promoted spellings is pinned by the contract's request direction (the
rewritten cases build the same frozen bodies); these tests pin the
mapping table and the no-silent-drop raises.
"""
from __future__ import annotations

import pytest

from lm15 import Config, Message, Request, UnsupportedFeatureError
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM
from lm15.providers.openai_chat import OpenAIChatLM
from lm15.serde import config_from_dict, config_to_dict
from lm15.testing import FakeTransport


def req(**config) -> Request:
    return Request(model="m", messages=(Message.user("hi"),), config=Config(**config))


def payload(lm, request):
    return lm._payload(request, stream=False) if not isinstance(lm, GeminiLM) else lm._payload(request)


# ─── Mapping table ───────────────────────────────────────────────────

def test_openai_spellings() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = payload(lm, req(service_tier="flex", user_id="end-user-7", store=False))
    assert body["service_tier"] == "flex"
    assert body["safety_identifier"] == "end-user-7"  # current field; `user` is the deprecated spelling
    assert body["store"] is False
    assert "user" not in body and "user_id" not in body


def test_openai_chat_spellings() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(req(service_tier="default", user_id="end-user-7", store=True), stream=False)
    assert body["service_tier"] == "default"
    assert body["user"] == "end-user-7"  # Chat Completions dialect spelling
    assert body["store"] is True


def test_anthropic_spellings_and_store_raises() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(req(service_tier="standard_only", user_id="u-1"), stream=False)
    assert body["service_tier"] == "standard_only"
    assert body["metadata"] == {"user_id": "u-1"}
    with pytest.raises(UnsupportedFeatureError, match="store"):
        lm._payload(req(store=False), stream=False)


def test_gemini_store_and_service_tier_map_user_id_raises() -> None:
    # serviceTier joined GenerateContent between the April and September
    # 2026 doc snapshots; verified live 2026-09-01 (accepted, echoed in
    # usageMetadata.serviceTier; unknown values rejected with HTTP 400).
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    assert lm._payload(req(store=False))["store"] is False
    assert lm._payload(req(service_tier="flex"))["serviceTier"] == "flex"
    assert "serviceTier" not in lm._payload(req())
    with pytest.raises(UnsupportedFeatureError, match="user_id"):
        lm._payload(req(user_id="u-1"))


def test_extensions_still_win_over_promoted_fields() -> None:
    # Passthrough precedence is the existing rule: extensions apply last.
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = payload(lm, req(service_tier="flex", extensions={"service_tier": "priority"}))
    assert body["service_tier"] == "priority"


# ─── Type + serde ────────────────────────────────────────────────────

def test_config_validation() -> None:
    with pytest.raises(ValueError):
        Config(service_tier="")
    with pytest.raises(ValueError):
        Config(user_id="")
    with pytest.raises(TypeError):
        Config(store="yes")


def test_serde_roundtrip_and_false_survives() -> None:
    cfg = Config(service_tier="flex", user_id="u-1", store=False)
    d = config_to_dict(cfg)
    assert d == {"service_tier": "flex", "user_id": "u-1", "store": False}
    assert config_from_dict(d) == cfg
    assert config_to_dict(Config()) == {}  # all-default still collapses
