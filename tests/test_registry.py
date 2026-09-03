"""The provider registry is the one place a provider is declared.

Every other table (ADAPTERS, ASYNC_ADAPTERS, CHAT_PRESET_ROUTES, the doctor's
known-provider list, the vet surface dump) is a view of it.  These tests pin
the rules the module docstring states, so a new provider that breaks one
fails here rather than drifting into a port.
"""

from __future__ import annotations

import dataclasses

import pytest

from lm15 import access
from lm15.compat import OPENAI_CHAT_PRESET_BASE_URLS, OpenAIChatCompat
from lm15.doctor import explain_auth
from lm15.features import AccessPolicy, EndpointSupport
from lm15.providers import OpenAIChatLM
from lm15.registry import PROVIDERS, ProviderDefinition, canonical_provider, lookup
from lm15.router import ADAPTERS, ASYNC_ADAPTERS, CHAT_PRESET_ROUTES, LMRouter, RouterConfig
from lm15.vet import op_surface_dump


class TestShape:
    def test_ids_are_canonical_and_match_access(self) -> None:
        for pid, d in PROVIDERS.items():
            assert pid == d.id == canonical_provider(pid)
            assert canonical_provider(d.access.provider) == pid

    def test_one_dialect_per_provider_string(self) -> None:
        # A provider string names ONE wire behavior; the adapter class is
        # the dialect, so two entries never share an id (a mapping cannot)
        # and every entry names a dialect the class actually speaks.
        dialect_of = {"openai-responses": "OpenAI", "openai-chat": "OpenAIChat", "anthropic": "Anthropic", "gemini": "Gemini"}
        for d in PROVIDERS.values():
            assert d.adapter.__name__.endswith("LM")
            if d.bound:
                assert d.adapter is OpenAIChatLM
            else:
                # Adapter-owned classes speak their dialect directly or by
                # subclassing it (XaiLM is an OpenAIChatLM).
                assert any(base.__name__ == f"{dialect_of[d.dialect]}LM" for base in d.adapter.__mro__), d.id

    def test_bound_entries_are_pure_data(self) -> None:
        for d in PROVIDERS.values():
            if not d.bound:
                assert d.compat is None
                assert d.placeholder_key is None
                continue
            assert d.compat is not None
            OpenAIChatCompat.preset(d.compat)  # exists
            assert d.access.base_url == OPENAI_CHAT_PRESET_BASE_URLS[d.compat]
            assert d.access.credential_policy == "key"

    def test_keyless_means_no_env_keys(self) -> None:
        for d in PROVIDERS.values():
            if d.placeholder_key is not None:
                assert d.env_keys == (), d.id
            else:
                assert d.credential_policy != "key" or d.env_keys, d.id

    def test_console_url_for_every_key_provider(self) -> None:
        # A user who lacks a key must be told where to get one.
        for d in PROVIDERS.values():
            if d.credential_policy != "oauth" and d.placeholder_key is None:
                assert d.console_url and d.console_url.startswith("https://"), d.id

    def test_definition_rejects_drift(self) -> None:
        bad_url = AccessPolicy(provider="groq", env_keys=("GROQ_API_KEY",), base_url="https://example.invalid")
        with pytest.raises(ValueError, match="compat table"):
            ProviderDefinition(id="groq", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=bad_url, compat="groq")
        with pytest.raises(ValueError, match="hyphenated"):
            ProviderDefinition(id="openai_chat", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=OpenAIChatLM.manifest)
        with pytest.raises(ValueError, match="names provider"):
            ProviderDefinition(id="groq", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=access.OPENROUTER, compat="groq")
        with pytest.raises(ValueError, match="no env_keys"):
            ProviderDefinition(id="groq", dialect="openai-chat", adapter=OpenAIChatLM,
                               async_adapter=OpenAIChatLM, access=access.GROQ, compat="groq",
                               placeholder_key="x")

    def test_frozen(self) -> None:
        with pytest.raises(TypeError):
            PROVIDERS["deepseek"] = PROVIDERS["groq"]  # type: ignore[index]
        with pytest.raises(dataclasses.FrozenInstanceError):
            PROVIDERS["groq"].note = "x"  # type: ignore[misc]


class TestViews:
    def test_router_tables_are_views(self) -> None:
        owned = {d.id for d in PROVIDERS.values() if not d.bound}
        bound = {d.id for d in PROVIDERS.values() if d.bound}
        assert set(ADAPTERS) == set(ASYNC_ADAPTERS) == owned
        assert set(CHAT_PRESET_ROUTES) == bound
        for pid, route in CHAT_PRESET_ROUTES.items():
            d = PROVIDERS[pid]
            assert route.env_keys == d.env_keys
            assert route.default_key == d.placeholder_key
        with pytest.raises(TypeError):
            ADAPTERS["x"] = OpenAIChatLM  # type: ignore[index]

    def test_surface_dump_reflects_every_provider(self) -> None:
        dumped = op_surface_dump({})["providers"]
        assert set(dumped) == set(PROVIDERS)
        for pid, d in PROVIDERS.items():
            assert dumped[pid]["env_keys"] == list(d.env_keys)
            assert dumped[pid]["auth_modes"] == list(d.access.auth_modes)
            assert dumped[pid]["supports"]["complete"] is d.supports.complete

    def test_doctor_knows_every_provider(self) -> None:
        for pid, d in PROVIDERS.items():
            if d.credential_policy == "oauth":
                continue  # reads a local file; covered by the auth fixtures
            report = explain_auth(pid, env={})
            kinds = [s.kind for s in report.steps]
            assert kinds[0] == "api_keys"
            for key in d.env_keys:
                assert f"env:{key}" in kinds
            if d.placeholder_key is not None:
                assert kinds[-1] == "placeholder" and report.configured
            elif d.credential_policy == "key":
                assert not report.configured

    def test_lookup_accepts_both_spellings(self) -> None:
        assert lookup("openai_chat") is PROVIDERS["openai-chat"]
        assert lookup("claude_code") is PROVIDERS["claude-code"]
        assert lookup("nope") is None


class TestBinding:
    def test_bound_lm_names_the_provider(self) -> None:
        # Before the registry, a routed groq LM called itself "openai_chat"
        # in errors and ModelInfo.provider.  The bound access policy makes
        # it name the provider, and carry the provider's env keys.
        router = LMRouter(RouterConfig(env={"GROQ_API_KEY": "g", "DEEPSEEK_API_KEY": "d"}))
        for pid in ("groq", "deepseek"):
            lm = router.lm(f"{pid}:model")
            assert isinstance(lm, OpenAIChatLM)
            assert lm.provider == pid
            assert lm.access is PROVIDERS[pid].access
            assert lm.base_url == PROVIDERS[pid].base_url
            assert lm.supports == EndpointSupport(complete=True, stream=True, models=True)

    def test_deepseek_declaration(self) -> None:
        d = PROVIDERS["deepseek"]
        assert d.dialect == "openai-chat" and d.bound
        assert d.env_keys == ("DEEPSEEK_API_KEY",)
        assert d.base_url == "https://api.deepseek.com"
        compat = OpenAIChatCompat.preset("deepseek")
        assert compat.thinking_format == "deepseek"
        assert compat.thinking_replay == "native"
        assert compat.assistant_reasoning_content == "include_empty"
        assert compat.max_tokens_field == "max_tokens"
