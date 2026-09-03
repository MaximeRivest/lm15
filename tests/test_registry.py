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
from lm15.compat import ANTHROPIC_PRESET_BASE_URLS, OPENAI_CHAT_PRESET_BASE_URLS, AnthropicCompat, OpenAIChatCompat
from lm15.doctor import explain_auth
from lm15.features import AccessPolicy, EndpointSupport
from lm15.providers import AnthropicLM, OpenAIChatLM
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
        bound_class = {"openai-chat": OpenAIChatLM, "anthropic": AnthropicLM}
        for d in PROVIDERS.values():
            assert d.adapter.__name__.endswith("LM")
            if d.bound:
                assert d.adapter is bound_class[d.dialect], d.id
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
            if d.dialect == "openai-chat":
                OpenAIChatCompat.preset(d.compat)  # exists
                assert d.access.base_url == OPENAI_CHAT_PRESET_BASE_URLS[d.compat]
            else:
                AnthropicCompat.preset(d.compat)
                assert d.access.base_url == ANTHROPIC_PRESET_BASE_URLS[d.compat]
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
        chat_bound = {d.id for d in PROVIDERS.values() if d.bound and d.dialect == "openai-chat"}
        assert set(ADAPTERS) == set(ASYNC_ADAPTERS) == owned
        assert set(CHAT_PRESET_ROUTES) == chat_bound  # the table's name says chat; other dialects route from the registry
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
        # The second bound dialect: same key, a different wire, its own class.
        lm = router.lm("deepseek-anthropic:deepseek-v4-flash")
        assert isinstance(lm, AnthropicLM) and lm.provider == "deepseek-anthropic"
        assert lm.base_url == "https://api.deepseek.com/anthropic/v1"
        assert lm.supports.models is False
        assert router.resolve("deepseek-anthropic:deepseek-v4-flash").adapter == "AnthropicLM"
        from lm15.router import AsyncLMRouter
        from lm15.providers import AsyncAnthropicLM

        alm = AsyncLMRouter(RouterConfig(env={"DEEPSEEK_API_KEY": "d"})).lm("deepseek-anthropic:m")
        assert isinstance(alm, AsyncAnthropicLM) and alm.base_url == lm.base_url

    def test_deepseek_anthropic_declaration_and_refusals(self) -> None:
        import json

        from lm15 import Config, FunctionTool, Message, Reasoning, Request, ToolChoice
        from lm15.errors import UnsupportedFeatureError, UnsupportedModelError

        d = PROVIDERS["deepseek-anthropic"]
        assert d.dialect == "anthropic" and d.bound and d.compat == "deepseek"
        assert d.env_keys == ("DEEPSEEK_API_KEY",) and d.access.auth_header == "x-api-key"
        lm = LMRouter(RouterConfig(env={"DEEPSEEK_API_KEY": "d"})).lm("deepseek-anthropic:deepseek-v4-flash")
        say = (Message.user("x"),)
        tool = FunctionTool(name="w", description="d", parameters={"type": "object", "properties": {}})

        def body(cfg, model="deepseek-v4-flash"):
            return json.loads(lm.build_request(Request(model=model, messages=say, tools=(tool,), config=cfg), stream=False).body)

        # Thinking: DeepSeek's shape, and an explicit off reaches the wire
        # (absence means ON there; live 2026-09-03).
        assert "thinking" not in body(Config(max_tokens=100))
        assert body(Config(max_tokens=100, reasoning=Reasoning(effort="off")))["thinking"] == {"type": "disabled"}
        b = body(Config(max_tokens=100, reasoning=Reasoning(effort="low")))
        assert b["thinking"] == {"type": "enabled"} and b["output_config"] == {"effort": "low"} and b["max_tokens"] == 100
        # Silent cells refused before the wire (guide--anthropic-api.md; live 2026-09-03).
        for cfg in (
            Config(reasoning=Reasoning(effort="low", thinking_budget=4096)),   # budget_tokens ignored
            Config(response_format={"type": "json_schema", "name": "x", "schema": {"type": "object"}}),  # schema ignored
            Config(tool_choice=ToolChoice(mode="auto", parallel=False)),      # disable_parallel_tool_use ignored
        ):
            with pytest.raises(UnsupportedFeatureError, match="deepseek-anthropic: .*ignore"):
                body(cfg)
        # claude-* is silently served by a DeepSeek model: refuse before the wire.
        with pytest.raises(UnsupportedModelError, match="silently substituted"):
            body(Config(max_tokens=10), model="claude-opus-4-1")
        # cache marks: not placed, not an error (implicit caching applies).
        from lm15 import CacheConfig

        b = body(Config(max_tokens=10, cache=CacheConfig(mode="auto")))
        assert "cache_control" not in json.dumps(b)
        # Plain Anthropic is untouched: manual class still sends a budget.
        plain = json.loads(AnthropicLM(api_key="k").build_request(
            Request(model="claude-haiku-4-5", messages=say, config=Config(max_tokens=100, reasoning=Reasoning(effort="low"))), stream=False).body)
        assert plain["thinking"]["type"] == "enabled" and "budget_tokens" in plain["thinking"]

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
        assert compat.user_field == "user_id"

    def test_user_field_rides_the_compat_name(self) -> None:
        # DeepSeek documents `user_id` and accepts `user` silently (live
        # 2026-09-03); OpenAI's dialect spells it `user`.
        import json

        from lm15 import Config, Message, Request

        req = Request(model="m", messages=(Message.user("x"),), config=Config(user_id="u1"))
        for preset, field_name in (("openai", "user"), ("groq", "user"), ("deepseek", "user_id")):
            body = json.loads(OpenAIChatLM(api_key="k", compat=preset).build_request(req, stream=False).body)
            assert body[field_name] == "u1", preset
            assert ({"user", "user_id"} - {field_name}).isdisjoint(body), preset
        with pytest.raises(ValueError):
            OpenAIChatCompat(user_field="uid")  # type: ignore[arg-type]

    def test_zai_declaration(self) -> None:
        d = PROVIDERS["zai"]
        assert d.dialect == "openai-chat" and d.bound
        assert d.env_keys == ("ZAI_API_KEY",)
        assert d.base_url == "https://api.z.ai/api/paas/v4"
        compat = OpenAIChatCompat.preset("zai")
        # The wire shape docs.z.ai documents (ChatThinking) — the deepseek
        # shape — not Qwen's enable_thinking the preset sent before 2026-09-03.
        assert compat.thinking_format == "deepseek"
        assert compat.thinking_replay == "native"
        assert compat.assistant_reasoning_content is None  # Z.AI answered 200 without it (live 2026-09-03)
        assert compat.user_field == "user_id"
        assert compat.forced_tool_choice == "reject" and compat.json_schema == "reject"

    def test_zai_refuses_silent_cells_before_the_wire(self) -> None:
        # Live 2026-09-03: tool_choice=required answered text, =none called
        # the tool, json_schema returned fenced free-form JSON — all HTTP 200.
        from lm15 import Config, FunctionTool, Message, Request, ToolChoice
        from lm15.errors import UnsupportedFeatureError

        tool = FunctionTool(name="w", description="d", parameters={"type": "object", "properties": {}})
        lm = LMRouter(RouterConfig(env={"ZAI_API_KEY": "k"})).lm("zai:glm-5.3-flash")
        for cfg in (
            Config(tool_choice=ToolChoice(mode="required")),
            Config(tool_choice=ToolChoice(mode="none")),
            Config(tool_choice=ToolChoice(mode="auto", allowed=("w",))),
            Config(response_format={"type": "json_schema", "name": "x", "schema": {"type": "object"}}),
        ):
            with pytest.raises(UnsupportedFeatureError, match="zai: .*silently ignored"):
                lm.build_request(Request(model="glm-5.3-flash", messages=(Message.user("x"),), tools=(tool,), config=cfg), stream=False)
        # The honoured forms still go out.
        for cfg in (Config(tool_choice=ToolChoice(mode="auto")), Config(response_format={"type": "json_object"})):
            lm.build_request(Request(model="glm-5.3-flash", messages=(Message.user("x"),), tools=(tool,), config=cfg), stream=False)
        # Other presets are untouched: the knob is per server, not dialect-wide.
        groq = LMRouter(RouterConfig(env={"GROQ_API_KEY": "k"})).lm("groq:m")
        groq.build_request(Request(model="m", messages=(Message.user("x"),), tools=(tool,), config=Config(tool_choice=ToolChoice(mode="required"))), stream=False)
