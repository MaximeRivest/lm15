"""Manifest-declared credential policy and the uniform login entry point.

The router and doctor derive OAuth behavior from
``ProviderManifest.credential_policy`` — there is no hardcoded provider-name
list left to drift (spec/auth.md AUTH-1, amended 2026-09-01).  ``login()``
is the one door for interactive flows: it runs the flow lm15 owns (xai) and
fails typed, with the exact fix, everywhere else.
"""
from __future__ import annotations

import pytest

from lm15.auth import login
from lm15.errors import UnsupportedFeatureError
from lm15.router import ADAPTERS, CHAT_PRESET_ROUTES, LMRouter, RouterConfig, _credential_policy

VALID_POLICIES = {"key", "oauth", "key-then-oauth"}


class TestManifestPolicies:
    def test_every_adapter_declares_a_valid_policy(self):
        for provider, cls in ADAPTERS.items():
            assert cls.manifest.credential_policy in VALID_POLICIES, provider

    def test_declared_policies_match_auth_modes(self):
        # The policy and the auth_modes tell one story, not two.
        assert _credential_policy("claude-code") == "oauth"
        assert _credential_policy("openai-codex") == "oauth"
        assert _credential_policy("xai") == "key-then-oauth"
        assert _credential_policy("openai") == "key"
        assert _credential_policy("anthropic") == "key"
        assert _credential_policy("gemini") == "key"

    def test_oauth_policies_declare_no_env_keys(self):
        # "oauth" means the env chain never runs; declaring env keys for an
        # oauth provider would be a contradiction in the manifest itself.
        for provider, cls in ADAPTERS.items():
            if cls.manifest.credential_policy == "oauth":
                assert cls.manifest.env_keys == (), provider

    def test_presets_are_key_providers(self):
        for provider in CHAT_PRESET_ROUTES:
            assert _credential_policy(provider) == "key"

    def test_resolution_describe_names_the_xai_fallback(self):
        resolution = LMRouter(RouterConfig(env={})).resolve("xai:grok-4")
        assert "local OAuth credential" in resolution.describe()


class TestUniformLogin:
    def test_xai_dispatches_to_the_owned_flow(self, monkeypatch):
        calls: list = []
        import lm15.auth as auth_module

        def fake_login_xai(auth_path=None, *, echo=print):
            calls.append((auth_path, echo))
            return "credential"

        monkeypatch.setattr(auth_module, "login_xai", fake_login_xai)
        assert login("xai", credentials_path="/tmp/creds.json") == "credential"
        assert calls[0][0] == "/tmp/creds.json"

    def test_cli_owned_flows_name_the_cli(self):
        with pytest.raises(UnsupportedFeatureError, match="/login"):
            login("claude-code")
        with pytest.raises(UnsupportedFeatureError, match="codex login"):
            login("openai-codex")

    def test_underscore_spelling_is_accepted(self):
        with pytest.raises(UnsupportedFeatureError, match="codex login"):
            login("openai_codex")

    def test_key_providers_name_the_console_url(self):
        with pytest.raises(UnsupportedFeatureError, match="platform.openai.com"):
            login("openai")
        with pytest.raises(UnsupportedFeatureError, match="console.anthropic.com"):
            login("anthropic")
        with pytest.raises(UnsupportedFeatureError, match="aistudio.google.com"):
            login("gemini")

    def test_keyless_local_servers_explain_themselves(self):
        with pytest.raises(UnsupportedFeatureError, match="keyless"):
            login("ollama")

    def test_unknown_provider_gets_generic_guidance(self):
        with pytest.raises(UnsupportedFeatureError, match="no login flow"):
            login("somebody-else")
