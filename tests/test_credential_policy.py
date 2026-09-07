"""Manifest-declared credential policy and the uniform login entry point.

The router and doctor derive OAuth behavior from
``ProviderManifest.credential_policy`` — there is no hardcoded provider-name
list left to drift (spec/auth.md AUTH-1, amended 2026-09-01).  ``login()``
is the one door for interactive flows: it runs the flow lm15 owns (xai) and
fails typed, with the exact fix, everywhere else.

The oauth-unless-explicit chain (explicit config → stored subscription →
env) is pinned hermetically here with a fake adapter, and against the
contract fixtures in test_auth_resolution_contract.py.
"""
from __future__ import annotations

import pytest

from lm15.auth import login
from lm15.errors import UnsupportedFeatureError
from lm15.features import EndpointSupport, ProviderManifest
from lm15.router import ADAPTERS, CHAT_PRESET_ROUTES, LMRouter, Resolution, RouterConfig, _build_lm, _credential_policy

VALID_POLICIES = {"key", "oauth", "oauth-unless-explicit"}


class TestManifestPolicies:
    def test_every_adapter_declares_a_valid_policy(self):
        for provider, cls in ADAPTERS.items():
            assert cls.manifest.credential_policy in VALID_POLICIES, provider

    def test_declared_policies_match_auth_modes(self):
        # The policy and the auth_modes tell one story, not two.
        assert _credential_policy("claude-code") == "oauth"
        assert _credential_policy("openai-codex") == "oauth"
        assert _credential_policy("xai") == "oauth-unless-explicit"
        assert _credential_policy("openai") == "key"
        assert _credential_policy("anthropic") == "key"
        assert _credential_policy("gemini") == "key"

    def test_oauth_unless_explicit_adapters_expose_the_probe(self):
        # The policy requires an offline stored-credential probe; a declared
        # policy without the probe would crash the router at build time.
        for provider, cls in ADAPTERS.items():
            if cls.manifest.credential_policy == "oauth-unless-explicit":
                assert callable(getattr(cls, "has_stored_credential", None)), provider

    def test_oauth_policies_declare_no_env_keys(self):
        # "oauth" means the env chain never runs; declaring env keys for an
        # oauth provider would be a contradiction in the manifest itself.
        for provider, cls in ADAPTERS.items():
            if cls.manifest.credential_policy == "oauth":
                assert cls.manifest.env_keys == (), provider

    def test_presets_are_key_providers(self):
        from lm15.registry import PROVIDERS

        for provider in CHAT_PRESET_ROUTES:
            if PROVIDERS[provider].hosted:
                # Cloud doors declare their cloud's chain (AUTH-1, 2026-09-03).
                assert _credential_policy(provider) in ("key", "aws-chain", "azure-chain", "gcp-chain")
            else:
                assert _credential_policy(provider) == "key"

    def test_resolution_describe_names_the_xai_chain(self):
        described = LMRouter(RouterConfig(env={})).resolve("xai:grok-4").describe()
        assert "stored subscription OAuth credential" in described
        # The chain reads in resolution order: explicit config, then the
        # subscription, then the env var.
        assert described.index("explicit api_keys") < described.index("subscription") < described.index("XAI_API_KEY")


def _fake_xai_adapter(stored: bool):
    class FakeXaiLM:
        manifest = ProviderManifest(
            provider="fakexai",
            supports=EndpointSupport(),
            auth_modes=("bearer", "xai-oauth"),
            env_keys=("FAKE_XAI_KEY",),
            credential_policy="oauth-unless-explicit",
        )

        def __init__(self, api_key=None, **kwargs):
            self.api_key = api_key

        @classmethod
        def has_stored_credential(cls) -> bool:
            return stored

    return FakeXaiLM


def _resolution() -> Resolution:
    return Resolution(requested="fakexai:m", model="m", provider="fakexai", adapter="FakeXaiLM", source="prefix")


class TestOauthUnlessExplicitChain:
    """Hermetic pin of the build-time order: config → stored login → env."""

    def test_explicit_config_beats_stored_login(self):
        cls = _fake_xai_adapter(stored=True)
        lm = _build_lm(_resolution(), RouterConfig(env={"FAKE_XAI_KEY": "envkey"}, api_keys={"fakexai": "configkey"}), {"fakexai": cls})
        assert lm.api_key == "configkey"

    def test_stored_login_beats_env(self):
        cls = _fake_xai_adapter(stored=True)
        lm = _build_lm(_resolution(), RouterConfig(env={"FAKE_XAI_KEY": "envkey"}), {"fakexai": cls})
        assert lm.api_key is None  # self-resolving OAuth constructor: no money per token

    def test_env_used_when_no_usable_login(self):
        cls = _fake_xai_adapter(stored=False)
        lm = _build_lm(_resolution(), RouterConfig(env={"FAKE_XAI_KEY": "envkey"}), {"fakexai": cls})
        assert lm.api_key == "envkey"

    def test_nothing_anywhere_defers_to_the_oauth_constructor(self):
        # The real adapter raises the typed login-hint error here; the
        # router's job is only to hand over to the self-resolving path.
        cls = _fake_xai_adapter(stored=False)
        lm = _build_lm(_resolution(), RouterConfig(env={}), {"fakexai": cls})
        assert lm.api_key is None


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
