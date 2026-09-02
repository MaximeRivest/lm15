"""CacheConfig intents, CacheInfo/CachePage, and CachedPrefix (MAP-6 step 1).

Wire mappings are pinned by the contract (cache direction, pending);
these tests pin the canonical values: validation, serde round-trips, and
the Request that ``cached + messages`` builds.
"""
from __future__ import annotations

import pytest

from lm15 import CacheConfig, CachedPrefix, CacheInfo, CachePage, Config, FunctionTool, Message, Request
from lm15.serde import (
    cache_config_from_dict, cache_config_to_dict, cache_info_from_dict, cache_info_to_dict,
    cache_page_from_dict, cache_page_to_dict, cached_prefix_from_dict, cached_prefix_to_dict, request_to_dict,
)


def _prefix(**kw) -> Request:
    return Request(model="m", system="S", tools=(FunctionTool(name="f"),), messages=[Message.user("DOC")], **kw)


# ─── CacheConfig ─────────────────────────────────────────────────────

def test_cache_config_intents_round_trip() -> None:
    for cfg in (CacheConfig(), CacheConfig(prefix="stable"), CacheConfig(prefix="history"),
                CacheConfig(prefix_until_index=2), CacheConfig(resource="caches/abc", prefix_until_index=0),
                CacheConfig(mode="off")):
        assert cache_config_from_dict(cache_config_to_dict(cfg)) == cfg


def test_cache_config_default_serializes_to_mode_only() -> None:
    assert cache_config_to_dict(CacheConfig()) == {"mode": "auto"}


def test_cache_config_rejects_conflicts() -> None:
    with pytest.raises(ValueError, match="both prefix and prefix_until_index"):
        CacheConfig(prefix="stable", prefix_until_index=0)
    with pytest.raises(ValueError, match="mode='off'"):
        CacheConfig(mode="off", prefix="history")
    with pytest.raises(ValueError, match="mode='off'"):
        CacheConfig(mode="off", resource="caches/x")
    with pytest.raises(ValueError, match="unsupported cache prefix"):
        CacheConfig(prefix="everything")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        CacheConfig(resource="")


# ─── CacheInfo / CachePage ───────────────────────────────────────────

def test_cache_info_round_trip_and_validation() -> None:
    info = CacheInfo(id="caches/1", model="m", tokens=3574, created_at="2026-09-01T10:00:00Z",
                     expires_at="2026-09-01T11:00:00Z", label="docs", provider_data={"name": "caches/1"})
    assert cache_info_from_dict(cache_info_to_dict(info)) == info
    assert cache_info_to_dict(CacheInfo(id="c", model="m")) == {"id": "c", "model": "m"}
    assert CacheInfo(id="c", model="m", tokens=2.0).tokens == 2  # Number rule
    with pytest.raises(ValueError):
        CacheInfo(id="c", model="m", tokens=-1)
    with pytest.raises(ValueError):
        CacheInfo(id="", model="m")


def test_cache_page_round_trip() -> None:
    page = CachePage(items=(CacheInfo(id="a", model="m"), CacheInfo(id="b", model="m")), next_cursor="tok")
    assert cache_page_from_dict(cache_page_to_dict(page)) == page
    assert cache_page_to_dict(CachePage()) == {}


# ─── CachedPrefix ────────────────────────────────────────────────────

def test_cached_prefix_builds_request_with_boundary_at_seam() -> None:
    cached = CachedPrefix(_prefix())
    req = cached + "What is it about?"
    assert req.model == "m" and req.system == "S" and req.tools == cached.prefix.tools
    assert [m.parts[0].text for m in req.messages] == ["DOC", "What is it about?"]
    assert req.config.cache == CacheConfig(prefix_until_index=0)
    assert cached.id is None and cached.expires_at is None


def test_cached_prefix_with_resource_sets_resource_id() -> None:
    info = CacheInfo(id="caches/1", model="m", expires_at="2026-09-01T11:00:00Z")
    cached = CachedPrefix(_prefix(), info)
    req = cached.request([Message.user("q1"), Message.assistant("a1"), Message.user("q2")],
                         config=Config(max_tokens=16))
    assert req.config.cache == CacheConfig(prefix_until_index=0, resource="caches/1")
    assert req.config.max_tokens == 16
    assert cached.id == "caches/1" and cached.expires_at == "2026-09-01T11:00:00Z"
    assert len(req.messages) == 4


def test_cached_prefix_accepts_a_suffix_request_without_system_or_tools() -> None:
    cached = CachedPrefix(_prefix())
    suffix = Request(model="m", messages=[Message.user("q")], config=Config(temperature=0))
    req = cached + suffix
    assert req.config.temperature == 0.0 and req.config.cache == CacheConfig(prefix_until_index=0)
    with pytest.raises(ValueError, match="prefix owns them"):
        cached + Request(model="m", system="other", messages=[Message.user("q")])
    with pytest.raises(ValueError, match="model must equal"):
        cached + Request(model="other", messages=[Message.user("q")])


def test_cached_prefix_rejections() -> None:
    with pytest.raises(ValueError, match="default Config"):
        CachedPrefix(_prefix(config=Config(temperature=0.5)))
    with pytest.raises(ValueError, match="one model"):
        CachedPrefix(_prefix(), CacheInfo(id="c", model="other"))
    cached = CachedPrefix(_prefix())
    with pytest.raises(ValueError, match="decided by the CachedPrefix"):
        cached.request("q", config=Config(cache=CacheConfig()))
    with pytest.raises(TypeError):
        cached.request([])
    with pytest.raises(TypeError):
        CachedPrefix("not a request")  # type: ignore[arg-type]


def test_cached_prefix_serde_round_trip() -> None:
    cached = CachedPrefix(_prefix(), CacheInfo(id="caches/1", model="m", tokens=10))
    d = cached_prefix_to_dict(cached)
    assert d["prefix"] == request_to_dict(cached.prefix) and d["resource"]["id"] == "caches/1"
    assert cached_prefix_from_dict(d) == cached
    assert cached_prefix_from_dict(cached_prefix_to_dict(CachedPrefix(_prefix()))) == CachedPrefix(_prefix())
