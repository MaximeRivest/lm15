from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

from .factory import providers as provider_env_keys
from .model_catalog import ModelSpec, fetch_models_dev

_CACHE_TTL_SECONDS = 300.0
_LIVE_CACHE: dict[str, tuple[float, list[ModelSpec]]] = {}


def _parse_env_file(path: str | Path, env_map: dict[str, tuple[str, ...]]) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = Path(path).expanduser().read_text()
    except (OSError, ValueError):
        return out

    key_to_provider: dict[str, str] = {}
    for p, keys in env_map.items():
        for k in keys:
            key_to_provider[k] = p

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        provider = key_to_provider.get(key.strip())
        if provider:
            out[provider] = value.strip().strip('"\'')
    return out


def _resolve_api_keys(
    api_key: str | dict[str, str] | None,
    provider: str | None,
    env: str | Path | None,
) -> dict[str, str]:
    env_map = provider_env_keys()
    providers = tuple(env_map.keys())

    explicit: dict[str, str] = {}
    if isinstance(api_key, dict):
        explicit = api_key
    elif isinstance(api_key, str):
        if provider:
            explicit = {provider: api_key}
        else:
            explicit = {p: api_key for p in providers}

    file_keys: dict[str, str] = _parse_env_file(env, env_map) if env else {}

    resolved: dict[str, str] = {}
    for p, vars_ in env_map.items():
        v = explicit.get(p) or file_keys.get(p)
        if not v:
            for var in vars_:
                found = os.getenv(var)
                if found:
                    v = found
                    break
        if v:
            resolved[p] = v
    return resolved


def _fetch_json(url: str, *, headers: dict[str, str] | None = None, timeout: float = 5.0) -> dict[str, Any]:
    req = urllib.request.Request(url=url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _fetch_openai_models(api_key: str, timeout: float) -> list[ModelSpec]:
    data = _fetch_json(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    out: list[ModelSpec] = []
    for item in data.get("data", []) or []:
        mid = str(item.get("id") or "")
        if not mid:
            continue
        out.append(
            ModelSpec(
                id=mid,
                provider="openai",
                context_window=None,
                max_output=None,
                input_modalities=(),
                output_modalities=(),
                tool_call=False,
                structured_output=False,
                reasoning=False,
                raw=item,
            )
        )
    return out


def _fetch_anthropic_models(api_key: str, timeout: float) -> list[ModelSpec]:
    data = _fetch_json(
        "https://api.anthropic.com/v1/models",
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
        timeout=timeout,
    )
    out: list[ModelSpec] = []
    for item in data.get("data", []) or []:
        mid = str(item.get("id") or "")
        if not mid:
            continue
        out.append(
            ModelSpec(
                id=mid,
                provider="anthropic",
                context_window=None,
                max_output=None,
                input_modalities=(),
                output_modalities=(),
                tool_call=False,
                structured_output=False,
                reasoning=False,
                raw=item,
            )
        )
    return out


def _fetch_gemini_models(api_key: str, timeout: float) -> list[ModelSpec]:
    data = _fetch_json(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
        timeout=timeout,
    )
    out: list[ModelSpec] = []
    for item in data.get("models", []) or []:
        name = str(item.get("name") or "")
        if not name:
            continue
        mid = name[len("models/") :] if name.startswith("models/") else name
        out.append(
            ModelSpec(
                id=mid,
                provider="gemini",
                context_window=item.get("inputTokenLimit"),
                max_output=item.get("outputTokenLimit"),
                input_modalities=(),
                output_modalities=(),
                tool_call=False,
                structured_output=False,
                reasoning=False,
                raw=item,
            )
        )
    return out


_FETCHERS = {
    "openai": _fetch_openai_models,
    "anthropic": _fetch_anthropic_models,
    "gemini": _fetch_gemini_models,
}


def _fetch_live_models_for_provider(provider: str, api_key: str, timeout: float) -> list[ModelSpec]:
    fetcher = _FETCHERS.get(provider)
    if fetcher is None:
        return []
    return fetcher(api_key, timeout)


def _merge_specs(primary: list[ModelSpec], fallback: list[ModelSpec]) -> list[ModelSpec]:
    merged: dict[tuple[str, str], ModelSpec] = {(s.provider, s.id): s for s in primary}
    for f in fallback:
        k = (f.provider, f.id)
        p = merged.get(k)
        if p is None:
            merged[k] = f
            continue
        merged[k] = ModelSpec(
            id=p.id,
            provider=p.provider,
            context_window=p.context_window if p.context_window is not None else f.context_window,
            max_output=p.max_output if p.max_output is not None else f.max_output,
            input_modalities=p.input_modalities or f.input_modalities,
            output_modalities=p.output_modalities or f.output_modalities,
            tool_call=p.tool_call or f.tool_call,
            structured_output=p.structured_output or f.structured_output,
            reasoning=p.reasoning or f.reasoning,
            raw={**f.raw, **p.raw},
        )
    return sorted(merged.values(), key=lambda x: (x.provider, x.id))


def _filter_specs(
    specs: list[ModelSpec],
    *,
    supports: set[str] | None,
    input_modalities: set[str] | None,
    output_modalities: set[str] | None,
) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for s in specs:
        features: set[str] = set()
        if s.tool_call:
            features.add("tools")
        if s.structured_output:
            features.add("json_output")
        if s.reasoning:
            features.add("reasoning")

        if supports and not supports.issubset(features):
            continue
        if input_modalities and not input_modalities.issubset(set(s.input_modalities)):
            continue
        if output_modalities and not output_modalities.issubset(set(s.output_modalities)):
            continue
        out.append(s)
    return out


def models(
    *,
    provider: str | None = None,
    live: bool = True,
    refresh: bool = False,
    timeout: float = 5.0,
    api_key: str | dict[str, str] | None = None,
    env: str | Path | None = None,
    supports: set[str] | None = None,
    input_modalities: set[str] | None = None,
    output_modalities: set[str] | None = None,
) -> list[ModelSpec]:
    providers = tuple(provider_env_keys().keys())
    selected = tuple([provider] if provider else providers)
    keys = _resolve_api_keys(api_key=api_key, provider=provider, env=env)

    live_specs: list[ModelSpec] = []
    if live:
        now = time.time()
        for p in selected:
            cached = _LIVE_CACHE.get(p)
            if cached and not refresh and now - cached[0] <= _CACHE_TTL_SECONDS:
                live_specs.extend(cached[1])
                continue
            k = keys.get(p)
            if not k:
                continue
            try:
                fetched = _fetch_live_models_for_provider(p, k, timeout)
            except Exception:
                fetched = []
            _LIVE_CACHE[p] = (now, fetched)
            live_specs.extend(fetched)

    fallback_specs: list[ModelSpec] = []
    try:
        fallback_specs = [s for s in fetch_models_dev(timeout=timeout) if s.provider in selected]
    except Exception:
        fallback_specs = []

    merged = _merge_specs(live_specs, fallback_specs)
    return _filter_specs(
        merged,
        supports=supports,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
    )


def providers_info(
    *,
    live: bool = True,
    refresh: bool = False,
    timeout: float = 5.0,
    api_key: str | dict[str, str] | None = None,
    env: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    env_map = provider_env_keys()
    keys = _resolve_api_keys(api_key=api_key, provider=None, env=env)
    specs = models(live=live, refresh=refresh, timeout=timeout, api_key=api_key, env=env)

    counts: dict[str, int] = {}
    for s in specs:
        counts[s.provider] = counts.get(s.provider, 0) + 1

    out: dict[str, dict[str, Any]] = {}
    for p, env_keys in env_map.items():
        out[p] = {
            "env_keys": env_keys,
            "configured": p in keys,
            "model_count": counts.get(p, 0),
        }
    return out
