from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from typing import Any

from .client import UniversalLM


ENTRY_POINT_GROUP = "lm15.providers"


@dataclass(slots=True, frozen=True)
class PluginLoadResult:
    loaded: tuple[str, ...]
    failed: tuple[str, ...]


def discover_provider_entry_points(group: str = ENTRY_POINT_GROUP):
    eps = metadata.entry_points()
    # py3.10+/3.11/3.12 compatibility
    if hasattr(eps, "select"):
        return tuple(eps.select(group=group))
    return tuple(e for e in eps if getattr(e, "group", None) == group)


def load_plugins(
    client: UniversalLM,
    *,
    group: str = ENTRY_POINT_GROUP,
    continue_on_error: bool = True,
    plugin_kwargs: dict[str, dict[str, Any]] | None = None,
    allowlist: set[str] | None = None,
) -> PluginLoadResult:
    loaded: list[str] = []
    failed: list[str] = []
    plugin_kwargs = plugin_kwargs or {}

    for ep in discover_provider_entry_points(group=group):
        name = ep.name
        if allowlist is not None and name not in allowlist:
            continue
        try:
            obj = ep.load()
            kwargs = plugin_kwargs.get(name, {})

            # Supported plugin shapes:
            # - class/factory returning adapter instance
            # - adapter instance directly
            adapter = obj(**kwargs) if callable(obj) else obj
            client.register(adapter)
            loaded.append(name)
        except Exception:
            failed.append(name)
            if not continue_on_error:
                raise

    return PluginLoadResult(loaded=tuple(loaded), failed=tuple(failed))
