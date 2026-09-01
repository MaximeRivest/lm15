"""
lm15.features — Provider capability and endpoint declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EndpointSupport:
    complete: bool = True
    stream: bool = True
    live: bool = False
    files: bool = False
    batches: bool = False
    images: bool = False
    speech: bool = False
    responses_api: bool = False
    models: bool = False

    # Escape hatch for endpoint names not yet promoted to typed booleans.
    extra: frozenset[str] = field(default_factory=frozenset)

    def supports_endpoint(self, name: str) -> bool:
        if name in self.extra:
            return True
        return bool(getattr(self, name, False))


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    provider: str
    supports: EndpointSupport
    auth_modes: tuple[str, ...] = field(default_factory=tuple)
    enterprise_variants: tuple[str, ...] = field(default_factory=tuple)
    env_keys: tuple[str, ...] = field(default_factory=tuple)
