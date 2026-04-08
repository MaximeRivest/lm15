from __future__ import annotations

from dataclasses import dataclass


class AuthStrategy:
    def apply_headers(self, headers: dict[str, str]) -> dict[str, str]:
        return headers

    def apply_params(self, params: dict[str, str]) -> dict[str, str]:
        return params


@dataclass(slots=True, frozen=True)
class BearerAuth(AuthStrategy):
    token: str

    def apply_headers(self, headers: dict[str, str]) -> dict[str, str]:
        out = dict(headers)
        out["Authorization"] = f"Bearer {self.token}"
        return out


@dataclass(slots=True, frozen=True)
class HeaderKeyAuth(AuthStrategy):
    header: str
    key: str

    def apply_headers(self, headers: dict[str, str]) -> dict[str, str]:
        out = dict(headers)
        out[self.header] = self.key
        return out


@dataclass(slots=True, frozen=True)
class QueryKeyAuth(AuthStrategy):
    param: str
    key: str

    def apply_params(self, params: dict[str, str]) -> dict[str, str]:
        out = dict(params)
        out[self.param] = self.key
        return out
