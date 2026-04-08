from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(slots=True, frozen=True)
class TransportPolicy:
    timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 0
    backoff_base_ms: int = 100
    proxy: str | None = None
    http2: bool = False


@dataclass(slots=True, frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json_body: dict | list | None = None
    body: bytes | None = None
    timeout: float | None = None


@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> dict:
        return json.loads(self.body)


class Transport:
    policy: TransportPolicy

    def request(self, req: HttpRequest) -> HttpResponse:
        raise NotImplementedError

    def stream(self, req: HttpRequest) -> Iterator[bytes]:
        raise NotImplementedError
