from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .errors import TransportError


@dataclass(slots=True, frozen=True)
class SSEEvent:
    event: str | None
    data: str


def parse_sse(
    lines: Iterator[bytes],
    *,
    max_line_bytes: int = 64 * 1024,
    max_event_bytes: int = 1024 * 1024,
) -> Iterator[SSEEvent]:
    event_name: str | None = None
    data_lines: list[str] = []
    event_bytes = 0

    for raw in lines:
        if len(raw) > max_line_bytes:
            raise TransportError(f"SSE line exceeds limit ({len(raw)} > {max_line_bytes})")
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        event_bytes += len(raw)
        if event_bytes > max_event_bytes:
            raise TransportError(f"SSE event exceeds limit ({event_bytes} > {max_event_bytes})")
        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name = None
            data_lines = []
            event_bytes = 0
            continue

        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    if data_lines:
        yield SSEEvent(event=event_name, data="\n".join(data_lines))
