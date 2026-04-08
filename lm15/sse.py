from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(slots=True, frozen=True)
class SSEEvent:
    event: str | None
    data: str


def parse_sse(lines: Iterator[bytes]) -> Iterator[SSEEvent]:
    event_name: str | None = None
    data_lines: list[str] = []

    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if data_lines:
                yield SSEEvent(event=event_name, data="\n".join(data_lines))
            event_name = None
            data_lines = []
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
