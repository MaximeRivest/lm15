from __future__ import annotations

import base64
import json
import urllib.parse
from typing import Any

from ..models import ModelInfo, ModelOrigin
from ..transports import TransportRequest
from ..types import (
    AudioPart,
    BinaryPart,
    CitationPart,
    DocumentPart,
    ImagePart,
    Message,
    Part,
    TextPart,
    ThinkingPart,
    TokenLogprob,
    ToolResultPart,
    TopLogprob,
    VideoPart,
)

JsonPayload = dict[str, Any] | list[Any]


def parts_to_text(parts: tuple[Part, ...]) -> str:
    """Lossy text rendering used for provider fields that only accept text."""

    out: list[str] = []
    for part in parts:
        if isinstance(part, TextPart):
            out.append(part.text)
        elif isinstance(part, ThinkingPart) and part.text:
            out.append(part.text)
        elif isinstance(part, CitationPart):
            bits = [x for x in (part.title, part.url, part.text) if x]
            if bits:
                out.append(" — ".join(bits))
    return "\n".join(out)


def message_text(msg: Message) -> str:
    return parts_to_text(msg.parts)


def media_data_uri(part: ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart) -> str:
    if part.data is None:
        raise ValueError(f"{part.type} part has no inline data")
    return f"data:{part.media_type};base64,{part.data}"


def media_bytes(part: ImagePart | AudioPart | VideoPart | DocumentPart | BinaryPart) -> bytes:
    return part.bytes


def extension_config(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def json_dumps(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def build_url(url: str, params: dict[str, Any] | None = None) -> str:
    if not params:
        return url
    clean = {k: v for k, v in params.items() if v is not None}
    if not clean:
        return url
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{urllib.parse.urlencode(clean)}"


def make_json_request(
    *,
    method: str,
    url: str,
    headers: dict[str, str] | list[tuple[str, str]] | None = None,
    params: dict[str, Any] | None = None,
    payload: JsonPayload | None = None,
    body: bytes | None = None,
    connect_timeout: float | None = None,
    read_timeout: float | None = None,
    write_timeout: float | None = None,
) -> TransportRequest:
    hdrs = list(headers.items()) if isinstance(headers, dict) else list(headers or [])
    if payload is not None:
        body = json_dumps(payload)
        if not any(k.lower() == "content-type" for k, _ in hdrs):
            hdrs.append(("Content-Type", "application/json"))
    return TransportRequest(
        method=method,
        url=build_url(url, params),
        headers=hdrs,
        body=body or b"",
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
        write_timeout=write_timeout,
    )


def model_infos_from_entries(
    entries: Any,
    *,
    provider: str,
    api_family: str,
    id_of: "callable",
) -> tuple[ModelInfo, ...]:
    """Map a provider's list-models entries to canonical ModelInfo.

    ``id`` is the usable Request.model string (``id_of`` extracts it); the
    verbatim wire entry is preserved under ``origin.provider_data`` (opaque,
    never cleaned).  Entries without a usable id are skipped.
    """
    if not isinstance(entries, list):
        return ()
    out: list[ModelInfo] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = id_of(entry)
        if not isinstance(model_id, str) or not model_id:
            continue
        out.append(
            ModelInfo(
                id=model_id,
                provider=provider,
                api_family=api_family,
                origin=ModelOrigin(type="provider", provider_data=entry),
            )
        )
    return tuple(out)


def part_to_openai_input(part: Part) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "input_text", "text": part.text}

    if isinstance(part, ImagePart):
        if part.url is not None:
            payload = {"type": "input_image", "image_url": part.url}
            if part.detail:
                payload["detail"] = part.detail
            return payload
        if part.data is not None:
            payload = {"type": "input_image", "image_url": media_data_uri(part)}
            if part.detail:
                payload["detail"] = part.detail
            return payload
        if part.file_id is not None:
            return {"type": "input_image", "file_id": part.file_id}

    if isinstance(part, AudioPart):
        if part.data is not None:
            media = (part.media_type or "audio/wav").split("/", 1)[-1]
            if media in {"mpeg", "mp3"}:
                media = "mp3"
            return {"type": "input_audio", "audio": part.data, "format": media}
        if part.url is not None:
            return {"type": "input_audio", "audio_url": part.url}
        if part.file_id is not None:
            return {"type": "input_audio", "file_id": part.file_id}

    if isinstance(part, (DocumentPart, BinaryPart)):
        if part.url is not None:
            return {"type": "input_file", "file_url": part.url}
        if part.data is not None:
            # OpenAI requires a filename alongside inline file_data (live
            # 2026-06-11: 400 missing_required_parameter without one); derive
            # a deterministic name from the media-type subtype.
            ext = (part.media_type or "application/octet-stream").split("/", 1)[-1].split("+", 1)[0] or "bin"
            return {"type": "input_file", "filename": f"file.{ext}", "file_data": media_data_uri(part)}
        if part.file_id is not None:
            return {"type": "input_file", "file_id": part.file_id}

    if isinstance(part, VideoPart):
        if part.url is not None:
            return {"type": "input_video", "video_url": part.url}
        if part.data is not None:
            return {"type": "input_video", "video_data": media_data_uri(part)}
        if part.file_id is not None:
            return {"type": "input_video", "file_id": part.file_id}

    if isinstance(part, ToolResultPart):
        return {"type": "input_text", "text": parts_to_text(part.content)}

    if isinstance(part, CitationPart):
        return {"type": "input_text", "text": parts_to_text((part,))}

    if isinstance(part, ThinkingPart):
        return {"type": "input_text", "text": part.text}

    return {"type": "input_text", "text": getattr(part, "text", "") or ""}


def message_to_openai_input(msg: Message) -> dict[str, Any]:
    return {"role": msg.role, "content": [part_to_openai_input(p) for p in msg.parts]}


def anthropic_source(part: ImagePart | DocumentPart | BinaryPart) -> dict[str, Any]:
    if part.url is not None:
        return {"type": "url", "url": part.url}
    if part.file_id is not None:
        return {"type": "file", "file_id": part.file_id}
    if part.data is not None:
        return {"type": "base64", "media_type": part.media_type, "data": part.data}
    if part.path is not None:
        data = base64.b64encode(part.path.read_bytes()).decode("ascii")
        return {"type": "base64", "media_type": part.media_type, "data": data}
    raise ValueError(f"{part.type} part has no usable source")


def openai_token_logprobs(entries: Any) -> tuple[TokenLogprob, ...]:
    """Map OpenAI-style logprob entries to canonical TokenLogprob tuples.

    Both OpenAI wire dialects share the entry shape (verified live
    2026-09-01): ``{token, logprob, bytes, top_logprobs: [{token, logprob,
    bytes}]}``.  Non-list input and malformed entries yield ().
    """
    if not isinstance(entries, list):
        return ()
    out: list[TokenLogprob] = []
    for entry in entries:
        if not isinstance(entry, dict) or "token" not in entry or "logprob" not in entry:
            continue
        top: list[TopLogprob] = []
        for alt in entry.get("top_logprobs") or []:
            if not isinstance(alt, dict) or "token" not in alt or "logprob" not in alt:
                continue
            alt_bytes = alt.get("bytes")
            top.append(
                TopLogprob(
                    token=str(alt["token"]),
                    logprob=float(alt["logprob"]),
                    bytes=tuple(alt_bytes) if isinstance(alt_bytes, list) else None,
                )
            )
        entry_bytes = entry.get("bytes")
        out.append(
            TokenLogprob(
                token=str(entry["token"]),
                logprob=float(entry["logprob"]),
                bytes=tuple(entry_bytes) if isinstance(entry_bytes, list) else None,
                top=tuple(top),
            )
        )
    return tuple(out)


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except Exception:
            return {"partial_json": value}
        if isinstance(parsed, dict):
            return parsed
        return {"value": parsed}
    return {}


def iso_utc(value: object) -> str | None:
    """Normalize a provider timestamp (unix epoch or ISO-8601 string) to
    canonical ``YYYY-MM-DDTHH:MM:SSZ``. Returns None when unparseable —
    the raw value stays available in provider_data."""
    import datetime

    try:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, (int, float)):
            dt = datetime.datetime.fromtimestamp(float(value), tz=datetime.timezone.utc)
        elif isinstance(value, str) and value:
            text = value.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            # datetime.fromisoformat rejects >6 fractional digits (Gemini
            # emits nanoseconds); trim the fraction to microseconds.
            import re

            text = re.sub(r"\.(\d{6})\d+", r".\1", text)
            dt = datetime.datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            dt = dt.astimezone(datetime.timezone.utc)
        else:
            return None
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OverflowError, OSError):
        return None


def multipart_form_body(
    *,
    fields: list[tuple[str, str]] | None = None,
    files: list[tuple[str, str, str, bytes]] | None = None,
) -> tuple[str, bytes]:
    """Build a multipart/form-data body (OpenAI and Anthropic file uploads).

    ``files`` entries are ``(field_name, filename, content_type, data)``.
    Returns ``(content_type_header_value, body)``.
    """
    import uuid

    boundary = f"lm15-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields or []:
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        chunks.append(f"{value}\r\n".encode("utf-8"))
    for name, filename, content_type, data in files or []:
        safe_filename = filename.replace('"', "%22")
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{safe_filename}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(data)
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def multipart_related_body(
    *,
    metadata: dict[str, Any],
    media_type: str,
    data: bytes,
) -> tuple[str, bytes]:
    """Build a multipart/related body (Gemini media upload: JSON metadata
    part followed by one media part).  Returns ``(content_type, body)``."""
    import uuid

    boundary = f"lm15-{uuid.uuid4().hex}"
    chunks: list[bytes] = [
        f"--{boundary}\r\n".encode("utf-8"),
        b"Content-Type: application/json; charset=UTF-8\r\n\r\n",
        json_dumps(metadata),
        b"\r\n",
        f"--{boundary}\r\n".encode("utf-8"),
        f"Content-Type: {media_type}\r\n\r\n".encode("utf-8"),
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return f"multipart/related; boundary={boundary}", b"".join(chunks)
