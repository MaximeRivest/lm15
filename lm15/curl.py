"""
Dump lm15 requests as curl commands for cross-SDK testing.

Usage:
    import lm15
    from lm15.curl import dump_curl, dump_http

    # Get a curl command string
    print(dump_curl("gpt-4.1-mini", "Hello.", env=".env"))

    # Get the structured HTTP request (for JSON diffing across SDKs)
    print(dump_http("gpt-4.1-mini", "Hello.", env=".env"))
"""

from __future__ import annotations

import json
import shlex
from typing import Any

from .capabilities import resolve_provider
from .factory import build_default
from .model import callable_to_tool
from .transports.base import HttpRequest
from .types import (
    BuiltinTool,
    Config,
    FunctionTool,
    LMRequest,
    Message,
    Part,
    Tool,
)


def _normalize_tools(tools: list[Tool | Any | str] | None) -> tuple[Tool, ...]:
    if not tools:
        return ()
    out: list[Tool] = []
    for t in tools:
        if isinstance(t, Tool):
            out.append(t)
        elif isinstance(t, str):
            out.append(BuiltinTool(name=t))
        elif callable(t):
            inferred = callable_to_tool(t)
            out.append(
                FunctionTool(
                    name=inferred.name,
                    description=inferred.description,
                    parameters=inferred.parameters,
                )
            )
    return tuple(out)


def _build_lm_request(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages: list[Message] | None = None,
    system: str | None = None,
    tools: list | None = None,
    reasoning: bool | dict | None = None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop: list[str] | None = None,
) -> LMRequest:
    """Build an LMRequest from high-level parameters."""
    if messages is not None:
        final_messages = tuple(messages)
    elif prompt is not None:
        if isinstance(prompt, str):
            turn = (Message.user(prompt),)
        else:
            parts = [Part.text_part(item) if isinstance(item, str) else item for item in prompt]
            turn = (Message(role="user", parts=tuple(parts)),)
        if prefill:
            turn = turn + (Message.assistant(prefill),)
        final_messages = turn
    else:
        raise ValueError("either prompt or messages is required")

    tool_defs = _normalize_tools(tools)

    provider_cfg: dict[str, Any] = {}
    if prompt_caching:
        provider_cfg["prompt_caching"] = True
    if output == "image":
        provider_cfg["output"] = "image"
    elif output == "audio":
        provider_cfg["output"] = "audio"

    reasoning_cfg: dict[str, Any] | None = None
    if reasoning is True:
        reasoning_cfg = {"enabled": True}
    elif isinstance(reasoning, dict):
        reasoning_cfg = {"enabled": True, **reasoning}

    cfg = Config(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=tuple(stop or ()),
        reasoning=reasoning_cfg,
        provider=provider_cfg or None,
    )

    return LMRequest(
        model=model,
        messages=final_messages,
        system=system,
        tools=tool_defs,
        config=cfg,
    )


def build_http_request(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    stream: bool = False,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
    **kwargs,
) -> HttpRequest:
    """Build the provider-level HttpRequest without sending it.

    Returns the exact HttpRequest that would be sent to the provider,
    including URL, headers, and JSON body.
    """
    lm_request = _build_lm_request(model, prompt, **kwargs)
    resolved_provider = provider or resolve_provider(model)

    client = build_default(api_key=api_key, provider_hint=resolved_provider, env=env)
    adapter = client.adapters.get(resolved_provider)
    if adapter is None:
        raise ValueError(f"no adapter for provider '{resolved_provider}'")

    return adapter.build_request(lm_request, stream=stream)


def http_request_to_dict(req: HttpRequest) -> dict[str, Any]:
    """Convert an HttpRequest to a JSON-serializable dict.

    The output is suitable for cross-SDK comparison:
    - method, url, headers, params, body (parsed JSON)
    - Auth headers are redacted to "REDACTED" for safe sharing
    """
    body = None
    if req.json_body is not None:
        body = req.json_body
    elif req.body is not None:
        try:
            body = json.loads(req.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = "<binary>"

    # Redact auth
    headers = dict(req.headers)
    for key in ("Authorization", "x-api-key", "x-goog-api-key"):
        if key in headers:
            headers[key] = "REDACTED"

    return {
        "method": req.method,
        "url": req.url,
        "headers": headers,
        "params": req.params if req.params else None,
        "body": body,
    }


def http_request_to_curl(req: HttpRequest, *, redact_auth: bool = True) -> str:
    """Convert an HttpRequest to a curl command string.

    If redact_auth is False, real API keys are included (for actual execution).
    """
    parts = ["curl"]

    # Method
    if req.method != "GET":
        parts.append(f"-X {req.method}")

    # URL with params
    url = req.url
    if req.params:
        from urllib.parse import urlencode
        url = f"{url}?{urlencode(req.params)}"
    parts.append(shlex.quote(url))

    # Headers
    for key, value in req.headers.items():
        if redact_auth and key.lower() in ("authorization", "x-api-key", "x-goog-api-key"):
            value = "REDACTED"
        parts.append(f"-H {shlex.quote(f'{key}: {value}')}")

    # Body
    if req.json_body is not None:
        body_str = json.dumps(req.json_body, indent=2, ensure_ascii=False)
        parts.append(f"-d {shlex.quote(body_str)}")
    elif req.body is not None:
        try:
            parsed = json.loads(req.body)
            body_str = json.dumps(parsed, indent=2, ensure_ascii=False)
            parts.append(f"-d {shlex.quote(body_str)}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            parts.append("--data-binary @-")

    return " \\\n  ".join(parts)


# ── Convenience functions ────────────────────────────────────────


def dump_curl(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    stream: bool = False,
    redact_auth: bool = True,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
    **kwargs,
) -> str:
    """Build a curl command for the given call parameters.

    Example:
        >>> print(lm15.curl.dump_curl("gpt-4.1-mini", "Hello.", env=".env"))
        curl \\
          -X POST \\
          'https://api.openai.com/v1/responses' \\
          -H 'Authorization: REDACTED' \\
          -H 'Content-Type: application/json' \\
          -d '{"model": "gpt-4.1-mini", ...}'
    """
    req = build_http_request(
        model, prompt,
        stream=stream,
        provider=provider,
        api_key=api_key,
        env=env,
        **kwargs,
    )
    return http_request_to_curl(req, redact_auth=redact_auth)


def dump_http(
    model: str,
    prompt: str | list[str | Part] | None = None,
    *,
    stream: bool = False,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Build the HTTP request dict for cross-SDK comparison.

    Example:
        >>> data = lm15.curl.dump_http("gpt-4.1-mini", "Hello.", env=".env")
        >>> print(json.dumps(data, indent=2))
    """
    req = build_http_request(
        model, prompt,
        stream=stream,
        provider=provider,
        api_key=api_key,
        env=env,
        **kwargs,
    )
    return http_request_to_dict(req)
