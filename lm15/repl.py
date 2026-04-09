from __future__ import annotations

import atexit
import difflib
import os
import re
import sys
from typing import Callable

from .errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ULMError,
)


_DEBUG_REPL_ERRORS = bool(os.getenv("LM15_DEBUG"))
_ENABLED = False
_PREV_SYS_EXCEPTHOOK: Callable | None = None
_MODEL_ID_CACHE: tuple[str, ...] | None = None


def repl_debug(enabled: bool = True) -> None:
    global _DEBUG_REPL_ERRORS
    _DEBUG_REPL_ERRORS = enabled


def _is_interactive() -> bool:
    if hasattr(sys, "ps1") or bool(getattr(sys.flags, "interactive", 0)):
        return True
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _extract_model_name(message: str) -> str | None:
    patterns = (
        r"requested model '([^']+)' does not exist",
        r"requested model \"([^\"]+)\" does not exist",
        r"model '([^']+)' does not exist",
        r"model \"([^\"]+)\" does not exist",
    )
    for p in patterns:
        m = re.search(p, message, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _known_model_ids() -> tuple[str, ...]:
    from .capabilities import known_models

    models = known_models()
    if models:
        return models

    global _MODEL_ID_CACHE
    if _MODEL_ID_CACHE is not None:
        return _MODEL_ID_CACHE

    if os.getenv("LM15_REPL_FETCH_MODELS", "1") == "0":
        _MODEL_ID_CACHE = ()
        return _MODEL_ID_CACHE

    timeout = float(os.getenv("LM15_REPL_FETCH_MODELS_TIMEOUT", "1.0"))
    try:
        from .discovery import models

        specs = models(live=True, timeout=timeout)
        _MODEL_ID_CACHE = tuple(sorted({s.id for s in specs if s.id}))
    except Exception:
        _MODEL_ID_CACHE = ()
    return _MODEL_ID_CACHE


def _suggest_models(model_name: str) -> list[str]:
    models = _known_model_ids()
    if not models:
        return []
    return difflib.get_close_matches(model_name, models, n=3, cutoff=0.45)


def format_lm15_error(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc) or name

    lines = [f"LM15 {name}", msg]

    if isinstance(exc, AuthError):
        lines.append("Tip: verify API key env vars (OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY).")
    elif isinstance(exc, BillingError):
        lines.append("Tip: check billing/quota in your provider dashboard.")
    elif isinstance(exc, RateLimitError):
        lines.append("Tip: retry with backoff and/or reduce request rate.")
    elif isinstance(exc, ContextLengthError):
        lines.append("Tip: reduce prompt/context length or max output tokens.")
    elif isinstance(exc, TimeoutError):
        lines.append("Tip: increase timeout and reduce prompt size.")
    elif isinstance(exc, InvalidRequestError):
        model_name = _extract_model_name(msg)
        if model_name:
            suggestions = _suggest_models(model_name)
            if suggestions:
                lines.append("Did you mean: " + ", ".join(suggestions))
            else:
                lines.append("Tip: run with hydrated model specs for better suggestions.")

    if isinstance(exc, ServerError):
        lines.append("Tip: transient provider error; retry shortly.")

    return "\n".join(lines)


def _sys_excepthook(exc_type, exc, tb):
    if _DEBUG_REPL_ERRORS or not isinstance(exc, ULMError):
        assert _PREV_SYS_EXCEPTHOOK is not None
        return _PREV_SYS_EXCEPTHOOK(exc_type, exc, tb)
    print(format_lm15_error(exc), file=sys.stderr)


def _install_sys_hook() -> None:
    global _PREV_SYS_EXCEPTHOOK
    if _PREV_SYS_EXCEPTHOOK is not None:
        return
    _PREV_SYS_EXCEPTHOOK = sys.excepthook
    sys.excepthook = _sys_excepthook


def _uninstall_sys_hook() -> None:
    global _PREV_SYS_EXCEPTHOOK
    if _PREV_SYS_EXCEPTHOOK is None:
        return
    sys.excepthook = _PREV_SYS_EXCEPTHOOK
    _PREV_SYS_EXCEPTHOOK = None


def _install_ipython_hook() -> None:
    try:
        from IPython import get_ipython  # type: ignore
    except Exception:
        return

    ip = get_ipython()
    if ip is None:
        return

    def _custom(shell, etype, evalue, tb, tb_offset=None):
        if _DEBUG_REPL_ERRORS or not isinstance(evalue, ULMError):
            return shell.showtraceback((etype, evalue, tb), tb_offset=tb_offset)
        print(format_lm15_error(evalue), file=sys.stderr)
        return []

    try:
        ip.set_custom_exc((ULMError,), _custom)
    except Exception:
        return


def enable_repl_errors() -> None:
    global _ENABLED
    if _ENABLED:
        return
    if os.getenv("LM15_REPL_ERRORS", "1") == "0":
        return
    if not _is_interactive():
        return
    _install_sys_hook()
    _install_ipython_hook()
    _ENABLED = True
    atexit.register(_uninstall_sys_hook)
