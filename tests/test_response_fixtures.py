from __future__ import annotations

from pathlib import Path

import pytest

from lm15.result import materialize_response
from lm15.types import (
    CitationPart,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)

from conformance.response_fixtures import (
    CASES_ROOT,
    assert_expect_lm15,
    assert_no_stream_errors,
    body_path_for_expect,
    is_stream_case,
    iter_cases_with_expect_lm15,
    latest_body_path,
    load_body,
    load_case,
    materialize_stream_fixture,
    parse_complete_fixture,
    parse_openai_fixture,
    parse_stream_fixture,
    request_from_case,
)


TARGET_COMPLETE_FIXTURES = [
    ("openai", "web_search"),
    ("openai", "tools"),
    ("openai", "multi_turn_tool_result"),
    ("openai", "reasoning"),
    ("openai", "code_interpreter"),
    ("anthropic", "basic_text"),
    ("anthropic", "tools"),
    ("anthropic", "thinking"),
    ("anthropic", "web_search"),
    ("gemini", "basic_text"),
    ("gemini", "tools"),
    ("gemini", "google_search"),
    ("gemini", "code_execution"),
]

TARGET_STREAM_FIXTURES = [
    ("openai", "streaming"),
    ("anthropic", "streaming"),
    ("gemini", "streaming"),
]

ALL_EXPECT_COMPLETE = [
    (provider, feature)
    for provider, feature, case in iter_cases_with_expect_lm15()
    if not is_stream_case(case)
]

ALL_EXPECT_STREAM = [
    (provider, feature)
    for provider, feature, case in iter_cases_with_expect_lm15()
    if is_stream_case(case)
]


def test_openai_fixtures_use_output_text_for_assistant_history() -> None:
    bad: list[str] = []
    for path in sorted((CASES_ROOT / "openai").glob("*.json")):
        case = load_case("openai", path.stem)
        body = case.get("request", {}).get("body", {})
        for item_index, item in enumerate(body.get("input", []) or []):
            if not isinstance(item, dict) or item.get("role") != "assistant":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_index, part in enumerate(content):
                part_type = part.get("type") if isinstance(part, dict) else None
                if part_type != "output_text" and part_type != "refusal":
                    bad.append(f"{path.name}: input[{item_index}].content[{content_index}].type={part_type!r}")

    assert not bad, "OpenAI assistant history content must use output_text/refusal, not input_*: " + "; ".join(bad)


def _skip_if_no_body(provider: str, feature: str) -> Path:
    body_path = latest_body_path(provider, feature)
    if body_path is None:
        pytest.skip(f"No saved response body for {provider}.{feature}")
    return body_path


def _body_for_expect(provider: str, feature: str, *, stream: bool) -> Path:
    case = load_case(provider, feature)
    body_path = body_path_for_expect(provider, feature, case["expect_lm15"], stream=stream)
    if body_path is None:
        pytest.skip(f"No saved response body for {provider}.{feature}")
    return body_path


@pytest.mark.parametrize(("provider", "feature"), TARGET_COMPLETE_FIXTURES)
def test_targeted_complete_response_fixtures_parse(provider: str, feature: str) -> None:
    case = load_case(provider, feature)
    body_path = (
        _body_for_expect(provider, feature, stream=False)
        if "expect_lm15" in case
        else _skip_if_no_body(provider, feature)
    )
    response = parse_complete_fixture(provider, feature, body_path=body_path)

    assert isinstance(response, Response)
    if "expect_lm15" in case:
        assert_expect_lm15(response, case["expect_lm15"])


@pytest.mark.parametrize(("provider", "feature"), ALL_EXPECT_COMPLETE)
def test_expect_lm15_complete_response_fixtures(provider: str, feature: str) -> None:
    case = load_case(provider, feature)
    body_path = _body_for_expect(provider, feature, stream=False)

    response = parse_complete_fixture(provider, feature, body_path=body_path)

    assert isinstance(response, Response)
    assert_expect_lm15(response, case["expect_lm15"])


def test_openai_web_search_fixture_parses_citations() -> None:
    response = parse_openai_fixture("openai.web_search/2026-04-13T14-44-39Z.txt")

    assert response.message.parts_of(CitationPart)
    assert response.tool_calls == []


def test_openai_function_call_fixture_parses_tool_call() -> None:
    body_path = _body_for_expect("openai", "tools", stream=False)

    response = parse_complete_fixture("openai", "tools", body_path=body_path)

    tool_calls = response.message.parts_of(ToolCallPart)
    assert len(tool_calls) == 1
    assert tool_calls[0].name == "get_weather"
    assert response.finish_reason == "tool_call"


@pytest.mark.parametrize(("provider", "feature"), TARGET_STREAM_FIXTURES)
def test_targeted_stream_response_fixtures_parse_and_materialize(provider: str, feature: str) -> None:
    body_path = _skip_if_no_body(provider, feature)
    case = load_case(provider, feature)
    request = request_from_case(case)
    events = parse_stream_fixture(provider, feature, body_path=body_path)

    assert_no_stream_errors(events)
    assert any(isinstance(e, StreamDeltaEvent) for e in events)
    assert any(isinstance(e, StreamEndEvent) for e in events)

    response = materialize_response(iter(events), request)
    assert response.message.parts_of(TextPart)
    assert response.text


@pytest.mark.parametrize(("provider", "feature"), ALL_EXPECT_STREAM)
def test_expect_lm15_stream_response_fixtures(provider: str, feature: str) -> None:
    body_path = _body_for_expect(provider, feature, stream=True)
    case = load_case(provider, feature)

    response = materialize_stream_fixture(provider, feature, body_path=body_path)

    assert_expect_lm15(response, case["expect_lm15"])


def test_anthropic_thinking_fixture_parses_thinking_part() -> None:
    body_path = _body_for_expect("anthropic", "thinking", stream=False)

    response = parse_complete_fixture("anthropic", "thinking", body_path=body_path)

    assert response.message.parts_of(ThinkingPart)
    assert response.message.parts_of(TextPart)


def test_gemini_google_search_fixture_parses_citations() -> None:
    body_path = _body_for_expect("gemini", "google_search", stream=False)

    response = parse_complete_fixture("gemini", "google_search", body_path=body_path)

    assert response.message.parts_of(TextPart)
    assert response.message.parts_of(CitationPart)


def test_load_body_reads_fixture_bytes() -> None:
    body_path = _skip_if_no_body("openai", "web_search")

    assert load_body(body_path).startswith(b"{")
