from __future__ import annotations

import pytest

from conformance.check_doc_drift import DOC_ROOT, load_features, report_provider
from conformance.check_endpoint_fixtures import iter_cases as iter_endpoint_cases, run_case as run_endpoint_case
from conformance.check_error_fixtures import check_case as check_error_case, load_cases as load_error_cases
from conformance.check_request_fixtures import compare_case as compare_request_case, load_logical_cases
from conformance.check_response_fixtures import check_case as check_response_case
from conformance.check_serde_fixtures import FIXTURE_PATH as SERDE_VECTOR, check_json_case, load_cases as load_serde_cases
from conformance.response_fixtures import iter_cases_with_expect_lm15


def test_provider_request_fixtures_match_lm15_python_output() -> None:
    failures = [
        result
        for result in (compare_request_case(case) for case in load_logical_cases())
        if result.status != "pass"
    ]
    assert not failures, [
        {"id": result.case_id, "status": result.status, "reason": result.reason}
        for result in failures
    ]


def test_response_fixtures_satisfy_expect_lm15() -> None:
    failures = [
        result
        for result in (check_response_case(provider, feature, case) for provider, feature, case in iter_cases_with_expect_lm15())
        if result.status != "pass"
    ]
    assert not failures, [
        {"id": result.case_id, "status": result.status, "reason": result.reason}
        for result in failures
    ]


def test_error_fixtures_map_to_structured_lm15_errors() -> None:
    failures = [
        result
        for result in (check_error_case(case) for case in load_error_cases())
        if result.status != "pass"
    ]
    assert not failures, [
        {"id": result.case_id, "status": result.status, "reason": result.reason}
        for result in failures
    ]


def test_non_chat_endpoint_round_trips_match_provider_shapes() -> None:
    failures = [
        result
        for result in (run_endpoint_case(name, fn) for name, fn in iter_endpoint_cases())
        if result.status != "pass"
    ]
    assert not failures, [
        {"id": result.case_id, "status": result.status, "reason": result.reason}
        for result in failures
    ]


@pytest.mark.skipif(not SERDE_VECTOR.exists(), reason="lm15-contract corpus not checked out")
def test_serde_fixtures_round_trip_through_json() -> None:
    failures = []
    for case in load_serde_cases():
        json_result, _obj = check_json_case(case)
        if json_result.status != "pass":
            failures.append(json_result)
    assert not failures, [
        {"id": result.case_id, "kind": result.kind, "check": result.check, "status": result.status, "reason": result.reason}
        for result in failures
    ]


@pytest.mark.skipif(not DOC_ROOT.exists(), reason="curl-fixtures scrape directory not checked out")
def test_provider_doc_drift_has_no_unmapped_parameters() -> None:
    features = load_features()
    reports = [report_provider(provider, features) for provider in ("openai", "anthropic", "gemini")]
    drift = {report.provider: report.unmapped_params for report in reports if report.unmapped_params}
    assert not drift, drift
