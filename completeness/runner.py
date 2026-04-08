from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(slots=True)
class TestRecord:
    id: str
    provider: str
    probe: str
    required: bool
    mode: str
    status: str
    details: str


@dataclass(slots=True)
class Summary:
    required_total: int
    required_passed: int
    required_failed: int
    required_skipped: int
    score: float


def _load_matrix(path: Path) -> list[dict]:
    return json.loads(path.read_text())["tests"]


def _run_test(test: dict) -> TestRecord:
    try:
        module = importlib.import_module(f"completeness.probes.{test['probe']}")
        result = module.run(test, ROOT)
        status = result.status
        details = result.details
    except Exception as e:  # pragma: no cover - harness should never crash on one probe
        status = "fail"
        details = f"exception: {type(e).__name__}: {e}"

    return TestRecord(
        id=test["id"],
        provider=test["provider"],
        probe=test["probe"],
        required=bool(test.get("required", True)),
        mode=test.get("mode", "fixture"),
        status=status,
        details=details,
    )


def _summarize(records: list[TestRecord]) -> Summary:
    required = [r for r in records if r.required]
    total = len(required)
    passed = sum(1 for r in required if r.status == "pass")
    failed = sum(1 for r in required if r.status == "fail")
    skipped = sum(1 for r in required if r.status == "skip")
    denom = max(1, total - skipped)
    score = passed / denom
    return Summary(required_total=total, required_passed=passed, required_failed=failed, required_skipped=skipped, score=score)


def _provider_breakdown(records: list[TestRecord]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in records:
        b = out.setdefault(r.provider, {"pass": 0, "fail": 0, "skip": 0})
        b[r.status] += 1
    return out


def _write_report_md(path: Path, records: list[TestRecord], summary: Summary, providers: dict[str, dict[str, int]]) -> None:
    lines = [
        "# LM15 Completeness Report",
        "",
        f"- Required total: {summary.required_total}",
        f"- Required passed: {summary.required_passed}",
        f"- Required failed: {summary.required_failed}",
        f"- Required skipped: {summary.required_skipped}",
        f"- Score: {summary.score:.3f}",
        "",
        "## Per provider",
        "",
        "| provider | pass | fail | skip |",
        "|---|---:|---:|---:|",
    ]
    for p, b in sorted(providers.items()):
        lines.append(f"| {p} | {b['pass']} | {b['fail']} | {b['skip']} |")

    lines += ["", "## Tests", "", "| id | provider | probe | required | status | details |", "|---|---|---|---:|---|---|"]
    for r in records:
        lines.append(f"| {r.id} | {r.provider} | {r.probe} | {str(r.required).lower()} | {r.status} | {r.details} |")
    path.write_text("\n".join(lines) + "\n")


def run(mode: str, fail_under: float, json_out: Path, md_out: Path) -> int:
    tests = _load_matrix(ROOT / "completeness" / "spec_matrix.json")
    if mode in {"live", "all"}:
        tests += _load_matrix(ROOT / "completeness" / "live_matrix.json")

    if mode != "all":
        tests = [t for t in tests if t.get("mode") == mode]

    records: list[TestRecord] = []
    for t in tests:
        records.append(_run_test(t))

    summary = _summarize(records)
    providers = _provider_breakdown(records)

    report = {
        "summary": asdict(summary),
        "providers": providers,
        "tests": [asdict(r) for r in records],
    }
    json_out.write_text(json.dumps(report, indent=2) + "\n")
    _write_report_md(md_out, records, summary, providers)

    print(f"required_total={summary.required_total}")
    print(f"required_passed={summary.required_passed}")
    print(f"required_failed={summary.required_failed}")
    print(f"required_skipped={summary.required_skipped}")
    print(f"score={summary.score:.3f}")

    if summary.score < fail_under:
        return 2
    if summary.required_failed > 0:
        return 3
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="LM15 completeness harness")
    parser.add_argument("--mode", choices=["fixture", "live", "all"], default="fixture")
    parser.add_argument("--fail-under", type=float, default=1.0)
    parser.add_argument("--json-out", default=str(ROOT / "completeness" / "report.json"))
    parser.add_argument("--md-out", default=str(ROOT / "completeness" / "report.md"))
    args = parser.parse_args()

    code = run(mode=args.mode, fail_under=args.fail_under, json_out=Path(args.json_out), md_out=Path(args.md_out))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
