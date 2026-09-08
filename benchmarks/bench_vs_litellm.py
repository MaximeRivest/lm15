#!/usr/bin/env python
"""lm15 vs litellm, end to end, one Gemini model: install, first import,
cached import, first call, fifth call.

    GEMINI_API_KEY=... .venv/bin/python benchmarks/bench_vs_litellm.py [--runs 5] [--model gemini-3.8-flash]

Method (each number is machine-measured; nothing is typed in):

  install        fresh `uv venv` per package, `uv cache clean` first so the
                 wheel download is part of the cost on both sides; wall
                 time of `uv pip install`; size on disk of site-packages
                 minus an empty venv; transitive distribution count.
                 lm15 installs from the wheel `uv build` just produced.
  first import   the FIRST `python -c "import X"` after install, in a fresh
                 process: no .pyc yet, cold page cache for the package.
  cached import  median of N further `python -c "import X"` runs in fresh
                 processes (.pyc present, page cache warm), minus the same
                 interpreter's `python -c pass` baseline.
  first call     inside one fresh process: import already done, wall time
                 of the first completion (cold TLS, cold connection pool,
                 client construction).
  fifth call     the fifth completion in that same process (warm pool).
                 The whole process is repeated N times; the table shows the
                 median and every run, because the network is in the loop.

Both clients send the same prompt to the same model with the same knobs
(temperature 0, max output tokens 256 — Gemini 3.8 spends 50–120
hidden reasoning tokens on even this prompt, and a 64 cap made some runs
come back empty on both sides; every run records finish reason and
reasoning tokens so equal work is visible, not assumed) and read the text back. Streaming is
not measured here. Results go to benchmarks/results/vs_litellm_<ts>.json
and a markdown table is printed.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_ROOT = Path("/tmp/lm15-vs-litellm")
PROMPT = "Reply with exactly one word: ok"

LM15_CALL = """
import time, sys
t_import0 = time.perf_counter()
import lm15
t_import = time.perf_counter() - t_import0
from lm15 import LMRouter, Request, Message, Config
router = LMRouter()
req = Request(model="gemini:{model}", messages=(Message.user({prompt!r}),), config=Config(temperature=0, max_tokens=256))
times = []
work = []
for i in range(5):
    t0 = time.perf_counter()
    resp = router.complete(req)
    _ = resp.text
    times.append(time.perf_counter() - t0)
    work.append([resp.text, resp.finish_reason, resp.usage.reasoning_tokens, resp.usage.output_tokens])
print(__import__("json").dumps({{"import_s": t_import, "calls_s": times, "text": resp.text, "work": work}}))
"""

LITELLM_CALL = """
import time, sys, os
os.environ.setdefault("LITELLM_LOG", "ERROR")
t_import0 = time.perf_counter()
import litellm
t_import = time.perf_counter() - t_import0
litellm.suppress_debug_info = True
import logging; logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
times = []
work = []
for i in range(5):
    t0 = time.perf_counter()
    resp = litellm.completion(model="gemini/{model}", messages=[{{"role": "user", "content": {prompt!r}}}], temperature=0, max_tokens=256)
    _ = resp.choices[0].message.content
    times.append(time.perf_counter() - t0)
    d = resp.usage.completion_tokens_details
    work.append([resp.choices[0].message.content, resp.choices[0].finish_reason, getattr(d, "reasoning_tokens", None), resp.usage.completion_tokens])
print(__import__("json").dumps({{"import_s": t_import, "calls_s": times, "text": resp.choices[0].message.content, "work": work}}))
"""


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def wall(cmd: list[str], env: dict | None = None) -> float:
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-800:])
    return time.perf_counter() - t0


def du_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def site_packages(venv: Path) -> Path:
    return next((venv / "lib").glob("python*/site-packages"))


def dist_count(python: Path) -> int:
    out = run([str(python), "-c", "import importlib.metadata as m; print(len({d.metadata['Name'] for d in m.distributions()}))"]).stdout
    return int(out.strip())


def make_venv(name: str, target: str) -> tuple[Path, float]:
    path = VENV_ROOT / name
    if path.exists():
        shutil.rmtree(path)
    r = run(["uv", "venv", "--python", sys.executable, str(path)])
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    t = wall(["uv", "pip", "install", "--python", str(path / "bin/python"), target])
    return path, t


def measure_package(name: str, target: str, import_stmt: str, call_code: str, runs: int, model: str, baseline_bytes: int, env: dict) -> dict:
    print(f"\n[{name}] installing {target} ...", file=sys.stderr)
    venv, install_s = make_venv(name, target)
    py = venv / "bin/python"
    size = du_bytes(site_packages(venv)) - baseline_bytes
    deps = dist_count(py)
    print(f"[{name}] install {install_s:.1f}s, {size/2**20:.1f} MiB, {deps} dists", file=sys.stderr)

    # first import: no .pyc yet (uv does not byte-compile by default; verify)
    pyc = list(site_packages(venv).rglob("*.pyc"))
    first_import_s = wall([str(py), "-c", import_stmt])
    interpreter = statistics.median(wall([str(py), "-c", "pass"]) for _ in range(runs * 2))
    cached = [wall([str(py), "-c", import_stmt]) for _ in range(runs * 2)]
    print(f"[{name}] first import {first_import_s*1000:.0f} ms (pyc before: {len(pyc)}), cached median {statistics.median(cached)*1000:.0f} ms, interpreter {interpreter*1000:.0f} ms", file=sys.stderr)

    calls: list[dict] = []
    for i in range(runs):
        r = run([str(py), "-c", call_code.format(model=model, prompt=PROMPT)], env=env)
        if r.returncode != 0:
            raise RuntimeError(f"{name} call run {i}: {r.stderr[-1500:]}")
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        calls.append(rec)
        print(f"[{name}] run {i+1}: first call {rec['calls_s'][0]*1000:.0f} ms, 5th {rec['calls_s'][4]*1000:.0f} ms, work={rec['work']}", file=sys.stderr)
    return {
        "install_target": target,
        "install_s": install_s,
        "site_packages_bytes": size,
        "distributions": deps,
        "pyc_files_before_first_import": len(pyc),
        "first_import_s": first_import_s,
        "interpreter_baseline_s": interpreter,
        "cached_import_s": cached,
        "runs": calls,
    }


def med(xs: list[float]) -> float:
    return statistics.median(xs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--model", default="gemini-3.8-flash")
    args = ap.parse_args()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set")
    env = {**os.environ, "GEMINI_API_KEY": key}

    wheels = sorted((REPO / "dist").glob("lm15-*.whl"))
    if not wheels:
        sys.exit("no wheel in dist/: run `uv build` first")
    run(["uv", "cache", "clean"])  # the download is part of the install cost, both sides
    VENV_ROOT.mkdir(parents=True, exist_ok=True)
    base = VENV_ROOT / "_baseline"
    if base.exists():
        shutil.rmtree(base)
    run(["uv", "venv", "--python", sys.executable, str(base)])
    baseline_bytes = du_bytes(site_packages(base))

    results = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "prompt": PROMPT,
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "uv": run(["uv", "--version"]).stdout.strip(),
        "runs": args.runs,
        "packages": {},
    }
    results["packages"]["lm15"] = measure_package("lm15", str(wheels[-1]), "import lm15", LM15_CALL, args.runs, args.model, baseline_bytes, env)
    results["packages"]["litellm"] = measure_package("litellm", "litellm", "import litellm", LITELLM_CALL, args.runs, args.model, baseline_bytes, env)
    # litellm's version, for the record
    results["packages"]["litellm"]["version"] = run([str(VENV_ROOT / "litellm/bin/python"), "-c", "import importlib.metadata as m; print(m.version('litellm'))"]).stdout.strip()
    results["packages"]["lm15"]["version"] = run([str(VENV_ROOT / "lm15/bin/python"), "-c", "import importlib.metadata as m; print(m.version('lm15'))"]).stdout.strip()

    out_dir = REPO / "benchmarks" / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = out_dir / f"vs_litellm_{stamp}.json"
    out.write_text(json.dumps(results, indent=2) + "\n")

    p = results["packages"]
    def row(label, f, fmt):
        return f"| {label} | {fmt(f(p['lm15']))} | {fmt(f(p['litellm']))} |"
    ms = lambda s: f"{s*1000:.0f} ms"
    lines = [
        f"lm15 {p['lm15']['version']} vs litellm {p['litellm']['version']} — {args.model} — Python {results['python']} — {results['generated_at']}",
        "",
        "| | lm15 | litellm |",
        "|---|---:|---:|",
        row("install (uv, cold cache)", lambda x: x["install_s"], lambda s: f"{s:.1f} s"),
        row("install size", lambda x: x["site_packages_bytes"], lambda b: f"{b/2**20:.1f} MiB"),
        row("distributions installed", lambda x: x["distributions"], str),
        row("first import (no .pyc)", lambda x: x["first_import_s"], ms),
        row(f"cached import (median of {args.runs*2})", lambda x: med(x["cached_import_s"]), ms),
        row("  interpreter baseline", lambda x: x["interpreter_baseline_s"], ms),
        row(f"first call (median of {args.runs})", lambda x: med([r["calls_s"][0] for r in x["runs"]]), ms),
        row(f"5th call (median of {args.runs})", lambda x: med([r["calls_s"][4] for r in x["runs"]]), ms),
        row("first call, every run", lambda x: [r["calls_s"][0] for r in x["runs"]], lambda xs: ", ".join(f"{s*1000:.0f}" for s in xs)),
        row("5th call, every run", lambda x: [r["calls_s"][4] for r in x["runs"]], lambda xs: ", ".join(f"{s*1000:.0f}" for s in xs)),
        row("answers (text / finish / reasoning tokens), all 25 calls", lambda x: [w for r in x["runs"] for w in r["work"]],
            lambda ws: f"{sum(1 for w in ws if (w[0] or '').strip().lower().rstrip('.') == 'ok')}/{len(ws)} said ok; finish {sorted(set(w[1] for w in ws))}; reasoning tokens {min(w[2] or 0 for w in ws)}–{max(w[2] or 0 for w in ws)}"),
        "",
        f"results: {out.relative_to(REPO)}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
