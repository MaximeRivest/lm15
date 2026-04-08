from __future__ import annotations

import statistics
import subprocess
import sys


def bench(module: str, runs: int = 30):
    vals = []
    for _ in range(runs):
        p = subprocess.run(
            [sys.executable, "-c", f"import time; t=time.perf_counter(); import {module}; print((time.perf_counter()-t)*1000)"],
            capture_output=True,
            text=True,
            check=True,
        )
        vals.append(float(p.stdout.strip()))
    vals.sort()
    return {
        "median_ms": statistics.median(vals),
        "p95_ms": vals[int(0.95 * (len(vals) - 1))],
        "min_ms": vals[0],
        "max_ms": vals[-1],
    }


if __name__ == "__main__":
    print({"lm15": bench("lm15")})
