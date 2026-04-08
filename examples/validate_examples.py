from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))


def main() -> int:
    for p in EXAMPLES:
        py_compile.compile(str(p), doraise=True)

    env = dict(os.environ)
    env["LM15_EXAMPLES_SKIP_LIVE"] = "1"

    for p in EXAMPLES:
        proc = subprocess.run([sys.executable, str(p)], cwd=str(ROOT), capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            print(f"FAIL {p.name}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
            return 2
        print(f"OK {p.name}: {proc.stdout.strip()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
