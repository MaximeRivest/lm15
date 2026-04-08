#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
python3 "$ROOT/completeness/runner.py" --mode fixture --fail-under 1.0
