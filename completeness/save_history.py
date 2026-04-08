from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = ROOT / "completeness" / "report.json"
history_dir = ROOT / "completeness" / "history"
history_dir.mkdir(parents=True, exist_ok=True)

data = json.loads(report.read_text())
ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
out = history_dir / f"{ts}.json"
out.write_text(json.dumps(data, indent=2) + "\n")
print(out)
