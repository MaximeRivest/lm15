from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import UniversalLM, discover_provider_entry_points, load_plugins


def main() -> None:
    eps = discover_provider_entry_points()
    print("entry_points:", [e.name for e in eps])

    lm = UniversalLM()
    result = load_plugins(lm)
    print("loaded:", result.loaded)
    print("failed:", result.failed)


if __name__ == "__main__":
    main()
