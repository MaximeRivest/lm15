from __future__ import annotations

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15 import fetch_models_dev, hydrate_with_specs


def main() -> None:
    if os.getenv("LM15_EXAMPLES_SKIP_LIVE") == "1":
        print("SKIP: LM15_EXAMPLES_SKIP_LIVE=1")
        return

    specs = fetch_models_dev()
    hydrate_with_specs(specs)
    print("models:", len(specs))
    print("providers:", len({s.provider for s in specs}))


if __name__ == "__main__":
    main()
