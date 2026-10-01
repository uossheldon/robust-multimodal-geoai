from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.smoke import run_smoke
from src.training.train_s2_deeplab import train_full


def main() -> None:
    if "--smoke" in sys.argv:
        result = run_smoke(PROJECT_ROOT)
        output = PROJECT_ROOT / "results" / "s2_deeplab_smoke.json"
    else:
        result = train_full(PROJECT_ROOT)
        output = PROJECT_ROOT / "results" / "s2_deeplab" / "train_summary.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

