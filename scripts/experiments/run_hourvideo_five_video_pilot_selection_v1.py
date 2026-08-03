from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.hourvideo_five_video_pilot_selection_v1 import run


if __name__ == "__main__":
    result = run(
        ROOT,
        ROOT / "configs/experiments/hourvideo_five_video_pilot_selection_v1.json",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
