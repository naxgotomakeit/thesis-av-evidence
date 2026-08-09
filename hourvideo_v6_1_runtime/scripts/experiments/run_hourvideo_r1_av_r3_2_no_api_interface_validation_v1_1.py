from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1 import run


if __name__ == "__main__":
    result = run(
        ROOT,
        ROOT
        / "configs/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.json",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
