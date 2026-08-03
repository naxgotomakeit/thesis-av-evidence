from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.r3_2_global_stage1_clean_navigation_projection.core import run


if __name__ == "__main__":
    output = ROOT / "outputs/experiments/r3_2_global_stage1_clean_navigation_projection_v1"
    print(json.dumps(run(ROOT, output), ensure_ascii=False, indent=2))
