from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1.core import run


if __name__ == "__main__":
    print(run(ROOT, ROOT / "configs/experiments/r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1.json"))

