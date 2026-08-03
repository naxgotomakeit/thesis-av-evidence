from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r1_av_structural_audio_timeline_v1.core import run


if __name__ == "__main__":
    print(json.dumps(run(ROOT, ROOT / "outputs/experiments/r1_av_structural_audio_timeline_v1"), ensure_ascii=False, indent=2))

