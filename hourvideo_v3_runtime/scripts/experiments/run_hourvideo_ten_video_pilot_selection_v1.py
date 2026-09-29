from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_ten_video_pilot_selection_v1.core import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/hourvideo_ten_video_pilot_selection_v1.json")
    args = parser.parse_args()
    print(run(ROOT, ROOT / args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
