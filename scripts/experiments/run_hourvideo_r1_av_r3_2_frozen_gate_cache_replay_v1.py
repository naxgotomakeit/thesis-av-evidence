from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_frozen_gate_cache_replay_v1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/hourvideo_r1_av_r3_2_frozen_gate_cache_replay_v1.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(ROOT, args.config, allow_api_calls=args.live), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
