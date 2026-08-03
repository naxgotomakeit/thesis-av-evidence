from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.r1_av_r3_2_cached_visual_review_full_closed_loop_v1.core import replay_display_normalization, run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--replay-display-normalization", action="store_true")
    args = parser.parse_args()
    config = ROOT / "configs/experiments/r1_av_r3_2_cached_visual_review_full_closed_loop_v1.json"
    if args.replay_display_normalization:
        print(replay_display_normalization(ROOT, config))
        return
    if args.live:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    result = run(
        ROOT,
        config,
        allow_api_calls=args.live,
    )
    print(result)


if __name__ == "__main__":
    main()
