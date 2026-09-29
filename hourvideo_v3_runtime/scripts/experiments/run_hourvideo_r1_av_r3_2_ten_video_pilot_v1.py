from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.offline import run_stage
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import evaluate, preflight, run_live


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1.json")
    parser.add_argument("--stage", choices=["prepare", "audio", "detector", "captions", "preflight", "live", "evaluate"], required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = ROOT / args.config
    if args.stage == "preflight": result = preflight(ROOT, config)
    elif args.stage == "live": result = run_live(ROOT, config)
    elif args.stage == "evaluate": result = evaluate(ROOT, config)
    else: result = run_stage(ROOT, config, args.stage, resume=not args.no_resume)
    if isinstance(result, dict):
        concise = result.get("validation", result.get("summary", result.get("overall_validation", "completed")))
    else:
        concise = "completed"
    print(json.dumps({"stage": args.stage, "result": concise}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
