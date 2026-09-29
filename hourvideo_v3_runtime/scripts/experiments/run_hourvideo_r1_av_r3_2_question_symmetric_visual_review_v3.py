from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from experiments.hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.core import evaluate, preflight, run_live


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.json")
    parser.add_argument("--stage", choices=["preflight", "live", "evaluate"], required=True)
    args = parser.parse_args(); config = ROOT / args.config
    result = preflight(ROOT, config) if args.stage == "preflight" else run_live(ROOT, config) if args.stage == "live" else evaluate(ROOT, config)
    print(json.dumps({"stage": args.stage, "result": result}, ensure_ascii=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
