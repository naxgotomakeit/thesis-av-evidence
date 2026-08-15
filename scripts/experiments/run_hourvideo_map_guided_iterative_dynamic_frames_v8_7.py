from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_map_guided_iterative_dynamic_frames_v8_7.core import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/hourvideo_map_guided_iterative_dynamic_frames_v8_7_two_case_canary.json",
    )
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--confirm-two-questions", action="store_true")
    args = parser.parse_args()
    if args.execute_api and not args.confirm_two_questions:
        raise SystemExit("live V8.7 requires --confirm-two-questions")
    result = run(ROOT, ROOT / args.config, execute_api=args.execute_api)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
