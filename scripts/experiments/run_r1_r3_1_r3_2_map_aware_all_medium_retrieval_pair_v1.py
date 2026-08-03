from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair.core import preflight, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    if args.preflight:
        result = preflight(ROOT)
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for 18 Planner calls")
        result = run(ROOT, ROOT / f"outputs/experiments/r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1", api_key)
    print(json.dumps(result, ensure_ascii=False, indent=2))

