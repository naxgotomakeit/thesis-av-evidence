from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair.core import finalize_existing, preflight, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    if args.finalize_existing:
        result = finalize_existing(ROOT / "outputs/experiments/r1_r3_v2_map_aware_all_medium_retrieval_pair_v1")
    elif args.preflight:
        result = preflight(ROOT)
    else:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for the 12 Planner calls")
        result = run(ROOT, ROOT / "outputs/experiments/r1_r3_v2_map_aware_all_medium_retrieval_pair_v1", key)
    print(json.dumps(result, ensure_ascii=False, indent=2))
