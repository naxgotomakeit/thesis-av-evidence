from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.egopolice_r1_av_dual_channel_retrieval_smoke_v1.core import preflight, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    if args.preflight:
        result = preflight(ROOT)
    else:
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for six Planner calls")
        result = run(ROOT, ROOT / "outputs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1", key)
    print(json.dumps(result, ensure_ascii=False, indent=2))

