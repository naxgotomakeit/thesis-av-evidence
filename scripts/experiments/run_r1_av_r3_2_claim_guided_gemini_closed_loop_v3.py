from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r1_av_r3_2_claim_guided_gemini_closed_loop_v3 import revalidate_existing, run


if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-api-calls", action="store_true")
    parser.add_argument("--revalidate-existing", action="store_true")
    args = parser.parse_args()
    config = ROOT / "configs/experiments/r1_av_r3_2_claim_guided_gemini_closed_loop_v3.json"
    result = (revalidate_existing(ROOT, config) if args.revalidate_existing else
              run(ROOT, config, allow_api_calls=args.allow_api_calls))
    print(json.dumps(result, ensure_ascii=False, indent=2))
