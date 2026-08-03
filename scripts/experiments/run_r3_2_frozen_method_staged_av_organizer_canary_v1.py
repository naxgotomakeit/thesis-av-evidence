from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.r3_2_frozen_method_staged_av_organizer.core import prepare, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-api-preflight", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    config = ROOT / "configs/experiments/r3_2_frozen_method_staged_av_organizer_canary_v1.json"
    output = ROOT / "outputs/experiments/r3_2_frozen_method_staged_av_organizer_canary_v1"
    result = prepare(ROOT, config, output)["tests"] if args.no_api_preflight else run(
        ROOT, config, output, os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
