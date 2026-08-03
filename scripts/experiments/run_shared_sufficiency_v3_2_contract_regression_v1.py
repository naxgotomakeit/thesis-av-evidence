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
from experiments.shared_sufficiency_v3_2_contract import run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/shared_sufficiency_v3_2_contract_regression_v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/shared_sufficiency_v3_2_contract_regression_v1")
    parser.add_argument("--no-api-preflight", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse validated per-question raw responses already persisted in the isolated output root.")
    args = parser.parse_args()
    api_key = None
    if not args.no_api_preflight:
        load_dotenv(ROOT / ".env", override=False)
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or None
    result = run(ROOT, args.config.resolve(), args.output.resolve(), api_key, live=not args.no_api_preflight, resume=args.resume)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
