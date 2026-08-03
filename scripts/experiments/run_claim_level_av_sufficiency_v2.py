from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.experiments.claim_level_av_sufficiency_v2 import run_experiment


def main() -> None:
    load_dotenv(REPO / ".env", override=False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-api-calls", action="store_true",
        help="Required for the six formal Haiku calls; otherwise performs dry-run only.",
    )
    args = parser.parse_args()
    result = run_experiment(REPO, allow_api_calls=args.allow_api_calls)
    print(result)


if __name__ == "__main__":
    main()
