"""Canonical read-only entry point for EgoPolice AV Organizer V1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.egopolice_av_organizer_v1 import verify_frozen_baseline
from src.experiments.egopolice_av_organizer_v1.config import VIDEO_ID


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default=VIDEO_ID)
    parser.add_argument("--reuse-existing-artifacts", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--fresh-run", action="store_true")
    parser.add_argument("--allow-api-calls", action="store_true")
    args = parser.parse_args()
    if args.video_id != VIDEO_ID:
        parser.error("V1 is frozen only for video 540772226")
    if args.fresh_run:
        if not args.allow_api_calls:
            parser.error("--fresh-run requires explicit --allow-api-calls")
        parser.error(
            "The freeze facade never writes into canonical artifact roots. "
            "Use the manifest's canonical runners with a new experiment ID."
        )
    if args.allow_api_calls:
        parser.error("--allow-api-calls is valid only with --fresh-run")
    if not (args.verify_only and args.reuse_existing_artifacts):
        parser.error("use --reuse-existing-artifacts --verify-only")
    result = verify_frozen_baseline()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
