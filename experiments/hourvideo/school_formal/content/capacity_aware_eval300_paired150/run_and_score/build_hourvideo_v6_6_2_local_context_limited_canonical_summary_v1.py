#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1.canonical_summary import (  # noqa: E402
    build_manifest,
    build_scored,
    build_structural,
    freeze_raw_outputs,
    run_all,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1.json")
    parser.add_argument("--phase", choices=("freeze", "structural", "score", "manifest", "all"), default="all")
    args = parser.parse_args()
    config = ROOT / args.config
    actions = {
        "freeze": freeze_raw_outputs,
        "structural": build_structural,
        "score": build_scored,
        "manifest": build_manifest,
        "all": run_all,
    }
    result = actions[args.phase](ROOT, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
