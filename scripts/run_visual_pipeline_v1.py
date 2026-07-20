"""Entry point for the exact selected visual pipeline v1.

The production orchestration functions live in ``src.thesis_av.visual.pipeline``.
This command exposes the strict frozen-artifact fidelity replay without using a
count fixture or rerunning Qwen.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/visual_pipeline_v1.json",
    )
    parser.add_argument(
        "--validate-fidelity",
        action="store_true",
        help="Replay all selected local stages against the frozen long4 artifacts.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=ROOT / "outputs/visual_pipeline_v1_fidelity",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.validate_fidelity:
        command = [
            sys.executable,
            str(ROOT / "scripts/validate_visual_pipeline_v1_fidelity.py"),
            "--output",
            str(args.validation_output),
        ]
        return subprocess.run(command, cwd=ROOT, check=False).returncode
    print(
        json.dumps(
            {
                "pipeline": config["pipeline_id"],
                "production_entry": "src.thesis_av.visual.pipeline",
                "validation_command": (
                    "python scripts/run_visual_pipeline_v1.py --validate-fidelity"
                ),
                "note": (
                    "Raw-video, keyframe, caption, grouping, summary, and story "
                    "orchestration is exposed by the exact production functions."
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
