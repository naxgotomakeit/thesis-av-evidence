#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.diagnostics.b0.yki08_preflight import prepare_preflight, run


def main() -> int:
    thesis_root = ROOT.parents[1]
    parser = argparse.ArgumentParser(description="Run non-formal frozen-B0 YKI08 preflight")
    parser.add_argument("--data-root", type=Path, default=thesis_root / "data/EgoPolice_1.0.0")
    parser.add_argument("--model-path", type=Path, default=thesis_root / "models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--config", type=Path, default=ROOT / "config/baselines/egopolice_b0.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "config/data/egopolice_ablation_questions_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiments/B0/preflight_YKI08")
    parser.add_argument("--ffmpeg-path", default="ffmpeg")
    parser.add_argument("--ffprobe-path", default="ffprobe")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        payload = prepare_preflight(
            data_root=args.data_root,
            model_path=args.model_path,
            config_path=args.config,
            question_manifest_path=args.questions,
            ffprobe_path=args.ffprobe_path,
        )
    else:
        payload = run(
            data_root=args.data_root,
            model_path=args.model_path,
            config_path=args.config,
            question_manifest_path=args.questions,
            output_dir=args.output_dir,
            ffmpeg_path=args.ffmpeg_path,
            ffprobe_path=args.ffprobe_path,
        )
    summary = {
        "label": payload["label"],
        "formal_result": payload["formal_result"],
        "question_count": payload.get("aggregate", {}).get("overall", {}).get(
            "total", payload.get("question_count")
        ),
        "accuracy": payload.get("aggregate", {}).get("overall", {}).get("accuracy"),
        "gt_interval_hit_at_8": payload.get("aggregate", {}).get("gt_interval_hit_at_8"),
        "output_dir": str(args.output_dir) if not args.preflight_only else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

