from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.local_prepare import prepare_sources, run_audio


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/hourvideo_r1_av_r3_2_single_video_smoke_v1.json")
    parser.add_argument("--stage", choices=["prepare", "audio", "detector", "captions", "live", "evaluate"], required=True)
    args = parser.parse_args()
    cfg = load_json(ROOT / args.config)
    out = ROOT / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare_sources(ROOT, cfg, out)
    elif args.stage == "audio":
        run_audio(cfg, out)
    elif args.stage == "detector":
        env = os.environ.copy()
        env["YOLO_CONFIG_DIR"] = str((out / "ultralytics_runtime").resolve())
        command = [
            cfg["detector"]["python"], "-m", "experiments.hourvideo_r1_av_r3_2_single_video_smoke.detector_runner",
            "--job", str(out / "detector_job.json"), "--output", str(out),
        ]
        completed = subprocess.run(command, cwd=ROOT, env={**env, "PYTHONPATH": str(ROOT / "src")})
        if completed.returncode:
            raise SystemExit(completed.returncode)
    elif args.stage == "captions":
        command = [sys.executable, "-m", "experiments.hourvideo_r1_av_r3_2_single_video_smoke.caption_runner", "--config", args.config]
        completed = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if completed.returncode:
            raise SystemExit(completed.returncode)
    elif args.stage == "live":
        from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import run_live
        run_live(ROOT, cfg, out)
    else:
        from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import evaluate
        evaluate(ROOT, cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
