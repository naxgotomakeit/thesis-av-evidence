from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (  # noqa: E402
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import (  # noqa: E402
    _build_r1_map,
)
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.offline import (  # noqa: E402
    audio_case,
)


EXPERIMENT = "hourvideo_dev50_r1_offline_v1"
SOURCE_ROOT = ROOT / "outputs/experiments/hourvideo_dev50_qwen2_5_vl_7b_captions_v1"
OUTPUT_ROOT = ROOT / f"outputs/experiments/{EXPERIMENT}"
PREPARED_ARTIFACTS = (
    "question_input.json",
    "shared_hierarchy.json",
    "medium_siglip.float32.npy",
    "source_artifact_audit.json",
    "detector_job.json",
)


def _copy_immutable(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if sha256_file(target) != sha256_file(source):
            raise RuntimeError(f"existing seeded artifact differs: {target}")
        return
    shutil.copy2(source, target)


def _cases(video_uid: str | None = None) -> list[tuple[dict[str, Any], Path]]:
    rows: list[tuple[dict[str, Any], Path]] = []
    for source_config in sorted((SOURCE_ROOT / "case_configs").glob("*.json")):
        case = copy.deepcopy(load_json(source_config))
        uid = case["video_uid"]
        if video_uid and uid != video_uid:
            continue
        case["experiment"] = EXPERIMENT
        case["output_root"] = f"outputs/experiments/{EXPERIMENT}/cases/{uid}"
        target_config = OUTPUT_ROOT / "case_configs" / f"{uid}.json"
        write_json(target_config, case)
        rows.append((case, target_config))
    rows.sort(key=lambda row: int(row[0]["video_position"]))
    if video_uid and not rows:
        raise RuntimeError(f"unknown dev50 video_uid: {video_uid}")
    return rows


def _seed(cases: list[tuple[dict[str, Any], Path]]) -> None:
    records = []
    for case, _ in cases:
        uid = case["video_uid"]
        source = SOURCE_ROOT / "cases" / uid
        target = ROOT / case["output_root"]
        copied = []
        for name in PREPARED_ARTIFACTS:
            source_path = source / name
            if not source_path.is_file():
                raise FileNotFoundError(f"missing dev50 prepared source: {source_path}")
            target_path = target / name
            _copy_immutable(source_path, target_path)
            copied.append({"name": name, "sha256": sha256_file(target_path)})
        records.append({"video_uid": uid, "source": str(source), "target": str(target), "artifacts": copied})
    manifest = OUTPUT_ROOT / "prepared_artifact_reuse_manifest.json"
    previous = load_json(manifest).get("videos", []) if manifest.is_file() else []
    by_uid = {row["video_uid"]: row for row in previous + records}
    write_json(manifest, {
        "policy": "copy immutable question-independent hierarchy and detector job; no captions reused",
        "videos": [by_uid[uid] for uid in sorted(by_uid)],
    })


def _run_detector(case: dict[str, Any], case_out: Path) -> None:
    validation = case_out / "detector_validation.json"
    if validation.is_file() and load_json(validation).get("valid"):
        return
    ultralytics_runtime = case_out / "ultralytics_runtime"
    ultralytics_runtime.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "YOLO_CONFIG_DIR": str(ultralytics_runtime.resolve()),
    }
    subprocess.run(
        [
            case["detector"]["python"],
            "-m",
            "experiments.hourvideo_r1_av_r3_2_single_video_smoke.detector_runner",
            "--job",
            str(case_out / "detector_job.json"),
            "--output",
            str(case_out),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _build_map(case_out: Path) -> None:
    required = ("shared_hierarchy.json", "r1_medium_projection.json", "audio_asr.json")
    missing = [name for name in required if not (case_out / name).is_file()]
    if missing:
        raise FileNotFoundError(f"R1 map prerequisites missing in {case_out}: {missing}")
    hierarchy = load_json(case_out / "shared_hierarchy.json")
    projections = load_json(case_out / "r1_medium_projection.json")
    audio = load_json(case_out / "audio_asr.json")["segments"]
    embeddings = np.load(case_out / "medium_siglip.float32.npy", allow_pickle=False).astype(np.float32)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    result = _build_r1_map(hierarchy, projections, audio, embeddings)
    write_json(case_out / "r1_av_navigation_map.json", result)
    write_json(case_out / "r1_map_validation.json", {
        "valid": bool(result["coarse_regions"]),
        "map_type": result["map_type"],
        "medium_count": len(hierarchy["medium_nodes"]),
        "coarse_count": len(result["coarse_regions"]),
        "semantic_fields_available": result["semantic_fields_available"],
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["seed", "audio", "detector", "map"], required=True)
    parser.add_argument("--video-uid")
    args = parser.parse_args()

    cases = _cases(args.video_uid)
    _seed(cases)
    write_json(OUTPUT_ROOT / "input_manifest.json", {
        "experiment": EXPERIMENT,
        "scope": "HourVideo dev50 R1 offline",
        "video_count": 50,
        "source_prepare_manifest": str(SOURCE_ROOT / "input_manifest.json"),
        "source_prepare_manifest_sha256": sha256_file(SOURCE_ROOT / "input_manifest.json"),
        "detector": cases[0][0]["detector"],
        "captions_used": False,
        "videoseal_outputs_used": False,
        "external_api_calls": 0,
    })

    progress_path = OUTPUT_ROOT / f"{args.stage}_progress.json"
    progress = load_json(progress_path) if progress_path.is_file() else {}
    for case, _ in cases:
        uid = case["video_uid"]
        case_out = ROOT / case["output_root"]
        started = time.perf_counter()
        if args.stage == "audio" and not (case_out / "audio_asr.json").is_file():
            audio_case(case, case_out)
        elif args.stage == "detector":
            _run_detector(case, case_out)
        elif args.stage == "map":
            _build_map(case_out)
        progress[uid] = {
            "position": int(case["video_position"]),
            "stage": args.stage,
            "completed": True,
            "elapsed_sec": time.perf_counter() - started,
        }
        write_json(progress_path, progress)

    summary = {
        "stage": args.stage,
        "selected_video_count": len(cases),
        "seeded_count": sum(
            (OUTPUT_ROOT / "cases" / case["video_uid"] / "shared_hierarchy.json").is_file()
            for case, _ in _cases()
        ),
        "audio_count": sum((path / "audio_asr.json").is_file() for path in (OUTPUT_ROOT / "cases").glob("*")),
        "valid_detector_count": sum(
            (path / "detector_validation.json").is_file()
            and bool(load_json(path / "detector_validation.json").get("valid"))
            for path in (OUTPUT_ROOT / "cases").glob("*")
        ),
        "valid_map_count": sum(
            (path / "r1_map_validation.json").is_file()
            and bool(load_json(path / "r1_map_validation.json").get("valid"))
            for path in (OUTPUT_ROOT / "cases").glob("*")
        ),
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
    }
    write_json(OUTPUT_ROOT / f"{args.stage}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
