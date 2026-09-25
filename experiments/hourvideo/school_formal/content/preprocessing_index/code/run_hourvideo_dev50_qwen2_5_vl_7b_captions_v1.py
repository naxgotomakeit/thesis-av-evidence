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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (  # noqa: E402
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.offline import (  # noqa: E402
    prepare_case,
)


EXPERIMENT = "hourvideo_dev50_qwen2_5_vl_7b_captions_v1"
CAPTION_ARTIFACTS = (
    "r3_caption_progress.json",
    "r3_medium_captions.json",
    "r3_caption_timing.json",
    "r3_caption_cost.json",
    "r3_caption_validation.json",
)


def _valid_caption(case_out: Path) -> bool:
    validation = case_out / "r3_caption_validation.json"
    return validation.is_file() and bool(load_json(validation).get("valid"))


def _seed_valid_pilot_captions(cases: list[tuple[dict[str, Any], Path]]) -> list[dict[str, Any]]:
    pilot_root = ROOT / "outputs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1/cases"
    records: list[dict[str, Any]] = []
    for case, _ in cases:
        uid = case["video_uid"]
        source = pilot_root / uid
        target = ROOT / case["output_root"]
        if _valid_caption(target) or not _valid_caption(source):
            continue
        source_hierarchy = source / "shared_hierarchy.json"
        target_hierarchy = target / "shared_hierarchy.json"
        if source_hierarchy.read_bytes() != target_hierarchy.read_bytes():
            raise RuntimeError(f"pilot/dev50 hierarchy mismatch prevents caption reuse: {uid}")
        copied = []
        for name in CAPTION_ARTIFACTS:
            source_path = source / name
            if not source_path.is_file():
                raise FileNotFoundError(f"valid pilot caption is missing {name}: {uid}")
            target_path = target / name
            shutil.copy2(source_path, target_path)
            copied.append({"name": name, "sha256": sha256_file(target_path)})
        if not _valid_caption(target):
            raise RuntimeError(f"copied pilot caption did not validate: {uid}")
        records.append({
            "video_uid": uid,
            "source": str(source),
            "target": str(target),
            "hierarchy_sha256": sha256_file(target_hierarchy),
            "copied_artifacts": copied,
        })
    return records


def _build_cases(base_cfg: dict[str, Any], output_root: str) -> list[tuple[dict[str, Any], Path]]:
    hourvideo_root = Path(base_cfg["embedding_root"]).parent
    frame_audit_path = hourvideo_root / "manifests/dev50_frames_1fps_audit.json"
    video_audit_path = hourvideo_root / "manifests/dev50_download_audit.json"
    safe_path = ROOT / base_cfg["safe_question_manifest"]
    frame_rows = load_json(frame_audit_path)["videos"]
    video_rows = {row["video_uid"]: row for row in load_json(video_audit_path)["videos"]}
    safe_rows: dict[str, list[dict[str, Any]]] = {}
    for question in load_json(safe_path)["questions"]:
        safe_rows.setdefault(question["video_uid"], []).append(question)

    if len(frame_rows) != 50 or len(video_rows) != 50:
        raise RuntimeError("dev50 source audits must contain exactly 50 videos")

    cases: list[tuple[dict[str, Any], Path]] = []
    seen: set[str] = set()
    for position, frame_row in enumerate(frame_rows, 1):
        uid = str(frame_row["video_uid"])
        if uid in seen or uid not in video_rows or uid not in safe_rows:
            raise RuntimeError(f"invalid dev50 identity coverage: {uid}")
        seen.add(uid)
        questions = sorted(safe_rows[uid], key=lambda row: row["question_id"])
        question = next(
            (row for row in questions if all(option["content_type"] == "text" for option in row["answer_options"])),
            questions[0],
        )
        case = copy.deepcopy(base_cfg)
        case.update({
            "experiment": EXPERIMENT,
            "video_position": position,
            "video_uid": uid,
            "question_id": question["question_id"],
            "video_path": str(hourvideo_root / "videos/v2/video_540ss" / f"{uid}.mp4"),
            "frame_dir": str(hourvideo_root / "runtime" / uid / "frames_1fps"),
            "has_audio": bool(video_rows[uid].get("has_audio")),
            "output_root": f"{output_root}/cases/{uid}",
            "siglip_npz": f"{base_cfg['embedding_root']}/siglip/shards/{uid}.npz",
            "siglip_metadata": f"{base_cfg['embedding_root']}/siglip/shards/{uid}.json",
            "dino_npz": f"{base_cfg['embedding_root']}/dino_v2/shards/{uid}.npz",
            "dino_metadata": f"{base_cfg['embedding_root']}/dino_v2/shards/{uid}.json",
        })
        case_path = ROOT / output_root / "case_configs" / f"{uid}.json"
        write_json(case_path, case)
        cases.append((case, case_path))

    write_json(ROOT / output_root / "input_manifest.json", {
        "experiment": EXPERIMENT,
        "scope": "HourVideo dev50",
        "video_count": len(cases),
        "frame_count": sum(int(row["frame_count"]) for row in frame_rows),
        "medium_duration_sec": int(base_cfg["medium_duration_sec"]),
        "fine_duration_sec": int(base_cfg["fine_duration_sec"]),
        "caption_model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "caption_snapshot": base_cfg["qwen"]["snapshot"],
        "images_per_medium": int(base_cfg["qwen"]["images_per_medium"]),
        "max_new_tokens": int(base_cfg["qwen"]["max_new_tokens"]),
        "source_audits": {
            "frames": {"path": str(frame_audit_path), "sha256": sha256_file(frame_audit_path)},
            "videos": {"path": str(video_audit_path), "sha256": sha256_file(video_audit_path)},
            "safe_questions": {"path": str(safe_path), "sha256": sha256_file(safe_path)},
        },
        "videoseal_outputs_used": False,
        "external_api_calls": 0,
    })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "seed", "captions"], required=True)
    parser.add_argument(
        "--config",
        default="configs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1.json",
    )
    parser.add_argument(
        "--output-root",
        default=f"outputs/experiments/{EXPERIMENT}",
    )
    parser.add_argument("--video-uid")
    args = parser.parse_args()

    base_cfg = load_json(ROOT / args.config)
    cases = _build_cases(base_cfg, args.output_root)
    if args.video_uid:
        cases = [row for row in cases if row[0]["video_uid"] == args.video_uid]
        if not cases:
            raise RuntimeError(f"unknown dev50 video_uid: {args.video_uid}")

    if args.stage in {"seed", "captions"}:
        reused = _seed_valid_pilot_captions(cases)
        reuse_manifest = ROOT / args.output_root / "pilot_caption_reuse_manifest.json"
        previous = load_json(reuse_manifest).get("reused", []) if reuse_manifest.is_file() else []
        by_uid = {row["video_uid"]: row for row in previous + reused}
        write_json(reuse_manifest, {
            "policy": "copy only valid captions when pilot and dev50 shared_hierarchy bytes are identical",
            "reused": [by_uid[uid] for uid in sorted(by_uid)],
        })
        if args.stage == "seed":
            summary = {
                "stage": "seed",
                "newly_reused": len(reused),
                "dev50_valid_caption_count": sum(
                    _valid_caption(ROOT / case["output_root"]) for case, _ in cases
                ),
                "output_root": args.output_root,
            }
            write_json(ROOT / args.output_root / "seed_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False))
            return 0

    progress_path = ROOT / args.output_root / f"{args.stage}_progress.json"
    progress = load_json(progress_path) if progress_path.is_file() else {}
    for case, case_path in cases:
        uid = case["video_uid"]
        case_out = ROOT / case["output_root"]
        started = time.perf_counter()
        if not (case_out / "shared_hierarchy.json").is_file():
            prepare_case(ROOT, case)
        if args.stage == "captions" and not _valid_caption(case_out):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.hourvideo_r1_av_r3_2_single_video_smoke.caption_runner",
                    "--config",
                    str(case_path.relative_to(ROOT)),
                ],
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                check=True,
            )
        progress[uid] = {
            "position": int(case["video_position"]),
            "stage": args.stage,
            "completed": True,
            "caption_valid": _valid_caption(case_out),
            "elapsed_sec": time.perf_counter() - started,
        }
        write_json(progress_path, progress)

    completed = sum(1 for case, _ in _build_cases(base_cfg, args.output_root) if _valid_caption(ROOT / case["output_root"]))
    summary = {
        "stage": args.stage,
        "selected_video_count": len(cases),
        "dev50_valid_caption_count": completed,
        "output_root": args.output_root,
    }
    write_json(ROOT / args.output_root / f"{args.stage}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
