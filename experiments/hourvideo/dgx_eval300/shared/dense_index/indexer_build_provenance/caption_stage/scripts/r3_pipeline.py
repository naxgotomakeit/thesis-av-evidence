from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE.parents[2]
sys.path.insert(0, str(RUNTIME_ROOT))
sys.path.insert(0, str(RUNTIME_ROOT / "src"))

from experiments.hourvideo_dev50_haiku45_organizer_cost_v1.core import (  # noqa: E402
    OrganizerCallError,
    _call_once,
    _cost_record,
    _load_dedicated_key,
    _map_from_output,
)
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (  # noqa: E402
    load_json,
    sha256_file,
    write_json,
)
from experiments.medium_semantic_abstraction.qwen_local import (  # noqa: E402
    LocalQwen2VL,
    create_read_only_model_view,
)


CONFIG_PATH = HERE / "config.json"
INDEX_ROOT = HERE / "index"
CASES_ROOT = INDEX_ROOT / "cases"
CASE_CONFIGS = INDEX_ROOT / "case_configs"
LOG_ROOT = HERE / "logs"

C_PROMPT = (
    "The images are chronological samples from one short first-person video interval. "
    "Return a compact chronological list of all distinct visible events or hand-object state changes supported by the images. "
    "Use one item per event in the form '1. ...; 2. ...'. Include important objects and scene context when directly visible. "
    "Do not select only a single principal event. Always refer to the camera wearer as 'the camera wearer'. "
    "If the exact object or action is uncertain, use a generic factual description instead of guessing. "
    "Do not infer identity, gender, intent, ownership, causality, unseen actions, motion direction, future actions, or outcomes."
)
C_PROMPT_SHA256 = "dfe98ac6165881faac4d28a014cd46929e6fa87af7a5ac9c4437096dc9f7fa04"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config() -> dict[str, Any]:
    return load_json(CONFIG_PATH)


def _video_counts(cfg: dict[str, Any]) -> tuple[list[str], Counter[str]]:
    lines = [line.strip() for line in (HERE / cfg["uids_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != cfg["expected_question_uids"] or len(set(lines)) != len(lines):
        raise RuntimeError("frozen UID list count/uniqueness mismatch")
    videos = [line.split("_", 1)[0] for line in lines]
    counts = Counter(videos)
    if len(counts) != cfg["expected_video_uids"]:
        raise RuntimeError(f"expected {cfg['expected_video_uids']} videos, found {len(counts)}")
    if set(counts.values()) != {cfg["expected_questions_per_video"]}:
        raise RuntimeError(f"unexpected questions-per-video distribution: {dict(counts)}")
    return sorted(counts), counts


def _source_paths(cfg: dict[str, Any], uid: str) -> dict[str, Path]:
    case = Path(cfg["source_cases_root"]) / uid
    fine = Path(cfg["fine_embedding_root"])
    return {
        "shared_hierarchy": case / "shared_hierarchy.json",
        "medium_embedding": case / "medium_siglip.float32.npy",
        "fine_embedding": fine / f"{uid}.npz",
        "fine_embedding_metadata": fine / f"{uid}.json",
    }


def preflight() -> dict[str, Any]:
    cfg = _config()
    if hashlib.sha256(C_PROMPT.encode("utf-8")).hexdigest() != C_PROMPT_SHA256:
        raise RuntimeError("C prompt hash mismatch")
    videos, counts = _video_counts(cfg)
    rows = []
    failures = []
    for uid in videos:
        sources = _source_paths(cfg, uid)
        missing = [name for name, path in sources.items() if not path.is_file()]
        hierarchy = load_json(sources["shared_hierarchy"]) if not missing else {}
        medium_count = len(hierarchy.get("medium_nodes") or [])
        fine_count = len(hierarchy.get("fine_nodes") or [])
        frame_failures = []
        for medium in hierarchy.get("medium_nodes") or []:
            images = [Path(path) for path in medium.get("representative_frame_paths") or []]
            if not images or any(not image.is_file() for image in images):
                frame_failures.append(str(medium.get("medium_id")))
        if missing or frame_failures:
            failures.append({"video_uid": uid, "missing": missing, "invalid_frame_media": frame_failures})
        rows.append(
            {
                "video_uid": uid,
                "question_uid_count": counts[uid],
                "medium_count": medium_count,
                "fine_count": fine_count,
                "source_artifacts": {
                    name: {
                        "path": str(path.resolve()),
                        "size_bytes": path.stat().st_size if path.is_file() else None,
                        "sha256": sha256_file(path) if path.is_file() else None,
                    }
                    for name, path in sources.items()
                },
                "old_medium_caption_ignored": True,
                "old_coarse_map_ignored": True,
            }
        )
    report = {
        "stage": "preflight",
        "created_at": _utc(),
        "hostname": socket.gethostname(),
        "uids_file": str((HERE / cfg["uids_file"]).resolve()),
        "uids_sha256": sha256_file(HERE / cfg["uids_file"]),
        "question_uid_count": sum(counts.values()),
        "video_uid_count": len(videos),
        "video_uids": videos,
        "prompt_sha256": C_PROMPT_SHA256,
        "question_content_read": False,
        "fine_caption_enabled": False,
        "failures": failures,
        "cases": rows,
    }
    write_json(HERE / "preflight.json", report)
    if failures:
        raise RuntimeError(f"preflight failed for {len(failures)} videos")
    return report


def prepare() -> dict[str, Any]:
    cfg = _config()
    pre = preflight()
    rows = []
    for uid in pre["video_uids"]:
        source = _source_paths(cfg, uid)
        case = CASES_ROOT / uid
        case.mkdir(parents=True, exist_ok=True)
        targets = {
            "shared_hierarchy": case / "shared_hierarchy.json",
            "medium_embedding": case / "medium_siglip.float32.npy",
            "fine_embedding": case / "fine_siglip.npz",
            "fine_embedding_metadata": case / "fine_siglip.json",
        }
        for name, target in targets.items():
            if target.exists():
                if sha256_file(target) != sha256_file(source[name]):
                    raise RuntimeError(f"refusing mismatched existing asset: {target}")
            else:
                shutil.copy2(source[name], target)
            if sha256_file(target) != sha256_file(source[name]):
                raise RuntimeError(f"copied asset hash mismatch: {target}")
        hierarchy = load_json(targets["shared_hierarchy"])
        case_cfg = {
            "experiment": cfg["experiment"],
            "video_uid": uid,
            "siglip_npz": str(targets["fine_embedding"].resolve()),
            "siglip_metadata": str(targets["fine_embedding_metadata"].resolve()),
            "fine_registry": str(targets["shared_hierarchy"].resolve()),
            "medium_embedding": str(targets["medium_embedding"].resolve()),
            "question_content_used": False,
            "fine_caption_enabled": False,
        }
        write_json(CASE_CONFIGS / f"{uid}.json", case_cfg)
        rows.append(
            {
                "video_uid": uid,
                "fine_count": len(hierarchy["fine_nodes"]),
                "medium_count": len(hierarchy["medium_nodes"]),
                "artifacts": {
                    name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                    for name, path in targets.items()
                },
            }
        )
    report = {
        "stage": "prepare",
        "created_at": _utc(),
        "hostname": socket.gethostname(),
        "case_count": len(rows),
        "copied_not_recomputed": ["fine_embedding", "medium_embedding"],
        "question_content_read": False,
        "cases": rows,
    }
    write_json(HERE / "prepare_report.json", report)
    return report


def _gpu_record() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=20,
        ).strip()
    except Exception as exc:
        output = f"unavailable: {type(exc).__name__}: {exc}"
    return {"cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"), "nvidia_smi": output}


def caption() -> dict[str, Any]:
    cfg = _config()
    prepared = prepare()
    model_view, model_files = create_read_only_model_view(Path(cfg["qwen"]["snapshot"]), HERE / "runtime/qwen_model_view")
    overall_started = time.monotonic()
    model = LocalQwen2VL(model_view=model_view, seed=int(cfg["qwen"]["seed"]))
    cases_summary = []
    for case_row in prepared["cases"]:
        uid = case_row["video_uid"]
        case = CASES_ROOT / uid
        hierarchy = load_json(case / "shared_hierarchy.json")
        checkpoint_path = case / "caption_checkpoint.json"
        checkpoint = load_json(checkpoint_path) if checkpoint_path.is_file() else {
            "prompt_sha256": C_PROMPT_SHA256,
            "model": cfg["qwen"]["checkpoint_id"],
            "rows": [],
            "timings": [],
        }
        if checkpoint["prompt_sha256"] != C_PROMPT_SHA256 or checkpoint["model"] != cfg["qwen"]["checkpoint_id"]:
            raise RuntimeError(f"caption checkpoint contract mismatch: {uid}")
        rows = list(checkpoint["rows"])
        timings = list(checkpoint["timings"])
        if len(rows) != len(timings):
            raise RuntimeError(f"caption checkpoint length mismatch: {uid}")
        case_started = time.monotonic()
        for position, medium in enumerate(hierarchy["medium_nodes"], start=1):
            if position <= len(rows):
                if rows[position - 1]["medium_id"] != medium["medium_id"]:
                    raise RuntimeError(f"caption checkpoint Medium mismatch: {uid}")
                continue
            images = [Path(path) for path in medium["representative_frame_paths"]]
            if len(images) > cfg["qwen"]["images_per_medium"]:
                raise RuntimeError(f"too many representative frames: {uid}/{medium['medium_id']}")
            raw, timing = model.describe_images(
                image_paths=images,
                prompt=C_PROMPT,
                max_new_tokens=int(cfg["qwen"]["max_new_tokens"]),
            )
            if not raw:
                raise RuntimeError(f"empty caption: {uid}/{medium['medium_id']}")
            rows.append(
                {
                    "medium_id": medium["medium_id"],
                    "start_sec": medium["start_sec"],
                    "end_sec": medium["end_sec"],
                    "qwen_caption": raw,
                    "caption_source": "qwen2_5_vl_7b_three_ordered_frames_c_minimal_v1",
                    "source_frame_paths": [str(path.resolve()) for path in images],
                    "prompt_sha256": C_PROMPT_SHA256,
                }
            )
            timings.append({"medium_id": medium["medium_id"], **timing})
            write_json(
                checkpoint_path,
                {
                    "prompt_sha256": C_PROMPT_SHA256,
                    "model": cfg["qwen"]["checkpoint_id"],
                    "completed": len(rows),
                    "total": len(hierarchy["medium_nodes"]),
                    "rows": rows,
                    "timings": timings,
                },
            )
        write_json(case / "r3_medium_captions.json", rows)
        write_json(case / "r3_caption_timing.json", timings)
        validation = {
            "valid": len(rows) == len(hierarchy["medium_nodes"]),
            "medium_count": len(rows),
            "prompt_sha256": C_PROMPT_SHA256,
            "fine_caption_generated": False,
        }
        write_json(case / "r3_caption_validation.json", validation)
        cost = {
            "hostname": socket.gethostname(),
            "model": cfg["qwen"]["checkpoint_id"],
            "model_load_sec_shared": model.model_load_sec,
            "case_wall_sec_this_invocation": time.monotonic() - case_started,
            "caption_calls_total_for_case": len(rows),
            "input_tokens": sum(item["input_tokens"] for item in timings),
            "output_tokens": sum(item["output_tokens"] for item in timings),
            "peak_gpu_memory_bytes": max((item["peak_vram_bytes"] for item in timings), default=0),
            "external_api_calls": 0,
            "external_api_cost_usd": 0.0,
            "compute_dollar_cost": None,
            "compute_cost_note": "No machine-hour price was supplied; wall time and GPU telemetry are recorded instead.",
        }
        write_json(case / "r3_caption_cost.json", cost)
        cases_summary.append({"video_uid": uid, **cost, **validation})
    report = {
        "stage": "caption",
        "created_at": _utc(),
        "hostname": socket.gethostname(),
        "prompt": C_PROMPT,
        "prompt_sha256": C_PROMPT_SHA256,
        "model": cfg["qwen"]["checkpoint_id"],
        "model_load_sec": model.model_load_sec,
        "wall_sec": time.monotonic() - overall_started,
        "gpu": _gpu_record(),
        "model_files": model_files,
        "question_content_read": False,
        "cases": cases_summary,
    }
    write_json(HERE / "caption_summary.json", report)
    return report


def _organizer_payload(hierarchy: dict[str, Any], captions: list[dict[str, Any]]) -> dict[str, Any]:
    timeline = []
    for index, (medium, caption_row) in enumerate(zip(hierarchy["medium_nodes"], captions)):
        if medium["medium_id"] != caption_row["medium_id"]:
            raise RuntimeError("Caption/Medium order mismatch")
        timeline.append(
            {
                "medium_index": index,
                "interval": [medium["start_sec"], medium["end_sec"]],
                "caption": caption_row["qwen_caption"],
                "overlapping_asr": [],
            }
        )
    return {"timeline": timeline, "contract": {"global_view": True, "storyline": False, "hard_filtering": False}}


def organize() -> dict[str, Any]:
    cfg = _config()
    videos, _ = _video_counts(cfg)
    _load_dedicated_key(Path(cfg["organizer"]["dedicated_env_file"]))
    provider_cfg = {"anthropic": {
        "model": cfg["organizer"]["model"],
        "timeout_sec": cfg["organizer"]["timeout_sec"],
        "organizer_max_tokens": cfg["organizer"]["max_tokens"],
        "temperature": cfg["organizer"]["temperature"],
        "pricing_usd_per_million": cfg["organizer"]["pricing_usd_per_million"],
    }}
    overall_started = time.monotonic()
    cases_summary = []
    for uid in videos:
        case = CASES_ROOT / uid
        map_path = case / "r3_2_navigation_map.json"
        usage_path = case / "organizer_usage.json"
        if map_path.is_file() and usage_path.is_file():
            cases_summary.append(load_json(usage_path))
            continue
        hierarchy = load_json(case / "shared_hierarchy.json")
        captions = load_json(case / "r3_medium_captions.json")
        if len(captions) != len(hierarchy["medium_nodes"]):
            raise RuntimeError(f"caption coverage incomplete: {uid}")
        payload = _organizer_payload(hierarchy, captions)
        if any("question" in key.lower() or "option" in key.lower() for key in payload):
            raise RuntimeError("question-shaped key in Organizer payload")
        case_started = time.monotonic()
        try:
            raw, usage = _call_once(provider_cfg, payload)
            map_doc = _map_from_output(hierarchy, captions, [], raw)
        except Exception as exc:
            failed_usage = dict(getattr(exc, "usage", {}) or {})
            if failed_usage:
                failed_usage["estimated_cost"] = _cost_record(failed_usage, cfg["organizer"]["pricing_usd_per_million"])
            write_json(
                case / "organizer_error.json",
                {
                    "created_at": _utc(),
                    "hostname": socket.gethostname(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "usage": failed_usage,
                    "raw_text": getattr(exc, "raw_text", ""),
                },
            )
            raise
        usage.update(
            {
                "stage": "visual_only_organizer",
                "video_uid": uid,
                "hostname": socket.gethostname(),
                "case_wall_sec": time.monotonic() - case_started,
                "medium_count": len(captions),
                "coarse_count": len(map_doc["coarse_regions"]),
                "pricing_usd_per_million": cfg["organizer"]["pricing_usd_per_million"],
                "pricing_source": cfg["organizer"]["pricing_source"],
                "pricing_verified_date": cfg["organizer"]["pricing_verified_date"],
                "estimated_cost": _cost_record(usage, cfg["organizer"]["pricing_usd_per_million"]),
                "audio_used": False,
                "question_content_read": False,
            }
        )
        write_json(case / "organizer_input.json", payload)
        write_json(case / "organizer_raw_output.json", raw)
        write_json(map_path, map_doc)
        write_json(usage_path, usage)
        cases_summary.append(usage)
    report = {
        "stage": "organize",
        "created_at": _utc(),
        "hostname": socket.gethostname(),
        "model": cfg["organizer"]["model"],
        "wall_sec": time.monotonic() - overall_started,
        "case_count": len(cases_summary),
        "input_tokens": sum(item["input_tokens"] for item in cases_summary),
        "output_tokens": sum(item["output_tokens"] for item in cases_summary),
        "provider_latency_sec": sum(item["latency_sec"] for item in cases_summary),
        "estimated_cost_usd": sum(item["estimated_cost"]["total_usd"] for item in cases_summary),
        "question_content_read": False,
        "audio_used": False,
        "cases": cases_summary,
    }
    write_json(HERE / "organizer_summary.json", report)
    return report


def validate() -> dict[str, Any]:
    cfg = _config()
    videos, _ = _video_counts(cfg)
    rows = []
    failures = []
    for uid in videos:
        case = CASES_ROOT / uid
        expected = [
            "shared_hierarchy.json",
            "fine_siglip.npz",
            "fine_siglip.json",
            "medium_siglip.float32.npy",
            "r3_medium_captions.json",
            "r3_caption_validation.json",
            "r3_2_navigation_map.json",
            "organizer_usage.json",
        ]
        missing = [name for name in expected if not (case / name).is_file()]
        if missing:
            failures.append({"video_uid": uid, "missing": missing})
        rows.append(
            {
                "video_uid": uid,
                "missing": missing,
                "artifacts": {
                    name: {"size_bytes": (case / name).stat().st_size, "sha256": sha256_file(case / name)}
                    for name in expected if (case / name).is_file()
                },
            }
        )
    report = {
        "stage": "validate",
        "created_at": _utc(),
        "hostname": socket.gethostname(),
        "valid": not failures,
        "case_count": len(rows),
        "failures": failures,
        "question_content_read": False,
        "planner_started": False,
        "inspector_started": False,
        "eval300_started": False,
        "cases": rows,
    }
    write_json(HERE / "validation_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preflight", "prepare", "caption", "organize", "validate"])
    args = parser.parse_args()
    result = {
        "preflight": preflight,
        "prepare": prepare,
        "caption": caption,
        "organize": organize,
        "validate": validate,
    }[args.stage]()
    print(json.dumps({"stage": args.stage, "result": result}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
