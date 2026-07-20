"""Run the isolated local-Qwen Medium semantic abstraction comparison."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.medium_semantic_abstraction.experiment import (  # noqa: E402
    load_jsonl,
    prepare_medium_records,
    sha256_file,
)
from src.experiments.medium_semantic_abstraction.qwen_local import (  # noqa: E402
    LocalQwen2VL,
    create_read_only_model_view,
)
from src.experiments.medium_semantic_abstraction.reporting import render_comparison  # noqa: E402


CONFIG_PATH = ROOT / "config/experiments/medium_semantic_abstraction_v0_1.json"
OUT = ROOT / "outputs/experiments/medium_semantic_abstraction_v0_1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def identity_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def merge_checkpoint(prepared: list[dict[str, Any]], existing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["video_id"], row["medium_id"]): row for row in existing}
    for row in prepared:
        old = by_key.get((row["video_id"], row["medium_id"]))
        if not old:
            continue
        for condition in ("one_frame", "three_frame"):
            if "description" in old.get("conditions", {}).get(condition, {}):
                row["conditions"][condition]["description"] = old["conditions"][condition]["description"]
                row["conditions"][condition]["timing"] = old["conditions"][condition]["timing"]
                row["conditions"][condition]["checkpoint_reused"] = True
    return prepared


def packages() -> dict[str, str | None]:
    values = {}
    for name in ("torch", "transformers", "Pillow", "accelerate"):
        try:
            values[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            values[name] = None
    return values


def build_metrics(
    medium: list[dict[str, Any]], coarse: list[dict[str, Any]], model_meta: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in medium:
        by_video[row["video_id"]].append(row)
    coarse_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in coarse:
        coarse_by_video[row["video_id"]].append(row)
    per_video = []
    for video_id in sorted(by_video):
        entry: dict[str, Any] = {
            "video_id": video_id,
            "medium_node_count": len(by_video[video_id]),
            "coarse_node_count": len(coarse_by_video[video_id]),
            "conditions": {},
        }
        for condition in ("one_frame", "three_frame"):
            timings = [row["conditions"][condition]["timing"] for row in by_video[video_id]]
            entry["conditions"][condition] = {
                "image_count": sum(item["image_count"] for item in timings),
                "image_decode_sec": sum(item["image_decode_sec"] for item in timings),
                "image_preprocess_sec": sum(item["image_preprocess_sec"] for item in timings),
                "vlm_inference_sec": sum(item["model_inference_sec"] for item in timings),
                "medium_semantic_indexing_sec": sum(item["total_semantic_generation_sec"] for item in timings),
                "coarse_summarization_sec": sum(
                    row["conditions"][condition]["timing"]["total_semantic_generation_sec"]
                    for row in coarse_by_video[video_id]
                ),
            }
        per_video.append(entry)
    conditions = {}
    for condition in ("one_frame", "three_frame"):
        medium_timings = [row["conditions"][condition]["timing"] for row in medium]
        coarse_timings = [row["conditions"][condition]["timing"] for row in coarse]
        conditions[condition] = {
            "image_count": sum(item["image_count"] for item in medium_timings),
            "image_decode_sec_total": sum(item["image_decode_sec"] for item in medium_timings),
            "image_preprocess_sec_total": sum(item["image_preprocess_sec"] for item in medium_timings),
            "vlm_inference_sec_total": sum(item["model_inference_sec"] for item in medium_timings),
            "medium_semantic_indexing_sec_total": sum(item["total_semantic_generation_sec"] for item in medium_timings),
            "mean_medium_total_sec": statistics.fmean(item["total_semantic_generation_sec"] for item in medium_timings),
            "mean_video_medium_indexing_sec": statistics.fmean(
                row["conditions"][condition]["medium_semantic_indexing_sec"] for row in per_video
            ),
            "coarse_summarization_sec_total": sum(item["total_semantic_generation_sec"] for item in coarse_timings),
            "total_semantic_indexing_and_coarse_sec": (
                sum(item["total_semantic_generation_sec"] for item in medium_timings)
                + sum(item["total_semantic_generation_sec"] for item in coarse_timings)
            ),
        }
    aggregate = {
        "video_count": len(per_video),
        "medium_node_count": len(medium),
        "coarse_node_count": len(coarse),
        "conditions": conditions,
        "three_vs_one_image_ratio": conditions["three_frame"]["image_count"] / conditions["one_frame"]["image_count"],
        "three_vs_one_medium_inference_cost_ratio": conditions["three_frame"]["vlm_inference_sec_total"] / max(conditions["one_frame"]["vlm_inference_sec_total"], 1e-12),
        "human_semantic_quality_review_required": True,
    }
    runtime = {
        "model_load_sec": model_meta["model_load_sec"],
        "model_load_count": model_meta["model_load_count"],
        "model_instance_id": model_meta["model_instance_id"],
        "processor_instance_id": model_meta["processor_instance_id"],
        "model_vram_allocated_bytes": model_meta["model_vram_allocated_bytes"],
        "model_load_peak_vram_bytes": model_meta["model_load_peak_vram_bytes"],
        "local_inference_call_count": model_meta["local_inference_call_count"],
        "conditions": conditions,
        "per_video": per_video,
        "external_api_calls": 0,
        "model_loading_excluded_from_per_video_cost": True,
    }
    return per_video, aggregate, runtime


def main() -> int:
    run_started = time.perf_counter()
    config = load_json(CONFIG_PATH)
    OUT.mkdir(parents=True, exist_ok=True)
    hierarchy_path = ROOT / config["source_hierarchy"]
    manifest_path = ROOT / config["source_manifest"]
    source_config_path = ROOT / config["source_hierarchy_config"]
    hierarchies = load_jsonl(hierarchy_path)
    source_manifest = load_json(manifest_path)
    expected_ids = [str(row["video_id"]) for row in source_manifest["videos"]]
    if [str(row["video_id"]) for row in hierarchies] != expected_ids:
        raise RuntimeError("Frozen 10-video hierarchy identity mismatch")

    model_snapshot = Path(config["local_model"]["snapshot_path"])
    model_view, model_files = create_read_only_model_view(model_snapshot, OUT / "runtime_model_view")
    fingerprint_payload = {
        "config_sha256": sha256_file(CONFIG_PATH),
        "hierarchy_sha256": sha256_file(hierarchy_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_hierarchy_config_sha256": sha256_file(source_config_path),
        "model_files": [{key: row[key] for key in ("filename", "bytes", "content_identity")} for row in model_files],
    }
    fingerprint = identity_hash(fingerprint_payload)
    previous_manifest_path = OUT / "run_manifest.json"
    existing_medium: list[dict[str, Any]] = []
    existing_coarse: list[dict[str, Any]] = []
    if previous_manifest_path.is_file():
        previous = load_json(previous_manifest_path)
        if previous.get("compatibility_fingerprint") != fingerprint:
            raise RuntimeError("Existing semantic-abstraction checkpoint fingerprint is incompatible")
        if (OUT / "per_medium_results.json").is_file():
            existing_medium = load_json(OUT / "per_medium_results.json")
        if (OUT / "per_coarse_results.json").is_file():
            existing_coarse = load_json(OUT / "per_coarse_results.json")
    run_manifest = {
        "schema_version": "medium-semantic-abstraction-run-v1",
        "experiment_id": config["experiment_id"],
        "video_ids": expected_ids,
        "video_count": len(expected_ids),
        "compatibility_fingerprint": fingerprint,
        "fingerprint_components": fingerprint_payload,
        "fine_nodes_captioned": False,
        "coarse_reads_images": False,
        "question_conditioning": False,
        "gold_or_options_accessed": False,
        "external_api_calls": 0,
        "canonical_pipeline_modified": False,
    }
    write_json(previous_manifest_path, run_manifest)

    prepared_medium, coarse_templates = prepare_medium_records(
        hierarchies=hierarchies, root=ROOT, output_dir=OUT,
        one_fractions=[float(value) for value in config["frame_selection"]["one_frame_fractions"]],
        three_fractions=[float(value) for value in config["frame_selection"]["three_frame_fractions"]],
    )
    medium_results = merge_checkpoint(prepared_medium, existing_medium)

    print(
        json.dumps(
            {
                "provider": "local_only",
                "model": config["local_model"]["model"],
                "model_path": model_snapshot.as_posix(),
                "planned_medium_calls": 2 * len(medium_results),
                "planned_coarse_text_calls": 2 * len(coarse_templates),
                "external_api_calls": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    engine = LocalQwen2VL(model_view=model_view, seed=int(config["seed"]))
    model_meta = {
        "provider": "local_only",
        "model": config["local_model"]["model"],
        "snapshot_path": model_snapshot.as_posix(),
        "device": str(engine.device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "physical_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else 0,
        "model_load_sec": engine.model_load_sec,
        "model_load_count": engine.load_count,
        "model_instance_id": engine.model_instance_id,
        "processor_instance_id": engine.processor_instance_id,
        "model_vram_allocated_bytes": engine.model_vram_allocated_bytes,
        "model_load_peak_vram_bytes": engine.model_load_peak_vram_bytes,
        "resolved_model_files": model_files,
    }

    first = medium_results[0]
    smoke_reused = "description" in first["conditions"]["one_frame"]
    if not smoke_reused:
        image_paths = [Path(item["image_path"]) for item in first["conditions"]["one_frame"]["selected_images"]]
        description, timing = engine.describe_images(
            image_paths=image_paths, prompt=config["medium_prompt"],
            max_new_tokens=int(config["local_model"]["max_new_tokens_medium"]),
        )
        first["conditions"]["one_frame"].update({"description": description, "timing": timing, "checkpoint_reused": False})
    smoke = {
        "passed": bool(first["conditions"]["one_frame"].get("description")),
        "reused_existing_checkpoint": smoke_reused,
        "video_id": first["video_id"],
        "medium_id": first["medium_id"],
        "image_path": first["conditions"]["one_frame"]["selected_images"][0]["image_path"],
        "output": first["conditions"]["one_frame"]["description"],
        "device": model_meta["device"],
        "gpu": model_meta["gpu"],
        "physical_vram_bytes": model_meta["physical_vram_bytes"],
        "model_load_sec": model_meta["model_load_sec"],
    }
    write_json(OUT / "smoke_test.json", smoke)
    if not smoke["passed"]:
        raise RuntimeError("Local Qwen one-image smoke test did not return text")
    print(json.dumps({"smoke_test": "PASS", **smoke}, indent=2), flush=True)

    total_medium = len(medium_results)
    for index, row in enumerate(medium_results, start=1):
        for condition in ("one_frame", "three_frame"):
            data = row["conditions"][condition]
            if "description" in data:
                continue
            description, timing = engine.describe_images(
                image_paths=[Path(item["image_path"]) for item in data["selected_images"]],
                prompt=config["medium_prompt"],
                max_new_tokens=int(config["local_model"]["max_new_tokens_medium"]),
            )
            data.update({"description": description, "timing": timing, "checkpoint_reused": False})
        write_json(OUT / "per_medium_results.json", medium_results)
        print(f"Medium {index}/{total_medium}: {row['video_id']}/{row['medium_id']}", flush=True)

    existing_coarse_by_key = {(row["video_id"], row["coarse_id"]): row for row in existing_coarse}
    medium_by_key = {(row["video_id"], row["medium_id"]): row for row in medium_results}
    coarse_results = []
    for index, template in enumerate(coarse_templates, start=1):
        row = dict(template)
        old = existing_coarse_by_key.get((row["video_id"], row["coarse_id"]), {})
        row["conditions"] = {}
        for condition in ("one_frame", "three_frame"):
            ordered = [medium_by_key[(row["video_id"], mid)]["conditions"][condition]["description"] for mid in row["medium_ids"]]
            if "summary" in old.get("conditions", {}).get(condition, {}):
                saved = old["conditions"][condition]
                row["conditions"][condition] = {**saved, "ordered_medium_descriptions": ordered, "checkpoint_reused": True}
            else:
                summary, timing = engine.summarize_text(
                    ordered_descriptions=ordered, prompt=config["coarse_prompt"],
                    max_new_tokens=int(config["local_model"]["max_new_tokens_coarse"]),
                )
                row["conditions"][condition] = {
                    "ordered_medium_descriptions": ordered,
                    "summary": summary,
                    "timing": timing,
                    "checkpoint_reused": False,
                }
        coarse_results.append(row)
        write_json(OUT / "per_coarse_results.json", coarse_results)
        print(f"Coarse {index}/{len(coarse_templates)}: {row['video_id']}/{row['coarse_id']}", flush=True)

    model_meta["local_inference_call_count"] = engine.inference_count
    per_video, aggregate, runtime = build_metrics(medium_results, coarse_results, model_meta)
    runtime["experiment_wall_sec"] = time.perf_counter() - run_started
    runtime["checkpoint_reused_medium_conditions"] = sum(
        bool(row["conditions"][condition].get("checkpoint_reused"))
        for row in medium_results for condition in ("one_frame", "three_frame")
    )
    runtime["checkpoint_reused_coarse_conditions"] = sum(
        bool(row["conditions"][condition].get("checkpoint_reused"))
        for row in coarse_results for condition in ("one_frame", "three_frame")
    )
    write_json(OUT / "per_video_metrics.json", per_video)
    write_json(OUT / "aggregate_metrics.json", aggregate)
    write_json(OUT / "runtime_metrics.json", runtime)
    frozen_config = {
        **config,
        "config_sha256": sha256_file(CONFIG_PATH),
        "source_hierarchy_sha256": sha256_file(hierarchy_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "model": model_meta,
        "prompt_sha256": {
            "medium": hashlib.sha256(config["medium_prompt"].encode()).hexdigest(),
            "coarse": hashlib.sha256(config["coarse_prompt"].encode()).hexdigest(),
        },
        "packages": packages(),
        "git_branch": git_value("branch", "--show-current"),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_status": git_value("status", "--short"),
    }
    write_json(OUT / "frozen_config.json", frozen_config)
    html_validation = render_comparison(
        output_path=OUT / "comparison.html", medium_results=medium_results,
        coarse_results=coarse_results, aggregate=aggregate, model=model_meta,
    )
    write_json(OUT / "html_validation.json", html_validation)
    (OUT / "README.md").write_text(
        "# Medium semantic abstraction v0.1\n\n"
        "Compares deterministic midpoint (1 frame) with 25/50/75% (3 frames) using one persistent, "
        "strictly local Qwen2-VL-2B-Instruct instance across 82 frozen Safe Merge Medium nodes. "
        "Fine nodes are not captioned; Coarse summaries use ordered Medium text only. No questions, "
        "gold, options, external APIs, or canonical pipeline code are used. Semantic quality requires human review.\n",
        encoding="utf-8",
    )
    print(json.dumps({"smoke": "PASS", "medium_nodes": len(medium_results), "coarse_nodes": len(coarse_results), "aggregate": aggregate, "html": str(OUT / 'comparison.html'), "external_api_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
