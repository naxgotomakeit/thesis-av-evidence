from __future__ import annotations

import datetime
import ast
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import _build_r1_map
from experiments.hourvideo_v6_6_1_contract_telemetry_v1 import core as v661


ORDERED_UID_SHA256 = "6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1"
R3_SOURCE_TREE_SHA256 = "c0403d8216ba840116f9adbe2501631d63c362595a685cf3e1c3519dba14873e"
R3_WORK_TREE_SHA256 = "ff5d17a92e80a4dddeb5189e1d898647a595f0635c9da446eb61bf32dbbd7d65"
R3_MATERIALIZATION_MANIFEST_SHA256 = "4d376dacdebbd5ed2e0fb9229038d5f25aa33565d1e8db3ae3045da4ccefdb87"
R3_FINAL_ARCHIVE_SHA256 = "fa2c0c5ff82109daaf36249ebc8ec5b685e4ee73320bb910d6d2d923b347ff6b"
SIDES = ("r1_av", "r3_2")
FORBIDDEN_INPUT_KEYS = {
    "correct_answer_label", "gold", "gold_option_id", "answer_label", "correct",
}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _append_fsync(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _questions_without_gold(annotation_path: Path) -> dict[str, dict[str, Any]]:
    annotations = load_json(annotation_path)
    result: dict[str, dict[str, Any]] = {}
    for video_uid, video in annotations.items():
        for row in video["benchmark_dataset"]:
            question_id = str(row["qid"])
            result[question_id] = {
                "question_id": question_id,
                "video_uid": str(row["video_uid"]),
                "question_text": str(row["question"]),
                "answer_options": [
                    {"content_type": "text", "option_id": label, "text": str(row[f"answer_{index}"])}
                    for index, label in enumerate("ABCDE", 1)
                ],
            }
            if result[question_id]["video_uid"] != video_uid:
                raise ValueError(f"annotation video identity mismatch: {question_id}")
    return result


def _find_audio_source(root: Path, video_uid: str) -> Path:
    candidates = [
        root / "outputs/experiments/hourvideo_pilot10_api_maps_v6_3_sparse_official_full_v1/source/cases" / video_uid / "audio_asr.json",
        root / "outputs/archive/2026-08-12/contaminated_or_superseded_pilots/hourvideo_dev20_v6_1_local_av_v1/source/cases" / video_uid / "audio_asr.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no frozen R1 audio source for {video_uid}")


def _validate_hierarchy_and_map(
    hierarchy: dict[str, Any], map_doc: dict[str, Any], video_uid: str,
) -> dict[str, int]:
    if hierarchy["video_uid"] != video_uid:
        raise ValueError(f"hierarchy video mismatch: {video_uid}")
    medium_ids = [row["medium_id"] for row in hierarchy["medium_nodes"]]
    fine_ids = [row["fine_id"] for row in hierarchy["fine_nodes"]]
    if len(medium_ids) != len(set(medium_ids)) or len(fine_ids) != len(set(fine_ids)):
        raise ValueError(f"duplicate hierarchy IDs: {video_uid}")
    medium_set = set(medium_ids)
    for fine in hierarchy["fine_nodes"]:
        if fine["parent_medium_id"] not in medium_set:
            raise ValueError(f"orphan Fine node: {video_uid}/{fine['fine_id']}")
        if not Path(fine["source_frame_path"]).is_file():
            raise FileNotFoundError(f"missing Fine image: {fine['source_frame_path']}")
    covered: list[str] = []
    coarse_ids: list[str] = []
    for coarse in map_doc["coarse_regions"]:
        coarse_ids.append(coarse["coarse_id"])
        for medium_id in coarse["source_medium_ids"]:
            if medium_id not in medium_set:
                raise ValueError(f"orphan Medium map reference: {video_uid}/{medium_id}")
            covered.append(medium_id)
    if len(coarse_ids) != len(set(coarse_ids)):
        raise ValueError(f"duplicate Coarse IDs: {video_uid}")
    if set(covered) != medium_set:
        raise ValueError(f"incomplete Medium coverage: {video_uid}")
    return {"coarse": len(coarse_ids), "medium": len(medium_ids), "fine": len(fine_ids)}


def verify_locked_r3(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["formal_r3_root"])
    materialization = root / "manifests/materialization_manifest.json"
    archive = Path(cfg["formal_r3_archive"])
    archive_manifest = Path(cfg["formal_r3_archive_manifest"])
    checks = {
        "source_tree_sha256": load_json(Path(cfg["formal_r3_source_manifest"]))["tree_sha256"],
        "work_tree_sha256": load_json(materialization)["tree_sha256"],
        "materialization_manifest_sha256": sha256_file(materialization),
        "archive_sha256": sha256_file(archive),
        "archive_manifest_sha256": sha256_file(archive_manifest),
    }
    expected = {
        "source_tree_sha256": R3_SOURCE_TREE_SHA256,
        "work_tree_sha256": R3_WORK_TREE_SHA256,
        "materialization_manifest_sha256": R3_MATERIALIZATION_MANIFEST_SHA256,
        "archive_sha256": R3_FINAL_ARCHIVE_SHA256,
    }
    for key, value in expected.items():
        if checks[key] != value:
            raise ValueError(f"formal R3 lock mismatch for {key}: {checks[key]} != {value}")
    return {"root": str(root), **checks, "locked": True}


def prepare_source(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    source = root / cfg["source_experiment"]
    source.mkdir(parents=True, exist_ok=True)
    uid_path = Path(cfg["eval300_ordered_uid_path"])
    if sha256_file(uid_path) != ORDERED_UID_SHA256:
        raise ValueError("Eval300 ordered UID SHA mismatch")
    ordered_uids = [row.strip() for row in uid_path.read_text(encoding="utf-8").splitlines() if row.strip()]
    if len(ordered_uids) != 300 or len(set(ordered_uids)) != 300:
        raise ValueError("Eval300 must contain 300 unique ordered question IDs")
    questions = _questions_without_gold(Path(cfg["annotation_path"]))
    if any(question_id not in questions for question_id in ordered_uids):
        raise ValueError("Eval300 question missing from annotation identity source")
    r3_lock = verify_locked_r3(cfg)
    r3_cases = Path(cfg["formal_r3_root"]) / "cases"
    r1_cases = Path(cfg["formal_r1_root"]) / "cases"
    video_uids = list(dict.fromkeys(questions[question_id]["video_uid"] for question_id in ordered_uids))
    if len(video_uids) != 12:
        raise ValueError("Eval300 must map to exactly 12 videos")

    video_assets: dict[str, dict[str, Any]] = {}
    totals = {"coarse": 0, "medium": 0, "fine": 0}
    for video_uid in video_uids:
        r1_dir = r1_cases / video_uid
        r3_dir = r3_cases / video_uid
        hierarchy_path = r3_dir / "shared_hierarchy.json"
        medium_path = r3_dir / "medium_siglip.float32.npy"
        projection_path = r1_dir / "r1_medium_projection.json"
        if sha256_file(hierarchy_path) != sha256_file(r1_dir / "shared_hierarchy.json"):
            raise ValueError(f"R1/R3 hierarchy mismatch: {video_uid}")
        if sha256_file(medium_path) != sha256_file(r1_dir / "medium_siglip.float32.npy"):
            raise ValueError(f"R1/R3 Medium embedding mismatch: {video_uid}")
        hierarchy = load_json(hierarchy_path)
        r3_map_path = r3_dir / "r3_2_navigation_map.json"
        counts = _validate_hierarchy_and_map(hierarchy, load_json(r3_map_path), video_uid)
        for key in totals:
            totals[key] += counts[key]

        audio_path = _find_audio_source(root, video_uid)
        r1_map_dir = source / "video_assets" / video_uid
        r1_map_path = r1_map_dir / "r1_av_navigation_map.json"
        embeddings = np.load(medium_path, allow_pickle=False).astype(np.float32)
        embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        audio_doc = load_json(audio_path)
        audio_segments = audio_doc["segments"] if isinstance(audio_doc, dict) else audio_doc
        r1_map = _build_r1_map(
            hierarchy, load_json(projection_path), audio_segments, embeddings,
        )
        write_json(r1_map_path, r1_map)
        _validate_hierarchy_and_map(hierarchy, r1_map, video_uid)

        asset_paths = {
            "shared_hierarchy.json": str(hierarchy_path),
            "medium_siglip.float32.npy": str(medium_path),
            "r1_medium_projection.json": str(projection_path),
            "r3_medium_captions.json": str(r3_dir / "r3_medium_captions.json"),
            "r1_av_navigation_map.json": str(r1_map_path),
            "r3_2_navigation_map.json": str(r3_map_path),
        }
        for path in asset_paths.values():
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        video_assets[video_uid] = {
            "video_uid": video_uid,
            "asset_paths": asset_paths,
            "asset_sha256": {name: sha256_file(Path(path)) for name, path in asset_paths.items()},
            "siglip_npz": str(r3_dir / "fine_siglip.npz"),
            "siglip_npz_sha256": sha256_file(r3_dir / "fine_siglip.npz"),
            "r1_audio_source": str(audio_path),
            "r1_audio_source_sha256": sha256_file(audio_path),
            "counts": counts,
        }

    route_manifest_path = source / "ordered_routes.jsonl"
    if route_manifest_path.exists():
        route_manifest_path.unlink()
    case_rows: list[dict[str, Any]] = []
    forbidden_hits: list[str] = []
    for order, question_id in enumerate(ordered_uids):
        question = questions[question_id]
        forbidden_hits.extend(sorted(set(question) & FORBIDDEN_INPUT_KEYS))
        video_uid = question["video_uid"]
        case_dir = source / "cases" / question_id
        write_json(case_dir / "question_input.json", question)
        case_cfg = {
            "schema_version": "hourvideo_v6_6_2_formal_question_case_v1",
            "case_id": question_id,
            "question_id": question_id,
            "video_uid": video_uid,
            "ordered_question_index": order,
            "asset_paths": video_assets[video_uid]["asset_paths"],
            "asset_sha256": video_assets[video_uid]["asset_sha256"],
            "siglip_npz": video_assets[video_uid]["siglip_npz"],
            "siglip_npz_sha256": video_assets[video_uid]["siglip_npz_sha256"],
            "question_input_sha256": sha256_file(case_dir / "question_input.json"),
            "gold_fields_present": False,
        }
        write_json(source / "case_configs" / f"{question_id}.json", case_cfg)
        for side in SIDES:
            _append_fsync(route_manifest_path, {
                "route_index": order * 2 + SIDES.index(side),
                "question_index": order, "question_id": question_id,
                "video_uid": video_uid, "route": side,
                "route_key": f"{question_id}::{side}",
                "question_input_sha256": case_cfg["question_input_sha256"],
                "index_asset_sha256": (
                    case_cfg["asset_sha256"]["r1_av_navigation_map.json"]
                    if side == "r1_av" else case_cfg["asset_sha256"]["r3_2_navigation_map.json"]
                ),
            })
        case_rows.append({
            "order": order, "question_id": question_id, "video_uid": video_uid,
            "question_input_sha256": case_cfg["question_input_sha256"],
            "case_config_sha256": sha256_file(source / "case_configs" / f"{question_id}.json"),
        })
    if forbidden_hits:
        raise ValueError(f"forbidden answer fields entered source: {sorted(set(forbidden_hits))}")

    pilot_ids = set(cfg["pilot10_question_ids"])
    overlap = [question_id for question_id in ordered_uids if question_id in pilot_ids]
    if len(overlap) != 3:
        raise ValueError(f"expected Pilot/Eval300 overlap=3, got {len(overlap)}")
    manifest = {
        "schema_version": "hourvideo_v6_6_2_formal_eval300_source_v1",
        "created_at_utc": _now(), "gold_fields_present": False,
        "ordered_uid_path": str(uid_path), "ordered_uid_sha256": ORDERED_UID_SHA256,
        "question_count": len(case_rows), "route_count": len(case_rows) * 2,
        "video_count": len(video_uids), "execution_sides": list(SIDES),
        "case_identity": "question_id", "video_assets_shared": True,
        "cases": case_rows, "video_assets": video_assets,
        "hierarchy_totals": totals, "r3_lock": r3_lock,
        "pilot_overlap_question_ids": overlap,
        "held_out_definition": "Eval297 = ordered Eval300 excluding the 3 Pilot10 question IDs",
        "ordered_route_manifest": str(route_manifest_path),
        "ordered_route_manifest_sha256": sha256_file(route_manifest_path),
    }
    write_json(source / "source_manifest.json", manifest)
    return manifest


def score_fixed_population(
    ordered_question_ids: list[str], predictions: list[dict[str, Any]],
    gold_by_question_id: dict[str, str], pilot_question_ids: set[str], side: str,
) -> dict[str, Any]:
    """Post-hoc scorer with fixed denominators; never used by source or live execution."""
    if len(ordered_question_ids) != 300 or len(set(ordered_question_ids)) != 300:
        raise ValueError("full scoring population must be exactly 300 unique question IDs")
    side_predictions = [row for row in predictions if row.get("route") == side]
    by_id = {row["question_id"]: row for row in side_predictions}
    if len(by_id) != len(side_predictions):
        raise ValueError(f"duplicate prediction identity for route {side}")
    rows: list[dict[str, Any]] = []
    for question_id in ordered_question_ids:
        prediction = by_id.get(question_id)
        status = (prediction.get("execution_status") if prediction else None) or "failed"
        selected = prediction.get("selected_option_id") if prediction else None
        failure_kind = prediction.get("failure_kind") if prediction else "missing_prediction"
        correct = bool(selected is not None and selected == gold_by_question_id[question_id])
        rows.append({
            "question_id": question_id, "route": side,
            "selected_option_id": selected, "correct": correct,
            "execution_status": status, "failure_kind": failure_kind,
            "reasoning_termination": prediction.get("reasoning_termination") if prediction else None,
            "is_held_out": question_id not in pilot_question_ids,
        })
    held_out = [row for row in rows if row["is_held_out"]]
    if len(held_out) != 297:
        raise ValueError(f"held-out population must be Eval297, got {len(held_out)}")
    return {
        "route": side,
        "full_eval300": {
            "denominator": 300, "correct": sum(row["correct"] for row in rows),
            "accuracy": sum(row["correct"] for row in rows) / 300.0, "rows": rows,
        },
        "held_out_eval297": {
            "denominator": 297, "correct": sum(row["correct"] for row in held_out),
            "accuracy": sum(row["correct"] for row in held_out) / 297.0, "rows": held_out,
        },
        "execution_status_counts": {
            status: sum(row["execution_status"] == status for row in rows)
            for status in v661.EXECUTION_STATUSES
        },
        "reasoning_termination_counts": {
            status: sum(row["reasoning_termination"] == status for row in rows)
            for status in v661.REASONING_TERMINATIONS
        },
        "missing_prediction_count": sum(row["selected_option_id"] is None for row in rows),
        "timeout_count": sum(row["failure_kind"] == "timeout" for row in rows),
        "failed_counted_in_denominator": True,
    }


def formal_preflight(root: Path, config_path: Path, identity_filter: str | None = None) -> dict[str, Any]:
    cfg = load_json(config_path)
    source = root / cfg["source_experiment"]
    manifest = load_json(source / "source_manifest.json")
    if manifest["question_count"] != 300 or manifest["route_count"] != 600:
        raise ValueError("formal source population is not 300 cases / 600 routes")
    cases = v661._case_paths(cfg, root, identity_filter)
    if not cases:
        raise ValueError("no matching formal question cases")
    audits = []
    for case_cfg, case_dir in cases:
        question = load_json(case_dir / "question_input.json")
        question_id = question["question_id"]
        if question_id != case_cfg["question_id"]:
            raise ValueError(f"question identity mismatch: {question_id}")
        for name, digest in case_cfg["asset_sha256"].items():
            if sha256_file(Path(case_cfg["asset_paths"][name])) != digest:
                raise ValueError(f"asset SHA mismatch: {question_id}/{name}")
        if sha256_file(Path(case_cfg["siglip_npz"])) != case_cfg["siglip_npz_sha256"]:
            raise ValueError(f"Fine embedding SHA mismatch: {question_id}")
        planners = {}
        for side in SIDES:
            path = root / cfg["planner_source_experiment"] / "cases" / question_id / side / "planner.json"
            planners[side] = {"path": str(path), "present": path.is_file()}
        audits.append({"question_id": question_id, "video_uid": case_cfg["video_uid"], "planners": planners})
    result = {
        "status": "ready" if all(p["present"] for row in audits for p in row["planners"].values()) else "source_ready_planner_pending",
        "formal_version": "v6.6.2", "case_identity": "question_id",
        "question_count_checked": len(cases), "route_count_checked": len(cases) * 2,
        "source_manifest_sha256": sha256_file(source / "source_manifest.json"),
        "ordered_route_manifest_sha256": sha256_file(source / "ordered_routes.jsonl"),
        "r3_lock": verify_locked_r3(cfg), "audits": audits,
    }
    write_json(root / cfg["output_root"] / "formal_preflight.json", result)
    return result


def _effective_local_planner_system(base: Any, requirements: list[dict[str, Any]]) -> str:
    """Mirror the frozen provider adapter without changing its prompt or schema."""
    coarse_ids: list[Any] = []
    system = base.COARSE_LOCKED_PLANNER_PROMPT
    system += (
        "\n\nLOCAL STRUCTURED-OUTPUT REQUIREMENT: Return compact JSON with no whitespace "
        "padding. Emit exactly one plan for each required requirement ID. Use one short "
        "search_description, no more than two short query_variants, and one short "
        "modality_strategy per plan. For every plan, emit exactly one coarse_judgment "
        f"for each of these {len(coarse_ids)} Coarse IDs in this exact order: "
        f"{json.dumps(coarse_ids, ensure_ascii=False)}. Never repeat, skip, or add a Coarse "
        "ID. Each reason must be a factual phrase of at most six words; do not restate "
        "the region summary or the option. Immediately after the final judgment of the "
        "final requirement, close requirement_plans, emit coarse_lock_is_hard_scope=true, "
        "and close the top-level object."
    )
    return system


def audit_planner_amendment_offline(root: Path, config_path: Path) -> dict[str, Any]:
    """Perform all pre-model amendment gates, including exact Qwen chat tokenization."""
    from transformers import AutoTokenizer

    cfg = load_json(config_path)
    if int(cfg["anthropic"]["planner_max_tokens"]) != 8192:
        raise ValueError("superseding amendment must uniformly set Planner max tokens to 8192")
    v1_path = root / "configs/experiments/hourvideo_v6_6_2_formal_eval300_v1.json"
    v1 = load_json(v1_path)
    allowed_metadata = {
        "experiment", "source_experiment", "planner_source_experiment", "output_root",
        "supersedes", "superseding_amendment", "planner_amendment_canary_question_id",
        "qwen3_effective_max_model_len",
    }
    semantic_differences = []
    keys = set(v1) | set(cfg)
    for key in sorted(keys - allowed_metadata):
        if key == "anthropic":
            left = dict(v1[key]); right = dict(cfg[key])
            left_tokens = left.pop("planner_max_tokens"); right_tokens = right.pop("planner_max_tokens")
            if left != right or left_tokens != 4096 or right_tokens != 8192:
                semantic_differences.append({"field": "anthropic", "v1": v1[key], "v2": cfg[key]})
        elif v1.get(key) != cfg.get(key):
            semantic_differences.append({"field": key, "v1": v1.get(key), "v2": cfg.get(key)})
    if semantic_differences:
        raise ValueError(f"v2 has non-whitelisted semantic config differences: {semantic_differences}")

    prompt = v661._base.COARSE_LOCKED_PLANNER_PROMPT
    prompt_lower = prompt.lower()
    biased_terms = [term for term in ("r1", "r3", "semantic caption", "caption hierarchy") if term in prompt_lower]
    source_manifest = load_json(root / cfg["source_experiment"] / "source_manifest.json")
    representation_audit = {"r1": [], "r3": []}
    for video_uid, assets in source_manifest["video_assets"].items():
        r1_map = load_json(Path(assets["asset_paths"]["r1_av_navigation_map.json"]))
        r3_map = load_json(Path(assets["asset_paths"]["r3_2_navigation_map.json"]))
        representation_audit["r1"].append({
            "video_uid": video_uid, "map_type": r1_map["map_type"],
            "semantic_fields_available": r1_map["semantic_fields_available"],
            "all_regions_have_structural_fallback": all("visual_structured_fallback" in row for row in r1_map["coarse_regions"]),
        })
        representation_audit["r3"].append({
            "video_uid": video_uid, "map_type": r3_map["map_type"],
            "semantic_fields_available": r3_map["semantic_fields_available"],
            "all_regions_have_semantic_caption_inputs": all("exact_source_captions" in row for row in r3_map["coarse_regions"]),
        })
    planner_prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
    required_planner_prompt_sha = "adf1dc19348e91c26758c17af925c1ae202e979ec5763790a158aba822d6a1bd"
    interface_neutral = (
        planner_prompt_sha == required_planner_prompt_sha
        and not biased_terms
        and all(not row["semantic_fields_available"] and row["all_regions_have_structural_fallback"] for row in representation_audit["r1"])
        and all(row["semantic_fields_available"] and row["all_regions_have_semantic_caption_inputs"] for row in representation_audit["r3"])
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "${SCHOOL_PROJECT_ROOT}/models/Qwen3-8B",
        local_files_only=True,
    )
    max_model_len = int(cfg["qwen3_effective_max_model_len"])
    output_tokens = int(cfg["anthropic"]["planner_max_tokens"])
    token_rows = []
    for case_cfg, case_dir in v661._case_paths(cfg, root, None):
        question = load_json(case_dir / "question_input.json")
        requirements = v661.option_requirements(question)
        system = _effective_local_planner_system(v661._base, requirements)
        for side in SIDES:
            map_name = "r1_av_navigation_map.json" if side == "r1_av" else "r3_2_navigation_map.json"
            map_path = Path(case_cfg["asset_paths"][map_name])
            map_doc = load_json(map_path)
            coarse_ids = [row["coarse_id"] for row in map_doc["coarse_regions"]]
            canonical_payload = {
                "question": question,
                "requirements": requirements,
                "navigation_map": v661._base._planner_map(map_doc),
            }
            provider_payload, _ = v661._base._local_indexed_planner_contract(canonical_payload, coarse_ids)
            user_text = json.dumps(provider_payload, ensure_ascii=False, separators=(",", ":"))
            encoded_chat = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user_text}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            input_ids = encoded_chat["input_ids"] if hasattr(encoded_chat, "keys") else encoded_chat
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            input_tokens = len(input_ids)
            if input_tokens < 100:
                raise RuntimeError(f"invalid offline token count for {question['question_id']}/{side}: {input_tokens}")
            token_rows.append({
                "question_id": question["question_id"], "route": side,
                "map_sha256": sha256_file(map_path), "input_tokens": input_tokens,
                "planner_max_output_tokens": output_tokens,
                "total_reserved_tokens": input_tokens + output_tokens,
                "effective_max_model_len": max_model_len,
                "fits": input_tokens + output_tokens <= max_model_len,
            })
    overflow = [row for row in token_rows if not row["fits"]]
    calibration = []
    calibration_question_id = cfg["gate_question_id"]
    for side in SIDES:
        v1_planner = root / "outputs/experiments/hourvideo_v6_6_2_formal_eval300_v1/frozen_planner/cases" / calibration_question_id / side / "planner.json"
        if not v1_planner.is_file():
            raise FileNotFoundError(f"missing retained v1 token calibration artifact: {v1_planner}")
        recorded = int(load_json(v1_planner)["usage"]["input_tokens"])
        calculated = next(row["input_tokens"] for row in token_rows if row["question_id"] == calibration_question_id and row["route"] == side)
        calibration.append({"question_id": calibration_question_id, "route": side, "recorded_v1_input_tokens": recorded, "offline_calculated_input_tokens": calculated, "exact_match": recorded == calculated})
    calibration_pass = all(row["exact_match"] for row in calibration)
    report = {
        "schema_version": "v6_6_2_planner_8192_offline_gate_v1",
        "created_at_utc": _now(), "model_started": False,
        "config_difference_gate": "pass", "non_whitelisted_semantic_differences": [],
        "planner_prompt_sha256": planner_prompt_sha,
        "required_planner_prompt_sha256": required_planner_prompt_sha,
        "biased_prompt_terms": biased_terms, "indexer_neutral_prompt_gate": "pass" if interface_neutral else "fail",
        "representation_audit": representation_audit,
        "route_count": len(token_rows), "r1_count": sum(row["route"] == "r1_av" for row in token_rows),
        "r3_count": sum(row["route"] == "r3_2" for row in token_rows),
        "max_input_tokens": max(row["input_tokens"] for row in token_rows),
        "max_total_reserved_tokens": max(row["total_reserved_tokens"] for row in token_rows),
        "overflow_count": len(overflow), "context_capacity_gate": "pass" if not overflow else "fail",
        "tokenizer_calibration": calibration, "tokenizer_calibration_gate": "pass" if calibration_pass else "fail",
        "overflow_routes": overflow, "routes": token_rows,
    }
    output = root / "outputs/experiments/hourvideo_v6_6_2_formal_eval300_v2/validation"
    write_json(output / "planner_8192_offline_gate.json", report)
    if not interface_neutral:
        raise RuntimeError("Planner prompt/representation interface gate failed")
    if not calibration_pass:
        raise RuntimeError("offline Planner tokenizer does not match retained v1 request telemetry")
    if overflow:
        raise RuntimeError(f"Planner 8192 context capacity gate failed for {len(overflow)} routes")
    return report


def generate_planners(
    root: Path, config_path: Path, identity_filter: str | None = None,
    sides: tuple[str, ...] = SIDES,
) -> dict[str, Any]:
    """Generate label-blind, route-specific Frozen Planners with attempt telemetry."""
    cfg = load_json(config_path)
    planner_root = root / cfg["planner_source_experiment"]
    telemetry = planner_root / "planner_attempts.jsonl"
    base = v661._base
    original_call = base._anthropic_call
    active: dict[str, Any] = {}

    def instrumented_call(call_cfg, system, payload, schema, max_tokens):
        request_uuid = str(uuid.uuid4())
        started = time.monotonic()
        _append_fsync(telemetry, {
            "event": "request_start", "request_uuid": request_uuid, "written_at_utc": _now(),
            **active, "model": call_cfg["anthropic"]["model"],
            "payload_sha256": _json_sha(payload), "schema_sha256": _json_sha(schema),
        })
        try:
            parsed, usage = original_call(call_cfg, system, payload, schema, max_tokens)
        except Exception as error:
            usage = getattr(error, "attempt_telemetry", {})
            raw_text = str(usage.get("raw_text") or "")
            _append_fsync(telemetry, {
                "event": "attempt_end", "request_uuid": request_uuid, "written_at_utc": _now(),
                **active, "status": "failed", "failure_reason": f"{type(error).__name__}: {error}",
                "e2e_sec": time.monotonic() - started,
                **{key: usage.get(key) for key in ("model", "input_tokens", "output_tokens", "latency_sec", "response_id", "stop_reason")},
                "raw_model_response": raw_text,
                "raw_model_response_sha256": hashlib.sha256(raw_text.encode()).hexdigest() if raw_text else None,
            })
            raise
        raw_text = str(usage.get("raw_text") or "")
        _append_fsync(telemetry, {
            "event": "attempt_end", "request_uuid": request_uuid, "written_at_utc": _now(),
            **active, "status": "accepted", "failure_reason": None,
            "e2e_sec": time.monotonic() - started,
            **{key: usage.get(key) for key in ("model", "input_tokens", "output_tokens", "latency_sec", "response_id", "stop_reason")},
            "raw_model_response": raw_text,
            "raw_model_response_sha256": hashlib.sha256(raw_text.encode()).hexdigest() if raw_text else None,
        })
        return parsed, usage

    base._anthropic_call = instrumented_call
    generated = 0
    reused = 0
    try:
        for case_cfg, case_dir in v661._case_paths(cfg, root, identity_filter):
            question = load_json(case_dir / "question_input.json")
            requirements = v661.option_requirements(question)
            for side in sides:
                output_path = planner_root / "cases" / question["question_id"] / side / "planner.json"
                if output_path.is_file():
                    reused += 1
                    continue
                map_name = "r1_av_navigation_map.json" if side == "r1_av" else "r3_2_navigation_map.json"
                map_path = Path(case_cfg["asset_paths"][map_name])
                map_doc = load_json(map_path)
                coarse_ids = [row["coarse_id"] for row in map_doc["coarse_regions"]]
                active.clear(); active.update({
                    "question_id": question["question_id"], "video_uid": case_cfg["video_uid"],
                    "route": side, "input_asset_path": str(map_path),
                    "input_asset_sha256": sha256_file(map_path),
                })
                plan, usage, payload = base._call_coarse_locked_planner(
                    cfg, question, requirements, coarse_ids, map_doc,
                )
                base._validate_coarse_locked_plan(plan, question, requirements, set(coarse_ids))
                write_json(output_path, {
                    "schema_version": "v6_6_2_frozen_planner_v1", "created_at_utc": _now(),
                    "question_id": question["question_id"], "video_uid": case_cfg["video_uid"],
                    "route": side, "input_asset_path": str(map_path),
                    "input_asset_sha256": sha256_file(map_path), "input_payload_sha256": _json_sha(payload),
                    "planner_model": cfg["anthropic"]["model"], "provider": cfg["anthropic"]["provider"],
                    "temperature": cfg["anthropic"]["temperature"],
                    "max_tokens": cfg["anthropic"]["planner_max_tokens"],
                    "prompt_sha256": hashlib.sha256(base.COARSE_LOCKED_PLANNER_PROMPT.encode()).hexdigest(),
                    "planner_code_sha256": sha256_file(Path(base.__file__)),
                    "output": plan, "usage": usage, "gold_loaded": False,
                })
                generated += 1
    finally:
        base._anthropic_call = original_call
    attempt_rows = []
    if telemetry.is_file():
        attempt_rows = [json.loads(line) for line in telemetry.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = [row for row in attempt_rows if row.get("event") == "attempt_end"]
    result = {
        "generated": generated, "reused": reused, "total_present": generated + reused,
        "actual_request_count": len(completed),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in completed),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in completed),
        "request_latency_sec": sum(float(row.get("latency_sec") or 0.0) for row in completed),
        "planner_generation_wall_time_not_post_planner_e2e": True,
        "gpu_allocation_cost_formula": "request/elapsed GPU-hours x institutional GPU-hour rate",
        "post_planner_online_cost_included": False,
    }
    write_json(planner_root / "planner_generation_summary.json", result)
    return result


def planner_canary(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    question_id = cfg["gate_question_id"]
    result = generate_planners(root, config_path, question_id, SIDES)
    cases = v661._case_paths(cfg, root, question_id)
    if len(cases) != 1:
        raise ValueError("Planner canary question identity is not unique")
    case_cfg, case_dir = cases[0]
    question = load_json(case_dir / "question_input.json")
    routes = []
    for side in SIDES:
        path = root / cfg["planner_source_experiment"] / "cases" / question_id / side / "planner.json"
        doc = load_json(path)
        map_name = "r1_av_navigation_map.json" if side == "r1_av" else "r3_2_navigation_map.json"
        map_path = Path(case_cfg["asset_paths"][map_name])
        coarse_ids = {row["coarse_id"] for row in load_json(map_path)["coarse_regions"]}
        v661._base._validate_coarse_locked_plan(doc["output"], question, v661.option_requirements(question), coarse_ids)
        routes.append({"question_id": question_id, "route": side, "status": "contract_valid", "planner_sha256": sha256_file(path)})
    report = {"gold_loaded": False, "correctness_checked": False, "generation": result, "routes": routes}
    write_json(root / cfg["planner_source_experiment"] / "planner_canary_report.json", report)
    return report


def validate_planner_population(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    rows = []
    for case_cfg, case_dir in v661._case_paths(cfg, root, None):
        question = load_json(case_dir / "question_input.json")
        for side in SIDES:
            path = root / cfg["planner_source_experiment"] / "cases" / question["question_id"] / side / "planner.json"
            doc = load_json(path)
            map_name = "r1_av_navigation_map.json" if side == "r1_av" else "r3_2_navigation_map.json"
            map_path = Path(case_cfg["asset_paths"][map_name])
            coarse_ids = {row["coarse_id"] for row in load_json(map_path)["coarse_regions"]}
            v661._base._validate_coarse_locked_plan(doc["output"], question, v661.option_requirements(question), coarse_ids)
            if doc["input_asset_sha256"] != sha256_file(map_path):
                raise ValueError(f"Planner input asset mismatch: {question['question_id']}/{side}")
            rows.append({"question_id": question["question_id"], "route": side, "sha256": sha256_file(path)})
    keys = {(row["question_id"], row["route"]) for row in rows}
    if len(rows) != 600 or len(keys) != 600:
        raise ValueError(f"Frozen Planner coverage is {len(rows)}/600")
    result = {"coverage": "600/600", "unique_route_keys": len(keys), "rows": rows}
    write_json(root / cfg["planner_source_experiment"] / "planner_manifest.json", result)
    return result


def build_closure_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    """Freeze the static local import closure plus formal inputs and generated Planners."""
    cfg = load_json(config_path)
    seeds = [
        root / "src/experiments/hourvideo_v6_6_2_shared_coarse_contract_v1/core.py",
        root / "src/experiments/hourvideo_v6_6_2_shared_coarse_contract_v1/formal_eval300.py",
    ]
    queue = list(seeds)
    code_files: set[Path] = set()
    while queue:
        path = queue.pop()
        if path in code_files or not path.is_file():
            continue
        code_files.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module_parts = list(path.relative_to(root / "src").with_suffix("").parts)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = module_parts[:-node.level]
                    module = ".".join(base + ([node.module] if node.module else []))
                else:
                    module = node.module or ""
                names = [module]
            for name in names:
                if not name.startswith("experiments"):
                    continue
                candidate = root / "src" / Path(*name.split("."))
                resolved = candidate.with_suffix(".py") if candidate.with_suffix(".py").is_file() else candidate / "__init__.py"
                if resolved.is_file() and resolved not in code_files:
                    queue.append(resolved)
    experiment_root = (root / cfg["output_root"]).parent
    extra_files = {
        config_path,
        root / "scripts/experiments/run_hourvideo_v6_6_2_formal_eval300_v1.py",
        root / "scripts/experiments/serve_hourvideo_v6_6_2_eval300_gate.sh",
        experiment_root / "source/source_manifest.json",
        experiment_root / "source/ordered_routes.jsonl",
        Path(cfg["formal_r3_source_manifest"]),
        Path(cfg["formal_r3_root"]) / "manifests/materialization_manifest.json",
        Path(cfg["formal_r3_archive_manifest"]),
    }
    extra_files.update((root / "tests/experiments/hourvideo_v6_6_2_shared_coarse_contract_v1").glob("*.py"))
    extra_files.update((root / "tests/experiments/hourvideo_v6_6_1_contract_telemetry_v1").glob("*.py"))
    extra_files.update(path for path in (experiment_root / "validation").glob("*") if path.is_file())
    planner_root = root / cfg["planner_source_experiment"]
    if planner_root.is_dir():
        extra_files.update(path for path in planner_root.rglob("*.json") if path.is_file())
        extra_files.update(path for path in planner_root.rglob("*.jsonl") if path.is_file())
    files = sorted(code_files | {path for path in extra_files if path.is_file()}, key=str)
    rows = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in files]
    tree_sha = hashlib.sha256("".join(f"{row['sha256']}  {row['path']}\n" for row in rows).encode()).hexdigest()
    result = {
        "schema_version": "v6_6_2_formal_eval300_reproducibility_closure_v1",
        "created_at_utc": _now(), "file_count": len(rows), "tree_sha256": tree_sha,
        "actual_static_local_import_closure": True, "top_level_only": False,
        "prompt_sha256_by_stage": {
            **v661._prompt_hashes(),
            "planner": hashlib.sha256(v661._base.COARSE_LOCKED_PLANNER_PROMPT.encode()).hexdigest(),
        },
        "formal_r3_lock": verify_locked_r3(cfg), "files": rows,
    }
    output = experiment_root / "manifest"
    write_json(output / "closure_manifest.json", result)
    return result
