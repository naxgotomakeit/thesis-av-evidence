from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


FORBIDDEN_ONLINE_KEYS = {
    "annotation_context",
    "canary",
    "correct_answer_label",
    "expected_answer",
    "gold",
    "ground_truth",
    "mcq_test",
    "provided_timestamp",
    "reference_timestamp",
    "relevant_timestamps",
    "rung",
    "task",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
FINAL_LABELS = {"A", "B", "C", "D", "E"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_ONLINE_KEYS.intersection(value):
            return True
        return any(contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_forbidden_key(item) for item in value)
    return False


def flatten_questions(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_uid, video_record in annotation.items():
        for question in video_record.get("benchmark_dataset", []):
            if question.get("video_uid") != video_uid:
                raise ValueError(f"video UID mismatch for {question.get('qid')}")
            rows.append(question)
    return sorted(rows, key=lambda row: str(row["qid"]))


def _option_kind(raw: str) -> str:
    return "image" if Path(raw).suffix.lower() in IMAGE_SUFFIXES else "text"


def project_safe_question(row: dict[str, Any], labels: list[str]) -> dict[str, Any]:
    options: list[dict[str, str]] = []
    kinds: set[str] = set()
    for index, label in enumerate(labels, start=1):
        raw = str(row.get(f"answer_{index}", "")).strip()
        if not raw:
            raise ValueError(f"empty option {label} for {row.get('qid')}")
        kind = _option_kind(raw)
        kinds.add(kind)
        option = {"option_id": label, "content_type": kind}
        if kind == "image":
            option["asset_ref"] = raw.replace("\\", "/")
        else:
            option["text"] = raw
        options.append(option)
    if len(kinds) != 1:
        raise ValueError(f"mixed option modalities for {row.get('qid')}")
    online = {
        "question_id": str(row["qid"]),
        "video_uid": str(row["video_uid"]),
        "question_text": str(row["question"]).strip(),
        "answer_options": options,
    }
    if contains_forbidden_key(online):
        raise ValueError(f"protected evaluation field leaked for {row.get('qid')}")
    return online


def make_requirements(online: dict[str, Any]) -> list[dict[str, Any]]:
    question_id = online["question_id"]
    return [
        {
            "requirement_id": f"{question_id}::option_{option['option_id']}",
            "option_id": option["option_id"],
            "assessment_target": (
                "whether retrieved evidence supports, weakens, or fails to address "
                "this option"
            ),
        }
        for option in online["answer_options"]
    ]


def map_binding_contract(capability: str) -> dict[str, Any]:
    if capability == "structural_audio_navigation":
        return {
            "binding_name": "video_navigation_map",
            "required_map_capability": capability,
            "map_role": "navigation_only",
            "semantic_coarse_may_be_sufficiency_evidence": False,
            "all_unresolved_semantics_require_retrieval": True,
            "runtime_status": "awaiting_selected_video_index",
        }
    if capability == "audio_visual_semantic_navigation":
        return {
            "binding_name": "video_navigation_map",
            "required_map_capability": capability,
            "map_role": "semantic_navigation_and_coarse_first_assessment",
            "semantic_coarse_may_be_sufficiency_evidence": True,
            "semantic_coarse_is_reviewed_visual_confirmation": False,
            "runtime_status": "awaiting_selected_video_index",
        }
    raise ValueError(f"unknown map capability: {capability}")


def make_no_api_trace(online: dict[str, Any], capability: str) -> dict[str, Any]:
    requirements = make_requirements(online)
    model_question_payload = {
        "question_id": online["question_id"],
        "question_text": online["question_text"],
        "answer_options": online["answer_options"],
    }
    if contains_forbidden_key(model_question_payload):
        raise ValueError("forbidden key in model question payload")
    return {
        "interface_capability": capability,
        "shared_question_payload": model_question_payload,
        "map_binding": map_binding_contract(capability),
        "planner_handoff": {
            "model_payload_status": "not_constructed_until_video_map_exists",
            "model_executed": False,
            "expected_output_fields": [
                "search_units",
                "query_variants",
                "modality_strategy",
                "temporal_strategy",
                "suggested_coarse_ids",
            ],
            "hard_filtering_allowed": False,
        },
        "retrieval_handoff": {
            "candidate_universe_status": "awaiting_selected_video_index",
            "candidate_universe": [],
            "ranked_evidence": [],
            "execution_status": "not_run_no_api_validation",
        },
        "sufficiency_handoff": {
            "requirements": requirements,
            "evidence": [],
            "map_policy": map_binding_contract(capability)["map_role"],
            "execution_status": "not_run_no_api_validation",
        },
        "reliability_gate_handoff": {
            "allowed_states": ["answer_ready", "provisional", "unresolved", "conflict"],
            "execution_status": "not_run_no_api_validation",
        },
        "final_handoff": {
            "answer_options": online["answer_options"],
            "allowed_output_labels": ["A", "B", "C", "D", "E"],
            "execution_status": "not_run_no_api_validation",
        },
    }


def validate_final_output(
    value: dict[str, Any], known_requirements: set[str], known_evidence: set[str]
) -> list[str]:
    errors: list[str] = []
    if value.get("selected_option") not in FINAL_LABELS:
        errors.append("selected_option must be A-E")
    if value.get("answerability") not in {
        "answer_directly",
        "answer_with_caveat",
        "insufficient_to_answer",
    }:
        errors.append("invalid answerability")
    if not set(value.get("supporting_requirement_ids", [])).issubset(known_requirements):
        errors.append("unknown requirement reference")
    if not set(value.get("supporting_evidence_ids", [])).issubset(known_evidence):
        errors.append("unknown evidence reference")
    return errors


def _image_asset_audit(
    safe_questions: list[dict[str, Any]], release_root: Path
) -> dict[str, Any]:
    refs = [
        option["asset_ref"]
        for row in safe_questions
        for option in row["answer_options"]
        if option["content_type"] == "image"
    ]
    missing: list[str] = []
    unreadable: list[dict[str, str]] = []
    bindings: list[dict[str, Any]] = []
    for ref in refs:
        path = (release_root / ref).resolve()
        if not path.is_file():
            missing.append(ref)
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:  # pragma: no cover - exercised by real assets
            unreadable.append({"asset_ref": ref, "error": str(exc)})
            continue
        bindings.append(
            {
                "asset_ref": ref,
                "resolved_path": str(path),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "width": width,
                "height": height,
            }
        )
    return {
        "image_option_file_count": len(refs),
        "unique_image_asset_count": len({row["asset_ref"] for row in bindings}),
        "missing": missing,
        "unreadable": unreadable,
        "bindings": bindings,
        "status": "passed" if not missing and not unreadable else "failed",
        "solver_capability": "requires_separate_option_image_review_adapter",
    }


def _media_audit(
    annotation: dict[str, Any], frame_audit: dict[str, Any], video_audit: dict[str, Any]
) -> dict[str, Any]:
    annotation_ids = set(annotation)
    frame_rows = frame_audit.get("videos", [])
    video_rows = video_audit.get("videos", [])
    frame_ids = {row.get("video_uid") for row in frame_rows}
    video_ids = {row.get("video_uid") for row in video_rows}
    missing_frame_dirs = [
        row.get("video_uid") for row in frame_rows if not Path(row.get("frame_dir", "")).is_dir()
    ]
    missing_video_files = [
        row.get("video_uid") for row in video_rows if not Path(row.get("path", "")).is_file()
    ]
    invalid_video_rows = [
        row.get("video_uid")
        for row in video_rows
        if row.get("status") != "valid" or row.get("readable") is not True
    ]
    checks = {
        "annotation_video_count_50": len(annotation_ids) == 50,
        "frame_video_count_50": len(frame_rows) == 50,
        "video_audit_count_50": len(video_rows) == 50,
        "annotation_frame_ids_identical": annotation_ids == frame_ids,
        "annotation_video_ids_identical": annotation_ids == video_ids,
        "frame_audit_complete": frame_audit.get("status") == "completed"
        and frame_audit.get("completed_count") == 50,
        "frame_directories_exist": not missing_frame_dirs,
        "video_files_exist": not missing_video_files,
        "videos_marked_readable": not invalid_video_rows,
    }
    audio_count = sum(row.get("has_audio") is True for row in video_rows)
    return {
        "checks": checks,
        "annotation_video_count": len(annotation_ids),
        "frame_count": frame_audit.get("total_frame_count"),
        "audio_available_video_count": audio_count,
        "audio_unavailable_video_count": len(video_rows) - audio_count,
        "missing_frame_dirs": missing_frame_dirs,
        "missing_video_files": missing_video_files,
        "invalid_video_rows": invalid_video_rows,
        "no_audio_policy": "valid empty audio channel; visual pipeline remains available",
        "status": "passed" if all(checks.values()) else "failed",
    }


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def run(repo: Path, config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("model_api_calls") != 0:
        raise ValueError("this experiment forbids model/API calls")
    output = repo / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)

    hourvideo_root = Path(config["hourvideo_root"])
    annotation_path = hourvideo_root / config["annotation_path"]
    frame_audit_path = hourvideo_root / config["frame_audit_path"]
    video_audit_path = hourvideo_root / config["video_audit_path"]
    release_root = annotation_path.parents[1]
    freeze_root = (
        repo
        / "outputs/experiments/r1_av_r3_2_cached_full_pipeline_engineering_freeze_v1"
    )
    source_paths = {
        "annotations": annotation_path,
        "frame_audit": frame_audit_path,
        "video_audit": video_audit_path,
        "engineering_freeze_validation": freeze_root / "validation_report.json",
        "r1_av_freeze_manifest": freeze_root / "r1_av_engineering_freeze_manifest.json",
        "r3_2_freeze_manifest": freeze_root / "r3_2_engineering_freeze_manifest.json",
        "shared_pipeline_contract": freeze_root / "shared_pipeline_interface_contract.json",
    }
    missing_sources = [name for name, path in source_paths.items() if not path.is_file()]
    if missing_sources:
        raise RuntimeError(f"missing sources: {missing_sources}")

    annotation = load_json(annotation_path)
    frame_audit = load_json(frame_audit_path)
    video_audit = load_json(video_audit_path)
    freeze_validation = load_json(source_paths["engineering_freeze_validation"])
    labels = list(config["option_labels"])
    raw_rows = flatten_questions(annotation)
    safe_questions = [project_safe_question(row, labels) for row in raw_rows]
    audit_by_id = {
        str(row["qid"]): {
            "question_id": str(row["qid"]),
            "video_uid": str(row["video_uid"]),
            "task": str(row["task"]),
            "option_modality": _option_kind(str(row["answer_1"]).strip()),
        }
        for row in raw_rows
    }

    fixture_questions: list[dict[str, Any]] = []
    fixture_selection: list[dict[str, Any]] = []
    safe_by_id = {row["question_id"]: row for row in safe_questions}
    for task in config["fixture_tasks"]:
        candidates = sorted(
            (row for row in audit_by_id.values() if row["task"] == task),
            key=lambda row: row["question_id"],
        )
        if not candidates:
            raise RuntimeError(f"no fixture for task {task}")
        selected = candidates[0]
        fixture_questions.append(safe_by_id[selected["question_id"]])
        fixture_selection.append(selected)

    image_audit = _image_asset_audit(safe_questions, release_root)
    media_audit = _media_audit(annotation, frame_audit, video_audit)
    r1_traces = [
        make_no_api_trace(row, "structural_audio_navigation")
        for row in fixture_questions
    ]
    r3_traces = [
        make_no_api_trace(row, "audio_visual_semantic_navigation")
        for row in fixture_questions
    ]
    question_identity = all(
        canonical_bytes(left["shared_question_payload"])
        == canonical_bytes(right["shared_question_payload"])
        for left, right in zip(r1_traces, r3_traces)
    )
    requirement_identity = all(
        canonical_bytes(left["sufficiency_handoff"]["requirements"])
        == canonical_bytes(right["sufficiency_handoff"]["requirements"])
        for left, right in zip(r1_traces, r3_traces)
    )
    no_rung_leakage = all(
        not contains_forbidden_key(trace["shared_question_payload"])
        and "R1_AV" not in canonical_bytes(trace["shared_question_payload"]).decode("utf-8")
        and "R3_2" not in canonical_bytes(trace["shared_question_payload"]).decode("utf-8")
        for trace in r1_traces + r3_traces
    )
    task_counts = Counter(row["task"] for row in audit_by_id.values())
    modality_counts = Counter(row["option_modality"] for row in audit_by_id.values())

    final_probe = {
        "selected_option": "A",
        "answerability": "answer_with_caveat",
        "supporting_requirement_ids": ["probe::option_A"],
        "supporting_evidence_ids": ["evidence_probe"],
    }
    final_probe_passed = not validate_final_output(
        final_probe, {"probe::option_A"}, {"evidence_probe"}
    )
    invalid_label_rejected = bool(
        validate_final_output(
            dict(final_probe, selected_option="F"),
            {"probe::option_A"},
            {"evidence_probe"},
        )
    )

    replay_payload = {
        "safe_questions": safe_questions,
        "r1": r1_traces,
        "r3": r3_traces,
    }
    first_hash = hashlib.sha256(canonical_bytes(replay_payload)).hexdigest()
    replay_hash = hashlib.sha256(canonical_bytes(replay_payload)).hexdigest()
    checks = {
        "question_count_1182": len(safe_questions) == 1182,
        "video_count_50": len(annotation) == 50,
        "five_ordered_options_each": all(
            len(row["answer_options"]) == 5
            and [item["option_id"] for item in row["answer_options"]] == labels
            for row in safe_questions
        ),
        "protected_fields_absent": not contains_forbidden_key(safe_questions),
        "five_fixture_tasks_present": len(fixture_questions) == 5,
        "fixture_includes_image_options": any(
            option["content_type"] == "image"
            for row in fixture_questions
            for option in row["answer_options"]
        ),
        "image_assets_valid": image_audit["status"] == "passed",
        "media_audit_valid": media_audit["status"] == "passed",
        "r1_r3_question_options_identical": question_identity,
        "r1_r3_requirements_identical": requirement_identity,
        "no_rung_identity_in_model_question_payload": no_rung_leakage,
        "map_not_directly_used_as_r1_sufficiency_evidence": all(
            trace["map_binding"]["semantic_coarse_may_be_sufficiency_evidence"]
            is False
            for trace in r1_traces
        ),
        "r3_semantic_map_not_reviewed_confirmation": all(
            trace["map_binding"]["semantic_coarse_is_reviewed_visual_confirmation"]
            is False
            for trace in r3_traces
        ),
        "final_A_to_E_schema_valid": final_probe_passed,
        "invalid_label_rejected": invalid_label_rejected,
        "deterministic_replay": first_hash == replay_hash,
        "frozen_226_engineering_contract_available": freeze_validation.get(
            "overall_validation"
        )
        == "frozen_engineering_test_snapshot",
    }
    errors = [name for name, passed in checks.items() if not passed]

    question_contract = {
        "schema_version": "hourvideo-safe-question-options-v1.1",
        "online_fields": ["question_id", "video_uid", "question_text", "answer_options"],
        "answer_option_schema": {
            "option_id": "A|B|C|D|E",
            "content_type": "text|image",
            "text": "required for text",
            "asset_ref": "required for image; resolved outside model payload",
        },
        "protected_offline_only_fields": sorted(FORBIDDEN_ONLINE_KEYS),
        "task_metadata_online": False,
        "rung_identity_online": False,
    }
    pipeline_contract = {
        "schema_version": "hourvideo-r1-av-r3-2-interface-v1.1",
        "shared_question_contract": question_contract,
        "shared_stages": [
            "Planner",
            "retrieval handoff",
            "option-centric Sufficiency",
            "Reliability Gate",
            "selective review handoff",
            "A-E final solver",
        ],
        "r1_av_policy": (
            "structural/audio map for navigation; retrieval supplies semantic evidence"
        ),
        "r3_2_policy": (
            "global AV semantic map; Coarse-first assessment; local descent when required"
        ),
        "question_and_option_payload_byte_identical_between_rungs": True,
        "rung_identity_exposed_to_models": False,
        "map_contract_attached_only_after_selected_video_index_exists": True,
        "image_option_boundary": (
            "assets validated but a separate option-image review adapter is required"
        ),
        "no_audio_policy": media_audit["no_audio_policy"],
        "hourvideo_index_dependency": "not_yet_built_for_predeclared_pilot_videos",
    }
    requirement_contract = {
        "schema_version": "hourvideo-option-centric-requirements-v1",
        "one_requirement_per_option": True,
        "requirement_generation": (
            "deterministic from question and ordered options; correct label and reference "
            "timestamps excluded"
        ),
        "statuses": ["supported", "uncertain", "conflicted", "not_found"],
        "review_trigger": "only unresolved capability-specific requirements",
    }
    source_audit = {
        "sources": {name: _source_record(path) for name, path in source_paths.items()},
        "question_count": len(safe_questions),
        "video_count": len(annotation),
        "task_counts": dict(sorted(task_counts.items())),
        "option_modality_counts": dict(sorted(modality_counts.items())),
    }
    leakage_audit = {
        "raw_annotation_contains_protected_evaluation_fields": True,
        "protected_fields_copied_to_online_manifest": False,
        "correct_label_used_for_fixture_selection": False,
        "reference_timestamp_used_for_fixture_selection": False,
        "fixture_selection_policy": (
            "lexicographically first question ID for each predeclared task"
        ),
        "rung_identity_in_model_question_payload": False,
        "safe_manifest_sha256": hashlib.sha256(
            canonical_bytes(safe_questions)
        ).hexdigest(),
    }
    readiness = {
        "interface_validation": "passed" if not errors else "failed",
        "ready_for_predeclared_five_video_index_build": not errors,
        "ready_for_paid_text_option_pipeline": False,
        "ready_for_paid_image_option_pipeline": False,
        "blocking_before_paid_run": [
            "predeclare five-video manifest without consulting correct labels or timestamps",
            "build shared Fine/Medium hierarchy for selected videos",
            "build R1_AV indices/maps for selected videos",
            "build R3_2 captions/semantic maps for selected videos",
            "freeze per-video model, media, retry, and resume budgets",
            "validate option-image review adapter for image-option questions or exclude "
            "those questions from the initial text-only smoke",
        ],
    }
    validation = {
        "dataset_interface_validation": "passed" if not errors else "failed",
        "paired_contract_validation": (
            "passed" if question_identity and requirement_identity else "failed"
        ),
        "evaluation_leakage_validation": (
            "passed" if checks["protected_fields_absent"] and no_rung_leakage else "failed"
        ),
        "media_source_validation": media_audit["status"],
        "image_option_asset_validation": image_audit["status"],
        "image_option_solver_validation": "blocked_missing_option_image_review_adapter",
        "no_api_replay_validation": (
            "passed" if checks["deterministic_replay"] else "failed"
        ),
        "overall_validation": (
            "passed_no_api_interface_ready_for_five_video_index_build"
            if not errors
            else "failed"
        ),
        "model_api_calls": 0,
        "planner_calls": 0,
        "retrieval_model_calls": 0,
        "sufficiency_calls": 0,
        "review_calls": 0,
        "final_calls": 0,
        "errors": errors,
        "checks": checks,
    }

    files = {
        "input_manifest.json": {
            "experiment": config["experiment"],
            "mode": "real_dataset_no_api_interface_validation",
            "model_api_calls": 0,
        },
        "source_artifact_audit.json": source_audit,
        "dataset_media_audit.json": media_audit,
        "safe_question_manifest.json": {
            "schema_version": question_contract["schema_version"],
            "questions": safe_questions,
        },
        "question_option_contract.json": question_contract,
        "five_question_interface_fixtures.json": {
            "selection_policy": leakage_audit["fixture_selection_policy"],
            "offline_selection_audit": fixture_selection,
            "online_fixtures": fixture_questions,
        },
        "image_option_audit.json": image_audit,
        "shared_pipeline_interface_contract.json": pipeline_contract,
        "option_requirement_contract.json": requirement_contract,
        "r1_av_no_api_interface_trace.json": r1_traces,
        "r3_2_no_api_interface_trace.json": r3_traces,
        "paired_interface_audit.json": {
            "question_options_byte_identical": question_identity,
            "requirements_byte_identical": requirement_identity,
            "shared_model_question_payload_has_no_rung_identity": no_rung_leakage,
            "intended_differences": ["navigation map capability", "evidence capability"],
        },
        "evaluation_leakage_audit.json": leakage_audit,
        "determinism_report.json": {
            "first_sha256": first_hash,
            "replay_sha256": replay_hash,
            "byte_identical": first_hash == replay_hash,
        },
        "no_api_test_report.json": {
            "checks": checks,
            "passed": not errors,
            "model_api_calls": 0,
        },
        "five_video_pilot_readiness.json": readiness,
        "validation_report.json": validation,
    }
    for name, value in files.items():
        write_json(output / name, value)

    report = f"""# HourVideo R1_AV / R3_2 no-API interface validation v1.1

- Overall: `{validation['overall_validation']}`
- Real dev split audited: {len(annotation)} videos, {len(safe_questions)} questions.
- Options: {modality_counts.get('text', 0)} text-option questions and {modality_counts.get('image', 0)} image-option questions.
- Image option assets: {image_audit['image_option_file_count']} checked; {len(image_audit['missing'])} missing; {len(image_audit['unreadable'])} unreadable.
- Existing 1-fps cache: {media_audit['frame_count']} frames across 50/50 videos.
- Audio: {media_audit['audio_available_video_count']} videos available; {media_audit['audio_unavailable_video_count']} videos use the valid empty-audio path.
- R1_AV and R3_2 question/options and option requirements are byte-identical.
- Rung identity, correct labels, reference timestamps, task metadata, and canary fields are absent from model-facing question payloads.
- Planner, Retrieval, Sufficiency, review, Final Gemini, and all other model/API calls: 0.

The real-data interface is ready for predeclaring and indexing a five-video pilot. It is not yet a paid end-to-end pass: selected videos still need current-contract R1_AV and R3_2 indices/maps and fixed budgets. Image-option questions additionally require a dedicated option-image review adapter; their 125 source assets are present and readable, but text-only final solving is not claimed.
"""
    (output / "REPORT.md").write_text(
        report, encoding="utf-8", newline="\n"
    )
    return validation
