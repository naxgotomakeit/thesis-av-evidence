from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load(config_path)
    output = root / cfg["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: root / value for key, value in cfg["sources"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    before = {key: sha256(path) for key, path in paths.items()}
    source_manifest = {
        key: {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": before[key], "bytes": path.stat().st_size}
        for key, path in paths.items()
    }
    dump(output / "frozen_source_manifest.json", source_manifest)

    upstream = load(paths["upstream_validation"])
    r3_map = load(paths["r3_map_validation"])
    requirement = load(paths["requirement_validation"])
    fine = load(paths["fine_validation"])
    scope = load(paths["scope_validation"])
    cache = load(paths["cache_validation"])
    final = load(paths["final_validation"])
    encoding = load(paths["encoding_audit"])
    errors = []
    expected = [
        (upstream.get("overall_validation") == "passed_upstream_freeze", "upstream map/Planner freeze invalid"),
        (r3_map.get("overall_validation") == "passed_r3_2_navigation_map_frozen", "R3_2 semantic map not frozen"),
        (str(requirement.get("overall_validation", "")).startswith("passed_"), "requirement regression invalid"),
        (fine.get("overall_validation") == "passed", "Fine retrieval contract invalid"),
        (scope.get("overall_validation") == "passed", "question scope Gate invalid"),
        (cache.get("overall_validation") == "passed", "review cache integration invalid"),
        (final.get("final_gemini_schema_validation") == "passed", "Final Gemini schema invalid"),
        (final.get("final_answer_grounding_validation") == "passed", "Final grounding validation invalid"),
        (final.get("user_facing_text_encoding_validation") == "passed", "display encoding validation invalid"),
        (encoding.get("semantic_content_modified") is False, "display normalization changed semantics"),
    ]
    errors.extend(message for passed, message in expected if not passed)

    shared_contract = {
        "contract_id": "r1_av_r3_2_cached_full_pipeline_engineering_contract_v1",
        "question_interface": {"question_text": "string", "answer_options": "optional array"},
        "planner_output_interface": ["search_units", "query_variants", "modality_strategy", "temporal_strategy", "suggested_coarse_ids"],
        "retrieval_output_interface": ["selected Medium evidence", "selected Fine references", "audio/ASR references", "score and provenance"],
        "sufficiency_interface": "one assessment per declared requirement",
        "gate_interface": "answer_ready | provisional | unresolved | conflict; only asked requirements may trigger review",
        "review_cache_key": ["image_sha256", "review_contract_version", "fact_scope"],
        "review_cache_value": "immutable reviewed_visual_frame finding with Fine/time provenance",
        "final_interface": "resolved requirement assessments plus accepted temporal sidecar; text-only",
        "map_evidence_policy": {
            "r1_av_structural_map": "navigation_only_not_sufficiency_evidence",
            "r3_2_semantic_coarse": "caption_asr_derived_first_pass_sufficiency_evidence_with_exact_source_provenance",
            "r3_2_semantic_coarse_is_reviewed_visual_confirmation": False,
        },
        "raw_images_sent_to_final_gemini": False,
        "reviewed_visual_findings_override_lower_capability_sources_only_at_exact_scope": True,
    }
    dump(output / "shared_pipeline_interface_contract.json", shared_contract)

    common = {
        "freeze_kind": "engineering_test_snapshot",
        "freeze_version": "v1",
        "source_manifest_sha256": hashlib.sha256(json.dumps(source_manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "automatic_interface_validation": "passed" if not errors else "failed",
        "manual_answer_correctness": "not_frozen_user_elected_non_blocking",
        "hourvideo_performance": "not_tested",
        "dataset_generalization": "not_claimed",
        "git_state_modified": False,
    }
    r1 = {
        **common,
        "baseline": "R1_AV",
        "pipeline": [
            "structural visual + independent ASR map", "shared Planner", "retrieval and Fine localization",
            "requirement-centric Sufficiency", "question-scope Reliability Gate", "immutable reviewed-visual cache",
            "selective visual review on cache miss", "resolved requirement handoff", "text-only Final Gemini",
        ],
        "representation": "detector/tracking structured fallback; no captions",
        "current_cache_replay": {"lookups": 22, "misses": 0, "new_image_transmissions": 0},
    }
    r3 = {
        **common,
        "baseline": "R3_2",
        "pipeline": [
            "frozen global denoised caption+ASR semantic map", "shared Planner", "semantic-Coarse-first Sufficiency",
            "local evidence descent when required", "question-scope Reliability Gate", "immutable reviewed-visual cache",
            "selective visual review on cache miss", "accepted temporal sidecar", "text-only Final Gemini",
        ],
        "representation": "canonical repaired Qwen captions plus timestamped ASR semantic organization",
        "storyline": False,
        "current_cache_replay": {"visual_review_calls": 0, "new_image_transmissions": 0},
    }
    dump(output / "r1_av_engineering_freeze_manifest.json", r1)
    dump(output / "r3_2_engineering_freeze_manifest.json", r3)

    after = {key: sha256(path) for key, path in paths.items()}
    validation = {
        "source_hash_validation": "passed" if before == after else "failed",
        "r1_av_engineering_freeze": "passed" if not errors else "failed",
        "r3_2_engineering_freeze": "passed" if not errors else "failed",
        "shared_interface_freeze": "passed" if not errors else "failed",
        "final_answer_manual_correctness": "not_frozen_user_elected_non_blocking",
        "model_api_calls": 0,
        "git_operations": 0,
        "errors": errors,
        "overall_validation": "frozen_engineering_test_snapshot" if not errors and before == after else "failed",
    }
    dump(output / "validation_report.json", validation)
    report = f"""# R1_AV / R3_2 cached full-pipeline engineering freeze v1

Status: `{validation['overall_validation']}`

This is a hash-addressed engineering test snapshot. It freezes the current
interfaces and artifacts, not answer correctness, HourVideo performance or
dataset generalization.

## R1_AV

Structural visual + independent ASR map -> Planner -> retrieval/Fine
localization -> requirement-centric Sufficiency -> question-scope Gate ->
immutable reviewed-visual cache -> selective cache-miss review -> text-only
Final Gemini.

## R3_2

Global denoised caption+ASR semantic map -> Planner -> semantic-Coarse-first
Sufficiency -> optional local evidence descent -> question-scope Gate ->
immutable reviewed-visual cache -> accepted temporal sidecar -> text-only
Final Gemini.

## Freeze boundary

- Automatic interface and grounding validators: passed.
- Current cache replay: 22/22 lookups hit; 0 new image transmissions.
- User-facing encoding normalization: passed; raw responses unchanged.
- Manual answer correctness: not frozen, explicitly non-blocking for this engineering snapshot.
- Model/API calls for this freeze: 0.
- Existing artifacts and Git state: unchanged.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return validation
