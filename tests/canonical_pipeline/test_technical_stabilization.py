"""Purely technical contract regressions for the frozen Baseline v1 method."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.canonical.build_human_review_site import EXPECTED_CASE_IDS, build_site
from src.canonical_pipeline.contracts import (
    EvidenceContractError,
    payload_evidence_ids,
    stabilize_relations,
    validate_packet_contract,
    validate_payload_contract,
)
from src.canonical_pipeline.fingerprint import fingerprints_compatible
from src.canonical_pipeline.query_scoring import (
    IndexContractError,
    validate_index_bundle_metadata,
)
from src.canonical_pipeline.regression import compare_state
from src.canonical_pipeline.reranking import _groups
from src.canonical_pipeline.runner import CanonicalOnlineRunner
from src.canonical_pipeline.state import ExecutionMode
from src.canonical_pipeline.versions import load_canonical_config
from src.final_qa.canonical_schema import (
    canonical_output_schema,
    validate_required_output_schema,
)
from src.final_qa.task7a_preflight import REQUIRED_OUTPUT_SCHEMA
from src.final_qa.task7b_v02 import FinalQAModelOutputV02
from src.instrumentation.timing import timing_consistency


ROOT = Path(__file__).parents[2]
CONFIG = load_canonical_config(ROOT)
PILOT_RESULTS = ROOT / "outputs/pilot_20/baseline_v1/full_pilot_v0_1/case_results.jsonl"
FROZEN_HASHES = {
    "outputs/question_planner/v2/task5a_plans.jsonl": "ddbc43529f409b2ea17db34429c8360f84e8f5bcae9a02b71701e3a94d99f483",
    "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl": "3a0baf544974c2211014b0c500e1327fece531adff3b700a1c0a929a1fd13709",
    "outputs/evidence_sufficiency/task5c_v1_2/task5c_results.jsonl": "aa47d83e32c5d5016a14b8aac507e0104734fe8d696399cb98edb515db7d7e68",
    "outputs/relation_reranking/task6_v1_2/task6_evidence_packets.jsonl": "e05f0e70c1284582afec567a2cd5194c1760fef974facac995353d3d95f0bef5",
    "outputs/final_qa_preflight/task7a_v1/task7a_payloads.jsonl": "f89d45ed147dfe727c337c24979946f257b3ab2430df4059347176fa736823cf",
}


def _jsonl(path: Path) -> dict[str, dict]:
    return {
        row["case_id"]: row
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _pilot() -> dict[str, dict]:
    return _jsonl(PILOT_RESULTS)


def _frozen() -> dict[str, dict[str, dict]]:
    return {
        "planner": _jsonl(CONFIG.path("planner_plans")),
        "retrieval": _jsonl(CONFIG.path("task5b_frozen")),
        "sufficiency": _jsonl(CONFIG.path("task5c_frozen")),
        "packet": _jsonl(CONFIG.path("task6_frozen")),
        "payload": _jsonl(CONFIG.path("task7a_frozen")),
    }


def test_known_orphan_cases_receive_complete_group_membership_without_selection_change() -> None:
    rows = _pilot()
    for case_id in ("00002_2", "00002_7"):
        source = rows[case_id]
        retained = source["reranking"]["retained"]
        record = {
            "case_id": case_id,
            "task5a_plan_summary": source["planner"]["structured_output"],
            "ambiguity_flags": source["sufficiency_fallback"].get("ambiguity_flags", []),
        }
        groups = _groups(record, retained, source["reranking"].get("relations", []))
        grouped = [item["candidate_id"] for group in groups for item in group["retained_candidates"]]
        assert sorted(grouped) == sorted(item["candidate_id"] for item in retained)
        assert len(grouped) == len(set(grouped))
        assert sorted(item["candidate_id"] for item in retained) == sorted(
            item["candidate_id"] for item in source["reranking"]["retained"]
        )


def test_known_dangling_and_duplicate_relations_are_only_technically_removed() -> None:
    rows = _pilot()
    fallback = rows["00061_5"]["reranking"]
    stable, audit = stabilize_relations(
        fallback["relations"],
        (item["candidate_id"] for item in fallback["retained"]),
    )
    assert stable == []
    assert len(audit["removed_invalid_relations"]) == 1
    assert audit["removed_invalid_relations"][0]["missing_endpoint_ids"] == [
        "speech_branch_missing_evidence"
    ]

    duplicate = rows["00018_7"]["reranking"]
    stable, audit = stabilize_relations(
        duplicate["relations"],
        (item["candidate_id"] for item in duplicate["retained"]),
    )
    assert len(stable) == 2
    assert len(audit["removed_duplicate_relations"]) == 2
    assert audit["endpoint_closure"] and audit["relations_unique"]


def test_packet_contract_counts_entities_assets_and_relations_separately() -> None:
    visual = {
        "candidate_id": "visual_packet",
        "canonical_visual_frames": [{"timestamp": 1.0}, {"timestamp": 2.0}],
    }
    acoustic = {"candidate_id": "acoustic", "local_audio_clip_reference": "clip.wav"}
    relation = {
        "source_candidate_id": "acoustic",
        "relation_type": "near",
        "target_candidate_id": "visual_packet",
        "temporal_gap_sec": 1.0,
        "relation_confidence": "temporal_only",
        "relation_basis": "anchor_distance",
    }
    packet = {
        "candidates_before_reranking": [visual, acoustic],
        "retained_candidates": [visual, acoustic],
        "intentionally_non_model_facing_candidates": [],
        "retained_evidence_groups": [
            {
                "group_id": "group",
                "retained_candidates": [visual, acoustic],
                "relations": [relation],
            }
        ],
        "relations": [relation],
    }
    audit = validate_packet_contract(packet)
    assert audit["canonical_evidence_entity_count"] == 2
    assert audit["visual_frame_asset_count"] == 2
    assert audit["audio_clip_asset_count"] == 1
    assert audit["relation_count"] == 1


def test_packet_and_payload_contracts_reject_identity_collisions() -> None:
    item = {"candidate_id": "same"}
    packet = {
        "retained_candidates": [item, item],
        "retained_evidence_groups": [],
        "relations": [],
    }
    with pytest.raises(EvidenceContractError, match="not unique"):
        validate_packet_contract(packet)

    evidence = {"evidence_id": "same"}
    payload = {
        "evidence_groups": [
            {
                "group_id": "group",
                "visual_evidence": [evidence],
                "speech_evidence": [evidence],
                "acoustic_evidence": [],
                "relations": [],
            }
        ]
    }
    with pytest.raises(EvidenceContractError, match="globally unique"):
        validate_payload_contract(payload)


def test_task7a_embeds_exact_task7b_v3_schema() -> None:
    assert REQUIRED_OUTPUT_SCHEMA == canonical_output_schema()
    assert REQUIRED_OUTPUT_SCHEMA == FinalQAModelOutputV02.model_json_schema()
    validate_required_output_schema(REQUIRED_OUTPUT_SCHEMA)
    altered = json.loads(json.dumps(REQUIRED_OUTPUT_SCHEMA))
    altered["title"] = "drift"
    with pytest.raises(ValueError, match="differs"):
        validate_required_output_schema(altered)


def test_index_contract_validates_every_video_without_model_loading() -> None:
    config = {"name": "sentence-transformers/sentence-t5-base"}
    rows = [{"transcript_segment_id": "s1", "start_time": 0.0, "end_time": 1.0}]
    metadata = {
        "video_id": "video-a",
        "model": config["name"],
        "embedding_normalized": True,
        "validation": {"dimension": 3},
        "rows": rows,
    }
    embeddings = np.ones((1, 3), dtype=np.float32)
    first = validate_index_bundle_metadata(
        modality="speech",
        video_id="video-a",
        metadata=metadata,
        rows=rows,
        embeddings=embeddings,
        encoder_config=config,
    )
    second_metadata = {**metadata, "video_id": "video-b"}
    second = validate_index_bundle_metadata(
        modality="speech",
        video_id="video-b",
        metadata=second_metadata,
        rows=rows,
        embeddings=embeddings,
        encoder_config=config,
        prior_signature=first,
    )
    assert second == first
    with pytest.raises(IndexContractError, match="video identity mismatch"):
        validate_index_bundle_metadata(
            modality="speech",
            video_id="wrong",
            metadata=metadata,
            rows=rows,
            embeddings=embeddings,
            encoder_config=config,
        )
    with pytest.raises(IndexContractError, match="declared embedding dimension"):
        validate_index_bundle_metadata(
            modality="speech",
            video_id="video-a",
            metadata=metadata,
            rows=rows,
            embeddings=np.ones((1, 4), dtype=np.float32),
            encoder_config=config,
        )


def test_timing_duplicate_names_are_not_silently_overwritten() -> None:
    records = [
        {"stage_name": "online_end_to_end_total", "executed": True, "duration_sec": 3.0},
        {"stage_name": "retrieval", "executed": True, "duration_sec": 1.0},
        {"stage_name": "retrieval", "executed": True, "duration_sec": 0.5},
        {"stage_name": "query_encode", "executed": True, "duration_sec": 0.2, "parent_stage": "retrieval"},
    ]
    result = timing_consistency(records, ["retrieval"])
    assert result["non_overlapping_top_level_duration_sum_sec"] == 1.5
    assert result["duplicate_stage_name_counts"]["retrieval"] == 2
    assert result["nested_stage_durations_are_not_additive"] is True


def test_fingerprint_compatibility_requires_full_deterministic_identity() -> None:
    left = {"fingerprint_sha256": "abc", "api_key": "must-not-be-used"}
    right = {"fingerprint_sha256": "abc"}
    changed = {"fingerprint_sha256": "def"}
    assert fingerprints_compatible(left, right)
    assert not fingerprints_compatible(left, changed)
    source = (ROOT / "src/canonical_pipeline/fingerprint.py").read_text(encoding="utf-8")
    assert "API_KEY" not in source


def test_original_six_keep_research_selection_budget_and_zero_calls() -> None:
    frozen = _frozen()
    runner = CanonicalOnlineRunner(CONFIG)
    for case_id in ("00002_7", "00004_1", "00018_1", "00003_2", "00006_3", "00061_5"):
        state = runner.run_case(case_id, mode=ExecutionMode.REGRESSION_REPLAY)
        comparison = compare_state(
            state, {name: records[case_id] for name, records in frozen.items()}
        )
        assert comparison["differences"] == []
        assert comparison["live_safe_to_proceed"] is True
        assert [item["candidate_id"] for item in state.evidence_packet["retained_candidates"]] == [
            item["candidate_id"] for item in frozen["packet"][case_id]["retained_candidates"]
        ]
        assert state.evidence_packet["budget_accounting"] == frozen["packet"][case_id]["budget_accounting"]
        assert sorted(payload_evidence_ids(state.final_payload)) == sorted(
            item["candidate_id"] for item in state.evidence_packet["retained_candidates"]
        )
        assert sum(state.external_calls.values()) == 0


def test_known_00018_2_candidate_accounting_remains_consistent() -> None:
    case = _pilot()["00018_2"]
    reranking = case["reranking"]
    retained = {item["candidate_id"] for item in reranking["retained"]}
    dropped = {item["candidate_id"] for item in reranking["actually_dropped"]}
    assert retained.isdisjoint(dropped)
    assert len(retained) == reranking["budget_accounting"]["retained_candidate_count"]
    assert reranking["budget_accounting"]["dropped_candidate_count"] == len(dropped)


def test_complete_human_review_renders_exact_approved_case_set() -> None:
    output = ROOT / "outputs/pilot_20/baseline_v1/human_review_v0_1/baseline_v1_20_case_human_review.html"
    audit = build_site(PILOT_RESULTS, output)
    assert audit["complete"] is True
    assert audit["expected_case_count"] == audit["rendered_case_count"] == 20
    assert audit["rendered_case_ids"] == EXPECTED_CASE_IDS
    assert audit["api_calls"] == 0
    document = output.read_text(encoding="utf-8")
    assert "BEFORE TASK6" in document and "AFTER TASK6" in document
    assert "POST-HOC ONLY — NOT AVAILABLE TO RUNTIME" in document
    assert document.count('<article class="case-card') == 20


def test_frozen_historical_artifacts_remain_hash_identical() -> None:
    for relative, expected in FROZEN_HASHES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert actual == expected
