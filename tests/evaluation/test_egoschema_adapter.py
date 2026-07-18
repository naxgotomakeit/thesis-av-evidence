from __future__ import annotations

from pathlib import Path

import pytest

from src.evaluation.egoschema_adapter import (
    EgoSchemaAdapterError,
    RetrievalProtocol,
    canonical_manifest_row,
    final_selection_payload,
    load_manifest,
    load_posthoc_label,
    retrieval_question,
    runtime_case,
)


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/manifests/egoschema_comparison_pilot.json"


def test_manifest_has_25_unique_cases_and_videos() -> None:
    manifest = load_manifest(MANIFEST)
    assert "posthoc_evaluation" not in manifest
    assert len(manifest["cases"]) == 25
    assert len({row["case_id"] for row in manifest["cases"]}) == 25
    assert len({row["video_id"] for row in manifest["cases"]}) == 25
    assert manifest["selection_audit"]["answer_label_counts"] == {
        "0": 5, "1": 5, "2": 5, "3": 5, "4": 5
    }


def test_protocol_a_and_b_have_explicit_option_visibility() -> None:
    manifest = load_manifest(MANIFEST)
    case = runtime_case(manifest, manifest["cases"][0]["case_id"])
    question_only = retrieval_question(case, RetrievalProtocol.QUESTION_ONLY)
    option_aware = retrieval_question(case, RetrievalProtocol.STANDARD_MULTIPLE_CHOICE)
    assert question_only == case.question
    assert all(option not in question_only for option in case.options)
    assert all(option in option_aware for option in case.options)


def test_canonical_runtime_row_and_final_payload_never_contain_gold() -> None:
    manifest = load_manifest(MANIFEST)
    case = runtime_case(manifest, manifest["cases"][0]["case_id"])
    row = canonical_manifest_row(case, RetrievalProtocol.QUESTION_ONLY)
    payload = final_selection_payload(
        case,
        RetrievalProtocol.QUESTION_ONLY,
        {"evidence_groups": [], "relations": []},
    )
    forbidden = {"answer", "gold", "gold_label", "correct_option", "label_index"}
    assert forbidden.isdisjoint(row)
    assert forbidden.isdisjoint(payload)
    assert case.options == tuple(payload["options"])


def test_gold_access_is_strictly_posthoc() -> None:
    manifest = load_manifest(MANIFEST)
    case_id = manifest["cases"][0]["case_id"]
    with pytest.raises(EgoSchemaAdapterError):
        load_posthoc_label(
            MANIFEST,
            case_id,
            raw_prediction_saved=True,
            validated_prediction_saved=False,
        )
    label = load_posthoc_label(
        MANIFEST,
        case_id,
        raw_prediction_saved=True,
        validated_prediction_saved=True,
    )
    assert 0 <= label["label_index"] <= 4
