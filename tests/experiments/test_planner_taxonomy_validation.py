from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.planner_taxonomy.analysis import (
    cross_axis_report,
    distribution_report,
    routing_actionability_report,
    stability_report,
)
from src.experiments.planner_taxonomy.core import (
    freeze_samples,
    load_question_only_sources,
    select_stability_subset,
    validate_annotation,
)
from src.experiments.planner_taxonomy.provider import (
    SYSTEM_RUBRIC_A,
    SYSTEM_RUBRIC_B,
    annotation_json_schema,
    question_only_payload,
)


def _source_rows(dataset: str, count: int = 180) -> list[dict[str, str]]:
    return [
        {
            "record_id": f"{dataset}:q{i:03d}",
            "question_id": f"q{i:03d}",
            "question": (
                f"What happens before the sound in event {i}?"
                if i % 3 == 0
                else f"Which object is visible in scene {i}?"
            ),
            "dataset": dataset,
            "text_sha256": f"hash{i:03d}",
        }
        for i in range(count)
    ]


def _annotation(row: dict, *, scope: str = "local", nature: str = "static", modality: str = "visual") -> dict:
    return {
        **row,
        "scope": scope,
        "nature_primary": nature,
        "nature_secondary": None,
        "nature_ambiguity": False,
        "modality": modality,
        "confidence": {"scope": 0.9, "nature": 0.9, "modality": 0.9},
        "reason_short": {"scope": "One neighborhood.", "nature": "State recognition.", "modality": "Visible object."},
        "taxonomy_failure": False,
        "taxonomy_failure_reason": None,
        "sample_type": row.get("sample_type", "representative"),
    }


def test_question_source_loader_discards_gold_options_timestamps_and_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = {
        "egoschema.json": json.dumps([{
        "q_uid": "a", "question": "What is visible?", "answer": 3,
        "option0": "secret option", "google_drive_id": "video-secret",
    }]),
        "egosound.json": json.dumps([{
        "question_id": "b", "question": "What sound occurs?", "answer": "secret",
        "timestamp": [10, 20], "video_path": "secret.mp4",
    }]),
    }
    monkeypatch.setattr(Path, "read_text", lambda self, encoding=None: payloads[self.name])
    result = load_question_only_sources({"egoschema": "egoschema.json", "egosound": "egosound.json"})
    for row in [*result["egoschema"], *result["egosound"]]:
        assert set(row) == {"record_id", "question_id", "question", "dataset", "text_sha256"}
        serialized = json.dumps(row)
        assert "secret" not in serialized


def test_samples_are_deterministic_disjoint_and_exact_size() -> None:
    sources = {"egoschema": _source_rows("egoschema"), "egosound": _source_rows("egosound")}
    first = freeze_samples(sources, seed=17, representative_per_dataset=100, stress_max_per_dataset=50)
    second = freeze_samples(sources, seed=17, representative_per_dataset=100, stress_max_per_dataset=50)
    assert first == second
    representative, stress = first
    assert len(representative) == 200
    assert len(stress) == 100
    assert not {row["record_id"] for row in representative} & {row["record_id"] for row in stress}
    assert all(row["cue_buckets"] for row in stress)


def test_provider_payload_is_question_text_only_with_opaque_ids() -> None:
    row = _source_rows("egoschema", 1)[0]
    payload, mapping = question_only_payload([row])
    parsed = json.loads(payload)
    assert parsed == {"items": [{"item_id": "item_000", "question": row["question"]}]}
    assert row["record_id"] not in payload
    assert mapping == {"item_000": row["record_id"]}


def test_annotation_schema_accepts_valid_mixed_and_rejects_inconsistent_secondary() -> None:
    valid = {
        "item_id": "item_000", "scope": "multi_event", "nature_primary": "dynamic",
        "nature_secondary": "static", "nature_ambiguity": True, "modality": "audio_visual",
        "confidence": {"scope": 0.8, "nature": 0.7, "modality": 0.9},
        "reason_short": {"scope": "Two events must be related.", "nature": "Change with state context.", "modality": "Sound linked to visible actor."},
        "taxonomy_failure": False, "taxonomy_failure_reason": None,
    }
    assert validate_annotation(valid, "item_000") == valid
    invalid = {**valid, "nature_ambiguity": False}
    with pytest.raises(ValueError, match="requires nature_ambiguity"):
        validate_annotation(invalid)


def test_json_schema_forbids_additional_fields_and_bounds_batch() -> None:
    schema = annotation_json_schema(20)
    assert schema["additionalProperties"] is False
    assert "maxItems" not in schema["properties"]["annotations"]
    assert "minItems" not in schema["properties"]["annotations"]
    assert schema["properties"]["annotations"]["items"]["additionalProperties"] is False


def test_stability_selection_exact_per_dataset_and_deterministic() -> None:
    rows = []
    for dataset in ("egoschema", "egosound"):
        for index, source in enumerate(_source_rows(dataset, 30)):
            annotated = _annotation(source)
            annotated["confidence"] = {"scope": 0.5 + index / 100, "nature": 0.8, "modality": 0.9}
            annotated["nature_ambiguity"] = index % 7 == 0
            rows.append(annotated)
    first = select_stability_subset(rows, per_dataset=20, seed=4)
    second = select_stability_subset(rows, per_dataset=20, seed=4)
    assert first == second
    assert len(first) == 40
    assert sum(row["dataset"] == "egoschema" for row in first) == 20
    assert {row["stability_stratum"] for row in first} == {
        "low_confidence_or_ambiguous", "high_confidence", "label_coverage"
    }


def test_stability_metrics_report_axis_and_tuple_disagreement() -> None:
    source = _source_rows("egoschema", 1)[0]
    first = _annotation(source)
    second = {**_annotation(source), "scope": "multi_event"}
    rows, metrics = stability_report([first], [second])
    assert metrics["agreement"]["scope"] == 0
    assert metrics["agreement"]["nature_primary"] == 1
    assert metrics["agreement"]["full_3_axis_tuple"] == 0
    assert rows[0]["any_disagreement"] is True


def test_distribution_separates_representative_from_stress() -> None:
    representative = _annotation({**_source_rows("egoschema", 1)[0], "sample_type": "representative"})
    stress = _annotation({**_source_rows("egosound", 2)[1], "sample_type": "diversity_stress"}, scope="global")
    report = distribution_report([representative, stress])
    assert report["groups"]["combined_representative"]["count"] == 1
    assert report["groups"]["combined_diversity_stress"]["count"] == 1
    assert report["stress_is_natural_prevalence_sample"] is False


def test_cross_axis_and_actionability_are_analysis_only() -> None:
    rows = [
        _annotation({**_source_rows("egoschema", 2)[0], "sample_type": "representative"}),
        _annotation({**_source_rows("egoschema", 2)[1], "sample_type": "representative"}, scope="global", nature="dynamic"),
    ]
    cross = cross_axis_report(rows)
    routing = routing_actionability_report(rows)
    assert cross["special_questions"]["global_dynamic"]["count"] == 1
    assert routing["analysis_only"] is True
    assert routing["must_not_be_used_as_runtime_routing"] is True


def test_rubrics_forbid_answers_media_and_dataset_inference() -> None:
    for rubric in (SYSTEM_RUBRIC_A, SYSTEM_RUBRIC_B):
        lowered = rubric.lower()
        assert "do not answer" in lowered
        assert "question text only" in lowered or "only the evidence demand" in lowered
        assert "dataset identity" in lowered
        assert "video" in lowered
        assert "audio" in lowered
