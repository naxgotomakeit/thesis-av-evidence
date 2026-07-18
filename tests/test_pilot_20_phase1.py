from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.evaluation.pilot_reporting import display_metric, pilot_report_schema, selection_html
from src.evaluation.pilot_selection import expected_modalities, select_case_ids, temporal_metadata
from src.evaluation.pilot_store import CaseCheckpointStore
from src.evaluation.registry import MetricDefinition, MetricRegistry, baseline_metric_registry
from src.instrumentation.timing import StageTimer


def test_metric_registry_is_extensible():
    registry = MetricRegistry()
    registry.register(MetricDefinition("fixture", "efficiency", "fixture", lambda case, context: 3))
    assert registry.evaluate({}, {})[0].value == 3


def test_unavailable_semantic_metrics_are_none_not_zero():
    results = {item.name: item for item in baseline_metric_registry().evaluate({"gold_reference_answer": "a", "validated_prediction": "a"}, {"reference_corpus": ["a"]})}
    for name in ("semantic_similarity", "llm_judge", "task_specific_metric"):
        assert results[name].available is False and results[name].value is None
        assert display_metric(results[name].to_dict()) == "unavailable"


def test_lexical_metrics_are_diagnostic_only():
    results = baseline_metric_registry().evaluate({"gold_reference_answer": "a", "validated_prediction": "a"}, {"reference_corpus": ["a"]})
    assert all(item.category == "diagnostic_reference_overlap" for item in results[:4])
    assert "not treated as authoritative" in pilot_report_schema()["lexical_metric_warning"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [("What happened at 00:03?", "strong"), ("What happened at the start?", "moderate"), ("What happened when it moved?", "weak"), ("Describe the sound.", "none")],
)
def test_temporal_hint_metadata(question, expected):
    row = {"question": question, "provided_timestamp_start": 1.0, "provided_timestamp_end": 3.0}
    metadata = temporal_metadata(row)
    assert metadata["temporal_hint_strength"] == expected
    assert metadata["provided_timestamp_width_sec"] == 2.0
    assert metadata["provided_timestamp_directly_used_by_canonical_runtime"] is False


def test_skipped_stage_duration_remains_null():
    row = StageTimer().skip("fallback_execution", "fallback_not_triggered").to_dict()
    assert row["skipped"] is True and row["duration_sec"] is None


def test_modality_strata_handle_speech_anchor_and_speech_exclusion_generically():
    cross = {"question_type": "Cross-Modal Reasoning", "question": "When the person mentioned a card, what was visible?"}
    count = {"question_type": "Counting", "question": "How many impact sounds occurred, excluding speech?"}
    assert expected_modalities(cross) == ["speech", "visual"]
    assert expected_modalities(count) == ["acoustic"]


def test_resume_store_is_atomic_and_does_not_overwrite():
    path = Path(__file__).resolve().parents[1] / "outputs/pilot_20/baseline_v1/_test_checkpoint_store"
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "case_1.json"
    checkpoint.unlink(missing_ok=True)
    store = CaseCheckpointStore(path)
    try:
        store.save("case_1", {"status": "complete"})
        assert store.load("case_1") == {"status": "complete"}
        assert store.completed_case_ids() == ["case_1"]
        with pytest.raises(FileExistsError):
            store.save("case_1", {"status": "changed"})
    finally:
        checkpoint.unlink(missing_ok=True)
        path.rmdir()


def test_selection_logic_cannot_access_gold_or_predictions():
    source = inspect.getsource(select_case_ids).casefold() + inspect.getsource(temporal_metadata).casefold()
    forbidden_accesses = ('get("answer")', '["answer"]', "get('answer')", "['answer']", 'get("gold")', 'get("prediction")')
    assert not any(token in source for token in forbidden_accesses)


def test_report_schema_enforces_posthoc_gold_order_and_missing_metric_rendering():
    schema = pilot_report_schema()
    assert schema["gold_loading_order"].index("load_gold_posthoc") > schema["gold_loading_order"].index("save_validated_prediction")
    assert display_metric({"available": False, "value": 0}) == "unavailable"


def test_selection_html_is_human_readable_and_handles_empty_tags():
    record = {"case_id": "x", "video_id": "v", "question_type": "Counting", "temporal_hint_strength": "weak", "provided_timestamp": None, "expected_modalities": ["speech"], "selection_tags": []}
    summary = {"case_count": 1, "existing_case_count": 0, "new_case_count": 1, "indexed_video_ids": ["v"]}
    page = selection_html([record], summary)
    assert "<table>" in page and "Live API/model calls: 0" in page


def test_phase1_paths_are_isolated_from_frozen_outputs():
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/pilot_20/baseline_v1"
    assert output != root / "outputs/canonical_pipeline/v0_1"
    assert output != root / "outputs/canonical_pipeline/live_v0_1"
