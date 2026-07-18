from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.canonical_pipeline.media_materialization import materialize_acoustic_candidate
from src.canonical_pipeline.query_scoring import FreshQueryScorer
from src.canonical_pipeline.reranking import validate_visual_frame_contract
from src.canonical_pipeline.runner import CanonicalOnlineRunner
from src.canonical_pipeline.state import ExecutionMode
from src.canonical_pipeline.versions import load_canonical_config


ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_canonical_config(ROOT)
PILOT = ROOT / "data/manifests/egosound_pilot_20.json"


def _data_root() -> Path:
    metadata = json.loads((ROOT / "outputs/audio_index/00002/audio_metadata.json").read_text(encoding="utf-8"))
    return Path(metadata["audio_path"]).parents[3]


def test_manifest_driven_loading_accepts_original_six_and_pilot_twenty():
    original = CanonicalOnlineRunner(CONFIG)
    for row in json.loads(CONFIG.path("case_manifest").read_text(encoding="utf-8")):
        assert original.validate_case(row["case_id"])["video_id"] == row["video_id"]
    pilot = CanonicalOnlineRunner(CONFIG, manifest_path=PILOT, data_root=_data_root())
    rows = json.loads(PILOT.read_text(encoding="utf-8"))
    assert len(rows) == 20
    assert all(pilot.validate_case(row["case_id"])["question"] for row in rows)


def test_unknown_or_other_manifest_case_is_rejected():
    runner = CanonicalOnlineRunner(CONFIG, manifest_path=PILOT, data_root=_data_root())
    with pytest.raises(KeyError):
        runner.validate_case("not_in_this_manifest")


def test_00004_task6_visual_contract_matches_frozen_task6_and_task7a():
    state = CanonicalOnlineRunner(CONFIG).run_case("00004_1", mode=ExecutionMode.REGRESSION_REPLAY)
    frozen_packet = next(row for row in map(json.loads, CONFIG.path("task6_frozen").read_text(encoding="utf-8").splitlines()) if row["case_id"] == "00004_1")
    frozen_payload = next(row for row in map(json.loads, CONFIG.path("task7a_frozen").read_text(encoding="utf-8").splitlines()) if row["case_id"] == "00004_1")
    frames = state.evidence_packet["selected_visual_frames"]
    assert frames == frozen_packet["selected_visual_frames"]
    # The evidence-facing payload remains frozen-equivalent.  The embedded
    # schema and contract metadata are deliberately technical contract fixes:
    # Task7A now embeds the exact Task7B v3 Pydantic schema instead of the old
    # illustrative shape.
    for key in (
        "case_id",
        "question",
        "operation",
        "answer_required_modalities",
        "supporting_modalities",
        "dataset_or_query_inconsistency_status",
        "evidence_groups",
        "pipeline_uncertainties",
        "final_answer_policy",
    ):
        assert state.final_payload[key] == frozen_payload[key]
    from src.final_qa.canonical_schema import validate_required_output_schema

    validate_required_output_schema(state.final_payload["required_output_schema"])
    assert state.final_payload["contract"]["output_schema_version"]
    assert [frame["timestamp"] for frame in frames] == sorted(frame["timestamp"] for frame in frames)
    assert any(frame["selection_rank"] != frame["presentation_order"] for frame in frames)
    assert state.evidence_packet["relations"] == frozen_packet["relations"]
    assert state.evidence_packet["budget_accounting"] == frozen_packet["budget_accounting"]


def test_task6_visual_contract_fails_early_without_selection_rank():
    with pytest.raises(ValueError, match="selection_rank"):
        validate_visual_frame_contract({"selected_visual_frames": [{"timestamp": 0.0, "canonical_frame_path": "x.jpg", "presentation_order": 1, "anchor_distance_sec": None}]})


def test_query_scorer_reuses_offline_embeddings_and_returns_finite_scores(monkeypatch):
    scorer = FreshQueryScorer(ROOT, device="cpu")
    dimensions = {"visual": 512, "speech": 768, "acoustic": 512}
    monkeypatch.setattr(scorer, "_encode", lambda modality, question, video_id: (np.ones(dimensions[modality], dtype=np.float32), 0.0, 0.001, modality))
    case = {"case_id": "new_question", "video_id": "00002", "question": "A new supported question"}
    results = scorer.score_case(case, ["visual", "speech", "acoustic"])
    assert set(results) == {"visual", "speech", "acoustic"}
    assert all(result.ranked_results and all(np.isfinite(item["similarity_score"]) for item in result.ranked_results) for result in results.values())
    assert all(result.historical_score_dependency is False for result in results.values())


def test_task5b_integration_source_does_not_require_historical_score_file():
    source = (ROOT / "src/canonical_pipeline/retrieval.py").read_text(encoding="utf-8")
    assert "allow_historical_score_files=query_scorer is None" in source
    assert "fresh_question_scores=bool(score_results)" in source
    assert 'acoustic_query=acoustic_result.query_vector if acoustic_result else None' in source


def test_query_scoring_timing_is_explicit_online_work():
    source = (ROOT / "src/canonical_pipeline/runner.py").read_text(encoding="utf-8")
    for name in ("visual_query_encode", "visual_similarity_search", "speech_query_encode", "speech_similarity_search", "acoustic_query_encode", "acoustic_similarity_search"):
        assert name.split("_", 1)[1] in source


def test_local_wav_materialization_when_absent_and_provenance():
    tmp_path = ROOT / "outputs/canonical_pipeline/generalization_v0_2/test_materialization"
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source.wav"
    sf.write(source, np.linspace(-0.2, 0.2, 32000, dtype=np.float32), 16000)
    candidate = {"candidate_id": "acoustic_test", "start_time": 0.25, "end_time": 1.25}
    reference, metadata, warning, _ = materialize_acoustic_candidate(candidate, case_id="case", source_audio=source, output_root=tmp_path / "out", role="direct_evidence", project_root=tmp_path)
    assert warning is None and reference and metadata
    assert (tmp_path / reference).is_file()
    assert metadata["requested_selected_interval"] == {"start_sec": 0.25, "end_sec": 1.25}
    assert metadata["materialized_now"] is (not metadata["reused_existing"])
    assert metadata["duration_sec"] == pytest.approx(1.0, abs=1 / 16000)


def test_no_gold_fields_cross_query_scoring_boundary():
    from src.retrieval.three_channel import safe_retrieval_case
    safe = safe_retrieval_case({"case_id": "x", "video_id": "00002", "question": "q", "answer": "forbidden"})
    assert safe == {"case_id": "x", "video_id": "00002", "question": "q"}


def test_frozen_artifacts_are_read_only_during_manifest_validation():
    paths = [CONFIG.path(name) for name in ("planner_plans", "task5b_frozen", "task5c_frozen", "task6_frozen", "task7a_frozen")]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    runner = CanonicalOnlineRunner(CONFIG, manifest_path=PILOT, data_root=_data_root())
    for row in json.loads(PILOT.read_text(encoding="utf-8")):
        runner.validate_case(row["case_id"])
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
