from __future__ import annotations

import copy
import hashlib
import json
import shutil
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from src.canonical_pipeline.fingerprint import (
    FingerprintMismatchError,
    relevant_source_fingerprint,
    validate_reusable_result_fingerprint,
)
from src.canonical_pipeline.media_materialization import (
    materialize_acoustic_candidate,
    materialize_model_facing_acoustic_entities,
)
from src.canonical_pipeline.provider_journal import (
    AmbiguousProviderAttemptError,
    ProviderAttemptJournal,
)
from src.canonical_pipeline.retrieval import _attach_dense_frame_lineage
from src.canonical_pipeline.state import CaseState, ExecutionMode
from src.canonical_pipeline import sufficiency as sufficiency_module
from src.retrieval.task5b import canonical_visual_frames


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def workspace_tmp() -> Path:
    """Use a sandbox-owned workspace temp; pytest basetemp ACLs are protected here."""
    path = ROOT / "_codex_runtime_tests" / uuid.uuid4().hex
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _wav(path: Path, value: float = 0.1, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.full(int(16000 * seconds), value, dtype=np.float32), 16000)


def test_stale_same_name_wav_is_rematerialized_from_verified_source(workspace_tmp: Path) -> None:
    tmp_path = workspace_tmp
    source = tmp_path / "source.wav"
    _wav(source, 0.2, 2.0)
    candidate = {
        "candidate_id": "acoustic_1",
        "modality": "acoustic",
        "start_time": 0.0,
        "end_time": 1.0,
    }
    reference, first, warning, _ = materialize_acoustic_candidate(
        candidate,
        case_id="case",
        source_audio=source,
        output_root=tmp_path / "runtime",
        role="direct_evidence",
        project_root=tmp_path,
    )
    assert warning is None and reference and first
    clip = tmp_path / reference
    assert first["materialized_now"] is True
    _wav(clip, -0.7, 1.0)  # readable collision with the same deterministic name

    second_reference, second, warning, _ = materialize_acoustic_candidate(
        candidate,
        case_id="case",
        source_audio=source,
        output_root=tmp_path / "runtime",
        role="direct_evidence",
        project_root=tmp_path,
    )
    assert warning is None and second_reference == reference and second
    assert second["reused_existing"] is False
    assert second["reuse_rejected_reason"] == "wav_content_hash_mismatch"
    sidecar = json.loads((tmp_path / second["provenance_path"]).read_text(encoding="utf-8"))
    assert hashlib.sha256(clip.read_bytes()).hexdigest() == sidecar["materialized_media"]["sha256"]
    _, third, warning, _ = materialize_acoustic_candidate(
        candidate,
        case_id="case",
        source_audio=source,
        output_root=tmp_path / "runtime",
        role="direct_evidence",
        project_root=tmp_path,
    )
    assert warning is None and third and third["reused_existing"] is True


def test_every_and_only_model_facing_acoustic_entity_gets_a_valid_wav(workspace_tmp: Path) -> None:
    tmp_path = workspace_tmp
    source = tmp_path / "source.wav"
    _wav(source, 0.2, 2.0)
    supporting = {
        "candidate_id": "supporting_acoustic",
        "modality": "acoustic",
        "start_time": 0.25,
        "end_time": 1.25,
        "roles": ["supporting"],
        "local_audio_clip_reference": None,
    }
    dropped = {
        "candidate_id": "not_model_facing",
        "modality": "acoustic",
        "start_time": 1.25,
        "end_time": 1.75,
        "roles": ["supporting"],
        "local_audio_clip_reference": None,
    }
    packet = {
        "retained_candidates": [copy.deepcopy(supporting)],
        "retained_evidence_groups": [
            {"group_id": "g", "retained_candidates": [copy.deepcopy(supporting)]}
        ],
        "local_audio_clips": [{"candidate_id": dropped["candidate_id"], "clip_path": "stale.wav"}],
    }
    selected_before = [item["candidate_id"] for item in packet["retained_candidates"]]
    audit = materialize_model_facing_acoustic_entities(
        packet,
        case_id="case",
        source_audio=source,
        output_root=tmp_path / "runtime",
        project_root=tmp_path,
    )
    assert [item["candidate_id"] for item in packet["retained_candidates"]] == selected_before
    assert audit["model_facing_acoustic_ids"] == ["supporting_acoustic"]
    assert [item["candidate_id"] for item in packet["local_audio_clips"]] == ["supporting_acoustic"]
    reference = packet["retained_candidates"][0]["local_audio_clip_reference"]
    assert reference and (tmp_path / reference).is_file()
    assert packet["retained_evidence_groups"][0]["retained_candidates"][0]["local_audio_clip_reference"] == reference


def test_pre_and_post_fallback_diagnostics_are_candidate_scoped(
    monkeypatch: pytest.MonkeyPatch, workspace_tmp: Path
) -> None:
    tmp_path = workspace_tmp
    state = CaseState(
        case_id="case",
        video_id="video",
        question="When is the phrase spoken?",
        video_duration_sec=10.0,
        mode=ExecutionMode.EXECUTE_LIVE,
    )
    state.planner_output = {
        "resolver_modalities": ["speech"],
        "answer_requirement": {"operation": "measure_delay"},
    }
    state.retrieval_result = {
        "selected_candidates": [
            {"candidate_id": "pre", "modality": "speech", "start_time": 0.0, "end_time": 1.0}
        ],
        "anchor_resolution": {},
    }

    def classify(context, plan, cues, candidates):
        del context, plan, cues
        fallback = len(candidates) == 1
        return ({
            "fallback_required": fallback,
            "sufficiency_reason_codes": ["missing"] if fallback else [],
            "critical_missing_evidence": [],
            "ambiguity_flags": [],
        }, {})

    def diagnostics(state, candidates, frozen):
        del state, frozen
        return [
            {"candidate_id": item["candidate_id"], "broad_source_affects_sufficiency": False}
            for item in candidates
        ]

    monkeypatch.setattr(sufficiency_module, "classify_v1_1", classify)
    monkeypatch.setattr(sufficiency_module, "_diagnostics", diagnostics)
    fallback = lambda state, context: {
        "triggered": True,
        "added_candidates": [
            {"candidate_id": "fallback_only", "modality": "speech", "start_time": 1.0, "end_time": 2.0}
        ],
        "model_calls": 0,
        "warnings": [],
    }
    sufficiency_module.run_evidence_sufficiency(
        state,
        fallback_executor=fallback,
        materialization_root=tmp_path,
        project_root=tmp_path,
    )
    result = state.sufficiency_result
    assert result is not None
    assert [item["candidate_id"] for item in result["pre_fallback_acoustic_evidence_diagnostics"]] == ["pre"]
    assert [item["candidate_id"] for item in result["post_fallback_acoustic_evidence_diagnostics"]] == ["pre", "fallback_only"]
    assert result["acoustic_evidence_diagnostics"] == result["post_fallback_acoustic_evidence_diagnostics"]


def test_dense_only_frames_gain_resolvable_lineage_without_selection_change() -> None:
    result = {
        "local_visual_refinement": {
            "dense_frames": [
                {"dense_frame_id": "dense_000", "timestamp": 1.0, "frame_path": "one.jpg"},
                {"dense_frame_id": "dense_001", "timestamp": 1.5, "frame_path": "two.jpg"},
            ],
            "micro_windows": [
                {"candidate_id": "micro_1", "start_time": 0.5, "end_time": 2.0}
            ],
        },
        "coarse_visual_candidates": [
            {"candidate_id": "coarse_1", "start_time": 0.0, "end_time": 3.0}
        ],
    }
    before = [(item["frame_path"], item["timestamp"]) for item in result["local_visual_refinement"]["dense_frames"]]
    dense = _attach_dense_frame_lineage(result)
    frames = canonical_visual_frames([], dense, video_id="video")
    assert [(item["canonical_frame_path"], item["timestamp"]) for item in frames] == before
    assert all(item["source_candidate_ids"] == ["micro_1"] for item in frames)
    assert all(item["provenance_records"][0]["source_coarse_candidate_ids"] == ["coarse_1"] for item in frames)
    assert all(item["provenance_records"][0]["lineage_basis"] == "temporal_overlap_with_refinement_candidate" for item in frames)


def test_provider_response_crash_blocks_automatic_duplicate_call(workspace_tmp: Path) -> None:
    tmp_path = workspace_tmp
    journal = ProviderAttemptJournal(tmp_path / "journal")
    calls = []

    def provider(system: str, user: str, number: int):
        calls.append((system, user, number))
        return SimpleNamespace(text='{"ok":true}', latency_sec=0.1, input_tokens=2, output_tokens=1)

    wrapped = journal.planner_request("case", provider)
    wrapped("system", "user", 1)
    assert journal.records("case")[0]["state"] == "response_saved"
    assert journal.records("case")[0]["history"] == ["pending", "sent", "response_saved"]
    resumed = ProviderAttemptJournal(tmp_path / "journal")
    with pytest.raises(AmbiguousProviderAttemptError, match="Automatic provider retry blocked"):
        resumed.assert_case_resumable("case")
    assert len(calls) == 1
    journal.validated("case", "question_planner", 1, {"schema_valid": True})
    checkpoint = tmp_path / "case.json"
    checkpoint.write_text("{}", encoding="utf-8")
    journal.mark_case_checkpointed("case", checkpoint)
    record = journal.records("case")[0]
    assert record["state"] == "checkpointed"
    assert record["history"] == [
        "pending", "sent", "response_saved", "validated", "checkpointed"
    ]
    journal.assert_case_resumable("case")


def _fingerprint(components: dict[str, object]) -> dict[str, object]:
    digest = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**components, "fingerprint_sha256": digest}


@pytest.mark.parametrize(
    "component",
    ["git_code", "dirty_source", "versions", "configs", "prompts", "schema", "indexes", "models", "manifest"],
)
def test_any_bound_fingerprint_component_mismatch_rejects_reuse(component: str) -> None:
    components = {name: "same" for name in (
        "git_code", "dirty_source", "versions", "configs", "prompts", "schema", "indexes", "models", "manifest"
    )}
    expected = _fingerprint(components)
    changed = dict(components)
    changed[component] = "changed"
    actual = _fingerprint(changed)
    record = {"run_fingerprint": actual, "run_fingerprint_sha256": actual["fingerprint_sha256"]}
    with pytest.raises(FingerprintMismatchError):
        validate_reusable_result_fingerprint(record, expected)


def test_exact_fingerprint_allows_reuse_and_source_hash_tracks_dirty_content(workspace_tmp: Path) -> None:
    tmp_path = workspace_tmp
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts/canonical").mkdir(parents=True)
    (tmp_path / "config").mkdir()
    (tmp_path / "configs").mkdir()
    source = tmp_path / "src/module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    first_source = relevant_source_fingerprint(tmp_path)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    second_source = relevant_source_fingerprint(tmp_path)
    assert first_source != second_source

    expected = _fingerprint({"working_source": second_source})
    record = {"run_fingerprint": copy.deepcopy(expected), "run_fingerprint_sha256": expected["fingerprint_sha256"]}
    validate_reusable_result_fingerprint(record, expected)


def test_completed_checkpoint_reuse_is_never_exempt_from_fingerprint_validation() -> None:
    full_source = (ROOT / "scripts/canonical/run_full_pilot.py").read_text(encoding="utf-8")
    smoke_source = (ROOT / "scripts/canonical/run_new_case_smoke.py").read_text(encoding="utf-8")
    assert "if incomplete:\n        for completed_case_id" not in full_source
    assert "for completed_case_id in sorted(existing_before.intersection(case_ids))" in full_source
    assert "if not all_already_complete:\n        for case_id in SMOKE_CASES" not in smoke_source
    assert "for case_id in SMOKE_CASES:\n        reusable = store.load(case_id)" in smoke_source
