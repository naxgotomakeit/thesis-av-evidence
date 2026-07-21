from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts.data.prepare_egopolice_50videos import load_candidates, select_frozen_50


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "config/data/egopolice_50videos.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_frozen_manifest_has_exact_distinct_50_and_usage_designations() -> None:
    rows = load_manifest()["videos"]
    assert len(rows) == 50
    assert len({row["video_id"] for row in rows}) == 50
    assert len({row["leakage_group"] for row in rows}) == 50
    assert [row["selection_rank"] for row in rows] == list(range(1, 51))
    assert all(row["usage_designation"] == "smoke_debug" for row in rows[:5])
    assert all(row["usage_designation"] == "development_fast_iteration" for row in rows[5:10])
    assert all(row["included_in_main_evaluation_50"] for row in rows)


def test_every_selected_video_has_all_three_mcq_types_and_consistent_counts() -> None:
    rows = load_manifest()["videos"]
    for row in rows:
        assert row["mcq_duration_types"] == ["1s", "10s", "60s"]
        assert row["question_count"] == sum(row["question_count_by_type"].values())
        assert row["question_count"] == len(row["associated_question_ids"])
        assert len(row["associated_question_ids"]) == len(set(row["associated_question_ids"]))
        assert row["relative_video_path"] == f"videos/{row['video_metadata_path']}"


def test_manifest_statistics_are_recomputed_from_rows() -> None:
    manifest = load_manifest()
    rows = manifest["videos"]
    stats = manifest["statistics"]
    assert stats["selected_video_count"] == 50
    assert stats["associated_question_count"] == sum(row["question_count"] for row in rows)
    for duration_type in ("1s", "10s", "60s"):
        assert stats["question_count_by_mcq_duration_type"][duration_type] == sum(
            row["question_count_by_type"][duration_type] for row in rows
        )


def test_selection_logic_does_not_read_answers_or_options() -> None:
    source = inspect.getsource(load_candidates) + inspect.getsource(select_frozen_50)
    assert 'row["answer"]' not in source
    assert 'row["options"]' not in source
    manifest = load_manifest()
    assert manifest["selection_uses_model_performance"] is False
    assert manifest["selection_uses_gold_answers"] is False
    assert manifest["selection_uses_ground_truth_timestamps_for_b0_sampling"] is False


def test_current_media_audit_never_claims_missing_files_are_available() -> None:
    for row in load_manifest()["videos"]:
        if row["availability_status"] == "missing":
            assert row["media_probe"] is None
            assert row["download_or_copy_status"] == "not_downloaded"
            assert row["missing_reason"]
            assert row["required_download_method"]
