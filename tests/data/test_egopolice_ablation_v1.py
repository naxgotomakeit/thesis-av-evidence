from __future__ import annotations

import inspect
import json
import os
from pathlib import Path, PurePosixPath

import pytest

from scripts.data.freeze_egopolice_ablation_v1 import (
    EXCLUSIONS,
    build_question_manifest,
    build_video_manifest,
    load_mcq_metadata,
    select_ablation20,
    select_question_ids,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
PARENT_PATH = ROOT / "config/data/egopolice_50videos.json"
VIDEO_PATH = ROOT / "config/data/egopolice_ablation20_v1.json"
QUESTION_PATH = ROOT / "config/data/egopolice_ablation_questions_v1.json"
DEFAULT_DATA_ROOT = Path(
    "/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0"
)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", DEFAULT_DATA_ROOT))


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_ablation20_is_exact_unique_parent_subset_without_debug_video():
    parent_ids = {row["video_id"] for row in _load(PARENT_PATH)["videos"]}
    rows = _load(VIDEO_PATH)["videos"]
    selected_ids = [row["video_id"] for row in rows]

    assert len(rows) == 20
    assert len(set(selected_ids)) == 20
    assert set(selected_ids) <= parent_ids
    assert set(selected_ids).isdisjoint(EXCLUSIONS)
    assert [row["selection_rank"] for row in rows] == list(range(1, 21))


def test_video_selection_regenerates_identically_from_parent_manifest():
    parent = _load(PARENT_PATH)
    regenerated, _ = select_ablation20(parent)
    frozen = _load(VIDEO_PATH)

    assert [row["video_id"] for row in regenerated] == [
        row["video_id"] for row in frozen["videos"]
    ]
    assert build_video_manifest(parent, Path("config/data/egopolice_50videos.json")) == frozen
    assert frozen["parent_frozen_50_manifest"]["sha256"] == sha256_file(PARENT_PATH)


def test_selection_functions_do_not_read_result_answer_option_or_interval_fields():
    source = inspect.getsource(select_ablation20) + inspect.getsource(select_question_ids)
    forbidden = (
        "prediction", "correct", "oracle", "hit_at", '["answer"]', '["options"]',
        '["question"]', '["start second"]', '["end second"]', "downloaded",
    )

    assert all(value not in source.lower() for value in forbidden)
    video_rule = _load(VIDEO_PATH)["selection_rule"]
    question_rule = _load(QUESTION_PATH)["selection_rule"]
    assert video_rule["model_results_used"] is False
    assert video_rule["downloaded_status_used"] is False
    assert question_rule["selection_reads_model_results"] is False
    assert question_rule["selection_reads_ground_truth_answer"] is False


@pytest.mark.skipif(
    not all((DATA_ROOT / f"mcq_{duration_class}.json").is_file() for duration_class in ("1s", "10s", "60s")),
    reason="Official EgoPolice MCQ metadata is unavailable",
)
def test_every_frozen_question_matches_official_metadata_and_valid_contract():
    metadata = load_mcq_metadata(DATA_ROOT)
    selected_video_ids = {row["video_id"] for row in _load(VIDEO_PATH)["videos"]}
    questions = _load(QUESTION_PATH)["questions"]

    assert len(questions) == 98
    assert len({row["question_id"] for row in questions}) == 98
    assert {row["video_id"] for row in questions} == selected_video_ids
    for frozen in questions:
        official = metadata[frozen["duration_class"]][frozen["question_id"]]
        official_video_id = str(PurePosixPath(str(official["video"])).with_suffix(""))
        assert frozen["video_id"] == official_video_id
        assert frozen["metadata_source_file"] == f"mcq_{frozen['duration_class']}.json"
        assert frozen["gt_interval_sec"] == [
            float(official["start second"]), float(official["end second"])
        ]
        assert frozen["gt_interval_sec"][1] > frozen["gt_interval_sec"][0]
        assert len(frozen["options"]) == 5
        assert frozen["options"] == [str(value) for value in official["options"]]
        assert frozen["ground_truth_index"] == official["answer"]
        assert frozen["ground_truth_text"] == frozen["options"][official["answer"]]


@pytest.mark.skipif(
    not all((DATA_ROOT / f"mcq_{duration_class}.json").is_file() for duration_class in ("1s", "10s", "60s")),
    reason="Official EgoPolice MCQ metadata is unavailable",
)
def test_question_selection_regenerates_identically_from_documented_rule():
    video_manifest = _load(VIDEO_PATH)
    metadata = load_mcq_metadata(DATA_ROOT)
    regenerated = build_question_manifest(
        video_manifest,
        Path("config/data/egopolice_ablation20_v1.json"),
        metadata,
        DATA_ROOT,
    )

    assert regenerated == _load(QUESTION_PATH)
    assert regenerated["parent_ablation20_manifest"]["sha256"] == sha256_file(VIDEO_PATH)
