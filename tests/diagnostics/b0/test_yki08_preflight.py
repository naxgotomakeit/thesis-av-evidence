from __future__ import annotations

import inspect
from pathlib import Path

from src.diagnostics.b0 import yki08_preflight as preflight


ROOT = Path(__file__).resolve().parents[3]


def test_frozen_yki08_question_selection_and_manifest_hash() -> None:
    questions, digest = preflight.load_yki08_questions(
        ROOT / "config/data/egopolice_ablation_questions_v1.json"
    )
    assert digest == preflight.EXPECTED_QUESTION_MANIFEST_SHA256
    assert [row["question_id"] for row in questions] == [
        "1s_3455", "1s_3447", "10s_4776", "10s_3437", "60s_1377"
    ]


def test_preflight_reuses_frozen_b0_configuration() -> None:
    config = preflight.load_frozen_config(ROOT / "config/baselines/egopolice_b0.json")
    assert config["num_frames"] == 8
    assert config["max_pixels"] == 262144
    assert config["dtype"] == "bfloat16"
    assert config["quantization"]["mode"] == "none"
    assert config["generation"]["do_sample"] is False


def test_gt_cannot_enter_the_frame_extractor() -> None:
    parameters = inspect.signature(preflight.extract_uniform_frames).parameters
    assert "question" not in parameters
    assert "gt_interval" not in parameters
    assert "annotation_interval" not in parameters


def test_non_formal_label_is_explicit() -> None:
    assert "PRE-FLIGHT" in preflight.NON_FORMAL_LABEL
    assert "NON-FORMAL" in preflight.NON_FORMAL_LABEL

