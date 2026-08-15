from __future__ import annotations

from experiments.hourvideo_map_guided_direct_answer_v8.core import (
    MAP_SYSTEM,
    MAP_SCHEMA_QUESTION_FIRST,
    CACHED_UNIFIED_SCHEMA,
    _cost,
    cached_map_text,
    select_review_images,
    stage_system,
    review_stage_instruction,
    validate_map_decision,
    validate_review,
)


QUESTION = {
    "question_id": "q1",
    "question_text": "What happened?",
    "answer_options": [{"option_id": "A", "text": "a"}, {"option_id": "B", "text": "b"}],
}


def test_strict_evidence_prompt_is_opt_in() -> None:
    assert stage_system({}, MAP_SYSTEM) == MAP_SYSTEM
    strict = stage_system({"strict_evidence_only": True}, MAP_SYSTEM)
    assert "Never use commonsense" in strict
    assert "word or action appearing" in strict


def test_no_commonsense_prompt_does_not_require_predicate_evidence() -> None:
    prompt = stage_system({"no_commonsense_only": True}, MAP_SYSTEM)
    assert "never reject an option merely because" in prompt
    assert "Every answer-critical predicate" not in prompt
    assert "request the smallest targeted set" not in prompt


def test_question_first_contract_is_generic() -> None:
    prompt = stage_system({"question_first_decision": True}, MAP_SYSTEM)
    assert "question as an information task" in prompt
    assert "puzzle" not in prompt.lower()
    assert "question_understanding" in MAP_SCHEMA_QUESTION_FIRST["required"]


def test_cached_map_block_is_question_independent() -> None:
    text = cached_map_text(navigation_map([]))
    assert "navigation_map" in text
    assert "map_image_catalog" in text
    assert "question_text" not in text
    assert "response_stage" in CACHED_UNIFIED_SCHEMA["required"]
    assert "question_understanding" in CACHED_UNIFIED_SCHEMA["required"]


def test_cache_cost_uses_write_and_read_rates() -> None:
    cfg = {"provider": "anthropic", "anthropic": {"pricing_usd_per_million": {
        "input": 1.0, "cache_write_5m": 1.25, "cache_read": 0.1, "output": 5.0,
    }}}
    value = _cost(cfg, [{
        "input_tokens": 100, "cache_creation_input_tokens": 1000,
        "cache_read_input_tokens": 2000, "output_tokens": 10, "total_tokens": 110,
    }])
    assert value["estimated_usd"] == (100 + 1250 + 200 + 50) / 1_000_000


def navigation_map(paths: list[str]) -> dict:
    return {
        "coarse_regions": [{
            "coarse_id": "C01", "start_sec": 0, "end_sec": 20,
            "source_medium_ids": ["M001", "M002"],
            "exact_source_captions": [
                {"medium_id": "M001", "start_sec": 0, "end_sec": 10, "source_frame_paths": paths[:3]},
                {"medium_id": "M002", "start_sec": 10, "end_sec": 20, "source_frame_paths": paths[3:]},
            ],
        }]
    }


def decision(status: str) -> dict:
    review = status != "answerable"
    return {
        "question_id": "q1", "map_status": status,
        "provisional_option_id": "A" if not review else "NONE", "reasoning": "because",
        "supporting_coarse_ids": ["C01"], "supporting_medium_ids": [],
        "requested_coarse_ids": ["C01"] if review else [], "requested_medium_ids": [],
        "requested_map_image_ids": ["MAPIMG0002", "MAPIMG0005"] if review else [],
        "review_goal": "resolve detail" if review else "",
    }


def test_answerable_map_decision_forbids_review() -> None:
    value = decision("answerable")
    assert validate_map_decision(value, QUESTION, navigation_map([])) == []
    value["review_goal"] = "No review needed; the map is sufficient."
    assert validate_map_decision(value, QUESTION, navigation_map([])) == []
    value["requested_coarse_ids"] = ["C01"]
    assert "answerable decision improperly requests review" in validate_map_decision(value, QUESTION, navigation_map([]))


def test_unclear_decision_requires_localized_map_ids() -> None:
    value = decision("unclear")
    source = navigation_map([f"frame_{index:05d}.jpg" for index in range(6)])
    assert validate_map_decision(value, QUESTION, source) == []
    value["requested_coarse_ids"] = []
    assert "review decision has no localized map IDs" in validate_map_decision(value, QUESTION, source)


def test_ai_selected_images_are_exact_and_bounded(tmp_path) -> None:
    paths = []
    for index in range(6):
        path = tmp_path / f"frame_{index:05d}.jpg"
        path.write_bytes(b"jpeg")
        paths.append(str(path))
    images = select_review_images(navigation_map(paths), decision("unclear"), maximum=2)
    assert len(images) == 2
    assert {row["medium_id"] for row in images} == {"M001", "M002"}
    assert all(row["exists"] for row in images)


def test_review_rejects_unknown_image_id() -> None:
    result = {
        "question_id": "q1", "selected_option_id": "A", "answer_text": "a", "reasoning": "x",
        "visual_review_changed_answer": False, "supporting_coarse_ids": ["C01"],
        "supporting_medium_ids": ["M001"], "supporting_image_ids": ["BAD"],
        "remaining_uncertainty": [],
    }
    errors = validate_review(result, QUESTION, navigation_map([]), [{"image_id": "IMG01"}])
    assert "unknown final image ID" in errors


def test_review_accepts_stable_map_image_alias() -> None:
    result = {
        "question_id": "q1", "selected_option_id": "A", "answer_text": "a", "reasoning": "x",
        "visual_review_changed_answer": False, "supporting_coarse_ids": ["C01"],
        "supporting_medium_ids": ["M001"], "supporting_image_ids": ["MAPIMG0001"],
        "remaining_uncertainty": [],
    }
    errors = validate_review(
        result, QUESTION, navigation_map([]),
        [{"image_id": "IMG01", "map_image_id": "MAPIMG0001"}],
    )
    assert errors == []


def test_local_review_image_contract_rejects_map_alias() -> None:
    result = {
        "question_id": "q1", "selected_option_id": "A", "answer_text": "a", "reasoning": "x",
        "visual_review_changed_answer": False, "supporting_coarse_ids": ["C01"],
        "supporting_medium_ids": ["M001"], "supporting_image_ids": ["MAPIMG0001"],
        "remaining_uncertainty": [],
    }
    images = [{"image_id": "IMG01", "map_image_id": "MAPIMG0001"}]
    errors = validate_review(
        result, QUESTION, navigation_map([]), images, local_image_ids_only=True,
    )
    assert "unknown final image ID" in errors
    result["supporting_image_ids"] = ["IMG01"]
    assert validate_review(
        result, QUESTION, navigation_map([]), images, local_image_ids_only=True,
    ) == []


def test_local_review_image_prompt_names_single_allowed_namespace() -> None:
    prompt = review_stage_instruction({"local_review_image_ids_only": True}, "review")
    assert "IMGxx is the only identifier" in prompt
    assert "namespace allowed in supporting_image_ids" in prompt
    assert "never copy a MAPIMGxxxx" in prompt
