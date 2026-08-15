from __future__ import annotations

from experiments.hourvideo_map_guided_iterative_dynamic_frames_v8_7.core import (
    SYSTEM,
    observable_commonsense_violations,
    RELATION_BINDING_ADDENDUM,
    RESPONSE_SCHEMA_RELATION_BINDING,
    resolve_requested_frames,
    validate_response,
)


QUESTION = {
    "question_id": "q1", "question_text": "What happened?",
    "answer_options": [{"option_id": "A", "text": "a"}, {"option_id": "B", "text": "b"}],
}


MAP = {
    "coarse_regions": [{
        "coarse_id": "C01", "start_sec": 0, "end_sec": 20,
        "source_medium_ids": ["M001"],
        "exact_source_captions": [{
            "medium_id": "M001", "start_sec": 0, "end_sec": 20,
            "source_frame_paths": [],
        }],
    }],
}


UNDERSTANDING = {
    "task_type": "event", "target_entities": ["ego"],
    "requested_relation": "event", "temporal_constraints": [],
    "spatial_constraints": [], "participant_constraints": [],
    "other_constraints": [], "critical_option_differences": ["a vs b"],
    "information_needed": ["event identity"],
}


def response(action: str) -> dict:
    return {
        "response_stage": "map_only", "action": action,
        "question_understanding": UNDERSTANDING, "question_id": "q1",
        "selected_option_id": "A" if action == "answer" else "NONE",
        "answer_text": "a" if action == "answer" else "", "reasoning": "x",
        "supporting_coarse_ids": ["C01"], "supporting_medium_ids": ["M001"],
        "supporting_image_ids": [], "observations": [],
        "requested_frames": [] if action == "answer" else [
            {"timestamp_sec": 2, "inspection_goal": "start"},
            {"timestamp_sec": 18, "inspection_goal": "end"},
        ],
        "remaining_uncertainty": [],
    }


def test_prompt_requires_two_at_a_time_without_commonsense() -> None:
    assert "exactly two" in SYSTEM
    assert "Do not use commonsense" in SYSTEM
    assert "scan the whole video" in SYSTEM


def test_stage_prompt_requires_none_while_requesting_more() -> None:
    from experiments.hourvideo_map_guided_iterative_dynamic_frames_v8_7.core import MAP_STAGE, REVIEW_STAGE
    assert "selected_option_id must be NONE" in MAP_STAGE
    assert "selected_option_id must be NONE" in REVIEW_STAGE


def test_request_more_requires_exact_pair() -> None:
    value = response("request_more")
    assert validate_response(value, QUESTION, MAP, 2, [], set()) == []
    value["requested_frames"] = value["requested_frames"][:1]
    assert "request_more must request exactly 2 frames" in validate_response(
        value, QUESTION, MAP, 2, [], set()
    )


def test_request_timestamp_must_be_inside_video_map() -> None:
    value = response("request_more")
    value["requested_frames"][0]["timestamp_sec"] = 21
    assert "requested timestamp outside video map range" in validate_response(
        value, QUESTION, MAP, 2, [], set()
    )


def test_dynamic_resolver_uses_nearest_unseen_frame(tmp_path) -> None:
    inventory = []
    for second in (0, 1, 2, 3):
        path = tmp_path / f"frame_{second:05d}.jpg"
        path.write_bytes(b"jpeg")
        inventory.append((second, path))
    requests = response("request_more")["requested_frames"]
    requests[0]["timestamp_sec"] = 0.6
    requests[1]["timestamp_sec"] = 2.6
    images, errors = resolve_requested_frames(requests, inventory, set(), 1, MAP)
    assert errors == []
    assert [row["resolved_timestamp_sec"] for row in images] == [1, 3]
    assert [row["image_id"] for row in images] == ["DYNIMG001", "DYNIMG002"]
    assert all(row["medium_id"] == "M001" for row in images)


def test_review_citations_must_use_seen_dynamic_ids() -> None:
    value = response("answer")
    value["response_stage"] = "dynamic_visual_review"
    value["observations"] = [
        {"image_id": "DYNIMG001", "observation": "a"},
        {"image_id": "DYNIMG002", "observation": "b"},
    ]
    value["supporting_image_ids"] = ["MAPIMG0001"]
    images = [{"image_id": "DYNIMG001"}, {"image_id": "DYNIMG002"}]
    assert "unknown supporting dynamic image ID" in validate_response(
        value, QUESTION, MAP, 2, images, {"DYNIMG001", "DYNIMG002"}
    )


def test_observable_commonsense_gate_catches_unseen_logical_guess() -> None:
    value = response("answer")
    value["reasoning"] = (
        "The event was not shown in detail, but this aligns with a logical sequence "
        "and would have been the most likely action."
    )
    violations = observable_commonsense_violations(value)
    assert "logical_sequence_prior" in violations
    assert "unobserved_claim_resolved_by_prior" in violations


def test_observable_commonsense_gate_ignores_request_more() -> None:
    value = response("request_more")
    value["reasoning"] = "The most likely action is unclear, so inspect frames."
    assert observable_commonsense_violations(value) == []


def test_relation_binding_prompt_distinguishes_paraphrase_from_prior() -> None:
    normalized = " ".join(RELATION_BINDING_ADDENDUM.split())
    assert "Semantic paraphrase matching" in normalized
    assert "typical" in normalized
    assert "relation_query" in RESPONSE_SCHEMA_RELATION_BINDING["required"]
    assert "option_bindings" in RESPONSE_SCHEMA_RELATION_BINDING["required"]


def test_relation_binding_validator_requires_all_options() -> None:
    value = response("answer")
    value["relation_query"] = {
        "participants": ["ego"], "action_or_state": "event", "object": "",
        "location": "", "temporal_relation": "", "other_qualifiers": [],
        "scope_ambiguities": [],
    }
    value["option_bindings"] = []
    errors = validate_response(
        value, QUESTION, MAP, 2, [], set(), relation_binding_required=True,
    )
    assert "option bindings must cover every option exactly once" in errors


def test_relation_binding_accepts_reviewed_dynamic_source_id() -> None:
    value = response("answer")
    value["relation_query"] = {
        "participants": ["ego"], "action_or_state": "event", "object": "",
        "location": "", "temporal_relation": "", "other_qualifiers": [],
        "scope_ambiguities": [],
    }
    value["option_bindings"] = [
        {
            "option_id": option_id, "normalized_relation": option_id,
            "aligned_time_windows": ["0-1s"] if option_id == "A" else [],
            "evidence_summary": "seen" if option_id == "A" else "not seen",
            "source_ids": ["DYNIMG001"] if option_id == "A" else [],
            "verdict": "supported" if option_id == "A" else "unknown",
            "missing_evidence": [] if option_id == "A" else ["event"],
        }
        for option_id in ("A", "B")
    ]
    assert validate_response(
        value, QUESTION, MAP, 2, [], {"DYNIMG001"},
        relation_binding_required=True,
    ) == []
