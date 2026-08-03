from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.experiments.planner_medium_retrieval.core import (
    DEFAULT_CONFIG,
    PlannerError,
    compute_medium_ranking,
    load_medium_embeddings,
    plan_question,
    route_coarse_nodes,
    sha256_file,
)
from src.experiments.planner_medium_retrieval.validation import validate_planner_output

ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = ROOT / "outputs/experiments/structured_organizer_v1/226/hierarchical_index_v1.json"
TEST_RUNTIME = ROOT / "outputs/experiments/planner_medium_retrieval_v1/unit_test_runtime"


def load_index():
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def question(qid="q_weapon_visible"):
    texts = {
        "q_global_summary": "What happened throughout the incident?",
        "q_weapon_visible": "Was a weapon visibly present?",
        "q_handcuff_before_medical": "Did handcuffing occur before medical assistance?",
    }
    return {"question_id": qid, "question": texts[qid], "answer_options": []}


def valid_plan(qid="q_weapon_visible"):
    if qid == "q_global_summary":
        return {
            "planner_version": "planner_v1", "question_id": qid, "scope": "global",
            "operation": "summary",
            "search_units": [{"unit_id": "u1", "description": "incident progression",
                              "query_variants": ["major events"], "required_modalities": ["visual"],
                              "temporal_relation": None}],
            "required_evidence": [{"slot_id": "s1", "description": "coverage",
                                   "search_unit_ids": ["u1"], "minimum_support": 1}],
            "retrieval_strategy": "global_coverage", "candidate_storyline_ids": [],
            "candidate_coarse_ids": [], "needs_temporal_neighbors": False,
            "uncertainty_notes": [],
        }
    if qid == "q_handcuff_before_medical":
        units = [
            {"unit_id": "handcuff", "description": "handcuffing", "query_variants": ["handcuffs"],
             "required_modalities": ["visual"], "temporal_relation": "before_or_after"},
            {"unit_id": "medical", "description": "medical assistance", "query_variants": ["medical aid"],
             "required_modalities": ["visual"], "temporal_relation": "before_or_after"},
        ]
        return {
            "planner_version": "planner_v1", "question_id": qid, "scope": "multi_event",
            "operation": "sequence", "search_units": units,
            "required_evidence": [{"slot_id": "s1", "description": "both",
                                   "search_unit_ids": ["handcuff", "medical"], "minimum_support": 1}],
            "retrieval_strategy": "multi_target_compare", "candidate_storyline_ids": [],
            "candidate_coarse_ids": ["C07", "C08"], "needs_temporal_neighbors": False,
            "uncertainty_notes": [],
        }
    return {
        "planner_version": "planner_v1", "question_id": qid, "scope": "local",
        "operation": "presence_localisation",
        "search_units": [{"unit_id": "u1", "description": "visible weapon",
                          "query_variants": ["firearm", "gun"], "required_modalities": ["visual"],
                          "temporal_relation": None}],
        "required_evidence": [{"slot_id": "s1", "description": "visible support",
                               "search_unit_ids": ["u1"], "minimum_support": 1}],
        "retrieval_strategy": "targeted", "candidate_storyline_ids": ["E03"],
        "candidate_coarse_ids": ["C07"], "needs_temporal_neighbors": False,
        "uncertainty_notes": [],
    }


def test_frozen_structured_index_unchanged():
    assert sha256_file(INDEX_PATH) == "c3cc1878d8cd4c0e626660b7a5e5a84e6a0f75ac3c9c9615c0f3781862bb7c07"


def test_planner_output_validates():
    assert validate_planner_output(valid_plan(), question=question(), index=load_index())["valid"]


def test_planner_answer_field_rejected():
    plan = valid_plan()
    plan["answer"] = "yes"
    assert not validate_planner_output(plan, question=question(), index=load_index())["valid"]


def test_invalid_storyline_or_coarse_id_rejected():
    plan = valid_plan()
    plan["candidate_storyline_ids"] = ["E99"]
    plan["candidate_coarse_ids"] = ["C99"]
    errors = validate_planner_output(plan, question=question(), index=load_index())["errors"]
    assert "unknown_storyline_id:E99" in errors and "unknown_coarse_id:C99" in errors


def test_unsupported_capability_rejected():
    plan = valid_plan()
    plan["search_units"][0]["required_modalities"] = ["detector"]
    assert "capability_unavailable:detector:has_detector_tags" in validate_planner_output(
        plan, question=question(), index=load_index()
    )["errors"]


def test_global_summary_routes_all_coarse():
    index = load_index()
    routing = route_coarse_nodes(valid_plan("q_global_summary"), index, question("q_global_summary"))
    assert routing["selected_coarse_ids"] == [f"C{i:02d}" for i in range(1, 11)]
    assert routing["selected_storyline_ids"] == ["E01", "E02", "E03"]


def test_targeted_retrieval_obeys_coarse_universe():
    index = load_index()
    plan = valid_plan()
    routing = route_coarse_nodes(plan, index, question())
    embeddings = load_medium_embeddings(index, INDEX_PATH)
    qemb = np.ones((1, 768), dtype=np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True)
    ranking, _ = compute_medium_ranking(
        index=index, index_path=INDEX_PATH, question=question(), plan=plan, routing=routing,
        query_embeddings=qemb, medium_embeddings=embeddings,
    )
    assert {row["parent_coarse_id"] for row in ranking} <= set(routing["selected_coarse_ids"])


def test_multi_target_keeps_search_units_separate():
    index = load_index()
    plan = valid_plan("q_handcuff_before_medical")
    routing = route_coarse_nodes(plan, index, question("q_handcuff_before_medical"))
    assert set(routing["per_search_unit"]) == {"handcuff", "medical"}


def test_medium_score_deterministic_and_formula():
    index = load_index()
    plan = valid_plan()
    routing = route_coarse_nodes(plan, index, question())
    embeddings = load_medium_embeddings(index, INDEX_PATH)
    qemb = np.ones((1, 768), dtype=np.float32)
    qemb /= np.linalg.norm(qemb, axis=1, keepdims=True)
    kwargs = dict(index=index, index_path=INDEX_PATH, question=question(), plan=plan,
                  routing=routing, query_embeddings=qemb, medium_embeddings=embeddings)
    first, selected = compute_medium_ranking(**kwargs)
    second, _ = compute_medium_ranking(**kwargs)
    assert first == second
    row = first[0]
    expected = .6 * row["visual_score_normalized"] + .3 * row["lexical_score"] + .1 * row["coarse_prior"]
    assert row["combined_score"] == pytest.approx(expected)
    assert all(item["medium_id"] in {n["medium_id"] for n in index["medium_nodes"]} for item in selected)


def test_missing_storyline_falls_back_to_all_medium():
    index = load_index()
    index["capabilities"]["has_storyline"] = False
    index["storyline_events"] = []
    plan = valid_plan()
    plan["candidate_storyline_ids"] = []
    routing = route_coarse_nodes(plan, index, question())
    assert routing["selected_coarse_ids"] == [f"C{i:02d}" for i in range(1, 11)]
    assert routing["fallback_reason"] == "has_storyline=false; direct all-Medium retrieval"


def test_invalid_embedding_reference_fails_loudly():
    index = load_index()
    for node in index["medium_nodes"]:
        node["pooled_visual_embedding_ref"]["path"] = "embeddings/missing.npy"
    with pytest.raises(PlannerError, match="Missing embedding"):
        load_medium_embeddings(index, INDEX_PATH)


def test_json_repair_occurs_at_most_once():
    path = TEST_RUNTIME / "repair_once"
    if path.exists():
        shutil.rmtree(path)
    calls = []
    repaired = valid_plan()
    responses = ["not json", json.dumps(repaired)]
    def caller(**kwargs):
        calls.append(kwargs)
        raw = responses[len(calls) - 1]
        return raw, {"input_tokens": 1, "output_tokens": 1, "latency_sec": 0.1}
    parsed, attempts, validation = plan_question(
        question=question(), index=load_index(), output_dir=path, api_key="not-used",
        config=DEFAULT_CONFIG, caller=caller,
    )
    assert validation["valid"] and parsed == repaired and len(attempts) == 2


def test_second_invalid_repair_fails_loudly():
    path = TEST_RUNTIME / "repair_fail"
    if path.exists():
        shutil.rmtree(path)
    calls = []
    def caller(**kwargs):
        calls.append(kwargs)
        return "not json", {"input_tokens": 1, "output_tokens": 1, "latency_sec": 0.1}
    with pytest.raises(PlannerError):
        plan_question(question=question(), index=load_index(), output_dir=path,
                      api_key="not-used", config=DEFAULT_CONFIG, caller=caller)
    assert len(calls) == 2


def test_index_counts_and_dimensions():
    index = load_index()
    assert (len(index["storyline_events"]), len(index["coarse_nodes"]), len(index["medium_nodes"])) == (3, 10, 30)
    assert load_medium_embeddings(index, INDEX_PATH).shape == (30, 768)


def test_no_existing_frozen_reference_modified():
    index = load_index()
    assert index["schema_version"] == "hierarchical_index_v1"
    assert index["capabilities"]["has_detector_tags"] is False
    assert index["capabilities"]["has_tracking"] is False
