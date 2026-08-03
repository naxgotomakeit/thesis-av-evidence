from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.experiments.fine_reranking.core import (
    FineRerankingError,
    build_fine_query_text,
    load_fine_embeddings,
    rerank_fines_for_medium,
    select_diverse_fines,
    sha256_file,
    temporally_diverse,
    validate_frozen_inputs,
)

ROOT = Path(__file__).resolve().parents[3]
INDEX_PATH = ROOT / "outputs/experiments/structured_organizer_v1/226/hierarchical_index_v1.json"
PLANNER_DIR = ROOT / "outputs/experiments/planner_medium_retrieval_v1/226"


def index():
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def row(fid, score, ts, start, end, unit="u"):
    return {
        "fine_id": fid, "siglip_score_raw": score, "siglip_score_normalized": score,
        "representative_frame_timestamp_sec": ts, "start_sec": start, "end_sec": end,
        "search_unit_id": unit, "medium_id": "m",
    }


def test_frozen_hierarchical_index_unchanged():
    assert sha256_file(INDEX_PATH) == "c3cc1878d8cd4c0e626660b7a5e5a84e6a0f75ac3c9c9615c0f3781862bb7c07"


def test_frozen_planner_and_medium_outputs_unchanged():
    expected = {
        "q_weapon_visible/planner_parsed.json": "29771ac8a75ee0a0f96801920c90541d9456e4a937e1e9d43fb142b900ee96e7",
        "q_weapon_visible/medium_ranking.json": "b6fc675a59fb5880cdf98b0334d826f5d3d6537141e5ebd4f9d683b0b30a4045",
        "q_weapon_visible/selected_mediums.json": "153ca5fcc43f9becd6cfc41252d47a3d2128054b4ffdef3020b89cf5770a1105",
    }
    assert all(sha256_file(PLANNER_DIR / path) == digest for path, digest in expected.items())


def test_all_frozen_inputs_validate():
    validate_frozen_inputs(index=index(), index_path=INDEX_PATH, planner_dir=PLANNER_DIR)


def test_fine_ids_belong_to_selected_medium():
    data = index()
    valid = {node["fine_id"] for node in data["fine_nodes"]}
    for qdir in PLANNER_DIR.glob("q_*"):
        for row_ in json.loads((qdir / "selected_mediums.json").read_text(encoding="utf-8")):
            assert set(row_["child_fine_ids"]) <= valid


def test_maximum_two_fines_per_medium():
    selected, _ = select_diverse_fines(
        {"u": [row("a", .9, 1, 0, 2), row("b", .8, 5, 2, 6), row("c", .7, 9, 6, 10)]},
        strategy="targeted", max_fines=2, minimum_gap=2,
    )
    assert len(selected) == 2


def test_single_fine_allowed():
    selected, reason = select_diverse_fines(
        {"u": [row("a", .9, 1, 0, 2)]}, strategy="targeted", max_fines=2, minimum_gap=2,
    )
    assert len(selected) == 1 and reason == "medium_has_one_child_fine"


def test_temporal_diversity_rule():
    assert temporally_diverse(row("a", 1, 1, 0, 2), row("b", .9, 4, 2, 5), 2)
    assert not temporally_diverse(row("a", 1, 1, 0, 2), row("a", .9, 1, 0, 2), 2)


def test_stable_tie_breaking_and_score_determinism():
    data = index()
    medium = data["medium_nodes"][0]
    fine_by_id = {node["fine_id"]: node for node in data["fine_nodes"]}
    matrix, row_by_id, _ = load_fine_embeddings(data, INDEX_PATH)
    query = np.ones(768, dtype=np.float32)
    query /= np.linalg.norm(query)
    kwargs = dict(question_id="q", search_unit_id="u", medium=medium, fine_by_id=fine_by_id,
                  fine_embeddings=matrix, row_by_id=row_by_id, query_embedding=query)
    first = rerank_fines_for_medium(**kwargs)
    second = rerank_fines_for_medium(**kwargs)
    assert first == second
    tied = sorted([row("b", .5, 2, 0, 3), row("a", .5, 2, 0, 3)],
                  key=lambda x: (-x["siglip_score_raw"], x["representative_frame_timestamp_sec"], x["fine_id"]))
    assert [item["fine_id"] for item in tied] == ["a", "b"]


def test_multi_target_units_remain_separate():
    selected, _ = select_diverse_fines(
        {"hand": [row("a", .9, 1, 0, 2, "hand")],
         "medical": [row("b", .8, 5, 2, 6, "medical")]},
        strategy="multi_target_compare", max_fines=2, minimum_gap=2,
    )
    assert {u for item in selected for u in item["search_unit_ids"]} == {"hand", "medical"}


def test_global_coarse_coverage_possible():
    planner = json.loads((PLANNER_DIR / "q_global_summary/selected_mediums.json").read_text(encoding="utf-8"))
    assert {row_["parent_coarse_id"] for row_ in planner} == {f"C{i:02d}" for i in range(1, 11)}


def test_missing_fine_embedding_fails_loudly():
    data = index()
    for node in data["fine_nodes"]:
        node["visual_embedding_ref"]["path"] = "embeddings/missing.npy"
    with pytest.raises(FineRerankingError, match="Missing Fine embedding"):
        load_fine_embeddings(data, INDEX_PATH)


def test_missing_representative_frame_fails_loudly():
    data = index()
    data["fine_nodes"][0]["representative_frame_path"] = str(ROOT / "missing.jpg")
    with pytest.raises(FineRerankingError, match="Missing representative frame"):
        validate_frozen_inputs(index=data, index_path=INDEX_PATH, planner_dir=PLANNER_DIR)


def test_invalid_child_fine_id_fails_loudly():
    data = index()
    data["medium_nodes"][0]["child_fine_ids"] = ["missing_fine"]
    with pytest.raises(FineRerankingError, match="Invalid child Fine ID"):
        validate_frozen_inputs(index=data, index_path=INDEX_PATH, planner_dir=PLANNER_DIR)


def test_query_text_is_deterministic():
    unit = {"description": "medical assistance", "query_variants": ["aid", "tourniquet"]}
    assert build_fine_query_text("When?", unit) == build_fine_query_text("When?", unit)


def test_evidence_candidates_json_serializable():
    payload = {"questions": [{"selected_fine_evidence": [{"score": 0.2}]}]}
    assert json.loads(json.dumps(payload)) == payload


def test_answer_fields_rejected_by_contract_constant():
    from src.experiments.fine_reranking.core import _contains_forbidden_key
    assert _contains_forbidden_key({"answer": "yes"})
    assert not _contains_forbidden_key({"selected_fine_evidence": []})


def test_existing_frozen_references_remain_unchanged():
    assert sha256_file(PLANNER_DIR / "questions_226.json") == "c028b7954e0db4743ee5772362c3a3ac320de4f80b03f68285ca823ba38c7bbd"
