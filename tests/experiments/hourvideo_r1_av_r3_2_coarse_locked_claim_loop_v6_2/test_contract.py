from pathlib import Path

import numpy as np

from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2.core import (
    AMBIGUOUS_SUPPORT_CAVEAT_MARKER,
    BUDGET_EXHAUSTED_CAVEAT_MARKER,
    _allocate_fine_quota_by_coarse,
    _allocate_fine_quota_by_requirement,
    _batch_claim_execution_schema,
    _claim_text_contradicts_status,
    _claim_to_assessment,
    _coarse_locked_planner_schema,
    _dedupe_fine_rows,
    _discriminator_findings_as_facts,
    _discriminator_schema,
    _downgrade_noncompliant_investigation,
    _evidence_ids_mentioned_in_text,
    _excluded_judgments_union,
    _finalize_answer,
    _is_decisive_fact,
    _locked_coarse_ids,
    _medium_lexical_fallback,
    _option_mapping_schema,
    _project_coarse_locked_plan,
    _rank_within_locked_coarse,
    _reason_is_closed_world_refutation,
    _repair_option_mapping_coverage,
    _repair_planner_coverage,
    _shared_investigation_schema,
    _shared_search_plan,
    _validate_batch_claim_execution,
    _validate_coarse_locked_plan,
    _validate_discriminators,
    _validate_option_mapping,
    _validate_shared_investigation,
    _write_run_manifest,
)
from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import option_requirements


class _StubEncoder:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, texts):
        return [self.vector for _ in texts]


QUESTION = {"question_id": "q1", "question_text": "Which?", "answer_options": [{"option_id": "A", "text": "alpha"}, {"option_id": "B", "text": "beta"}]}
COARSE_IDS = ["C01", "C02", "C03"]


def _requirements():
    return option_requirements(QUESTION)


def _judgments(selected_map):
    return [{"coarse_id": cid, "selected": cid in selected_map, "reason": f"reason for {cid}"} for cid in COARSE_IDS]


def _keyed_plan(selected_map):
    requirements = _requirements()

    def unit():
        return {
            "search_description": "find the thing",
            "query_variants": ["find the thing"],
            "modality_strategy": "visual",
            "coarse_judgments": _judgments(selected_map),
        }

    return {
        "question_id": QUESTION["question_id"],
        "requirement_plans": {row["requirement_id"]: unit() for row in requirements},
        "coarse_lock_is_hard_scope": True,
    }


def test_schema_requires_judgment_for_every_coarse_id():
    text = repr(_coarse_locked_planner_schema(_requirements(), COARSE_IDS))
    for cid in COARSE_IDS:
        assert cid in text
    assert "suggested_coarse_ids" not in text


def test_validate_accepts_full_coverage_plan_with_at_least_one_selected():
    requirements = _requirements()
    plan = _project_coarse_locked_plan(_keyed_plan({"C02"}), requirements)
    _validate_coarse_locked_plan(plan, QUESTION, requirements, set(COARSE_IDS))


def test_validate_rejects_missing_coarse_id():
    requirements = _requirements()
    keyed = _keyed_plan({"C02"})
    for unit in keyed["requirement_plans"].values():
        unit["coarse_judgments"] = [j for j in unit["coarse_judgments"] if j["coarse_id"] != "C03"]
    plan = _project_coarse_locked_plan(keyed, requirements)
    try:
        _validate_coarse_locked_plan(plan, QUESTION, requirements, set(COARSE_IDS))
    except ValueError:
        pass
    else:
        raise AssertionError("missing coarse_id judgment passed validation")


def test_validate_accepts_zero_selected_when_no_region_applies():
    # Legitimate on a "which happened first, A or B" question: a distractor option naming neither A
    # nor B can correctly have zero supporting Coarse regions -- real traffic hit this and it must not
    # be treated as a contract violation as long as every region still has a reason.
    requirements = _requirements()
    plan = _project_coarse_locked_plan(_keyed_plan(set()), requirements)
    _validate_coarse_locked_plan(plan, QUESTION, requirements, set(COARSE_IDS))


def test_validate_rejects_empty_reason():
    requirements = _requirements()
    keyed = _keyed_plan({"C02"})
    for unit in keyed["requirement_plans"].values():
        next(j for j in unit["coarse_judgments"] if j["coarse_id"] == "C01")["reason"] = "   "
    plan = _project_coarse_locked_plan(keyed, requirements)
    try:
        _validate_coarse_locked_plan(plan, QUESTION, requirements, set(COARSE_IDS))
    except ValueError:
        pass
    else:
        raise AssertionError("empty reason passed validation")


def test_repair_planner_coverage_fills_missing_coarse_id():
    # Real run (824e7896-904a-4ff5-b7ee-e21df21918c2) crashed here: the schema's per-item coarse_id
    # enum cannot force exactly-once coverage, so the model omitted a region entirely and the old
    # code path (no retry, no repair) crashed the whole run outright instead of degrading.
    requirements = _requirements()
    keyed = _keyed_plan({"C02"})
    for unit in keyed["requirement_plans"].values():
        unit["coarse_judgments"] = [j for j in unit["coarse_judgments"] if j["coarse_id"] != "C03"]
    plan = _project_coarse_locked_plan(keyed, requirements)
    repaired = _repair_planner_coverage(plan, COARSE_IDS)
    _validate_coarse_locked_plan(repaired, QUESTION, requirements, set(COARSE_IDS))
    row = repaired["requirement_plans"][0]
    filled = next(j for j in row["coarse_judgments"] if j["coarse_id"] == "C03")
    assert filled["selected"] is False
    assert filled["reason"]


def test_repair_planner_coverage_drops_duplicates_and_fills_empty_reason():
    requirements = _requirements()
    keyed = _keyed_plan({"C02"})
    for unit in keyed["requirement_plans"].values():
        judgments = unit["coarse_judgments"]
        dup = dict(next(j for j in judgments if j["coarse_id"] == "C01"))
        judgments.append(dup)
        next(j for j in judgments if j["coarse_id"] == "C03")["reason"] = "   "
        unit["coarse_judgments"] = judgments
    plan = _project_coarse_locked_plan(keyed, requirements)
    repaired = _repair_planner_coverage(plan, COARSE_IDS)
    _validate_coarse_locked_plan(repaired, QUESTION, requirements, set(COARSE_IDS))
    row = repaired["requirement_plans"][0]
    assert [j["coarse_id"] for j in row["coarse_judgments"]] == COARSE_IDS
    assert next(j for j in row["coarse_judgments"] if j["coarse_id"] == "C03")["reason"]


def test_repair_planner_coverage_leaves_compliant_plan_untouched():
    requirements = _requirements()
    plan = _project_coarse_locked_plan(_keyed_plan({"C01", "C02"}), requirements)
    repaired = _repair_planner_coverage(plan, COARSE_IDS)
    assert repaired == plan


def _fake_map_doc():
    return {
        "map_type": "structural",
        "semantic_fields_available": False,
        "coarse_regions": [
            {"coarse_id": cid, "start_sec": 0.0, "end_sec": 1.0, "navigation_summary": "s", "audio_channel": []}
            for cid in COARSE_IDS
        ],
    }


def test_call_coarse_locked_planner_retries_past_a_transient_max_tokens_error():
    # Real run (824e7896-904a-4ff5-b7ee-e21df21918c2, r3_2 side with 47 Coarse regions) hit
    # _anthropic_call raising RuntimeError("...reached max_tokens") -- a wide semantic map needs one
    # reason per region per option and can outgrow the budget on a given attempt. The old code had no
    # retry around the call itself (only around the post-hoc ValueError from validation), so this
    # crashed the whole run with no chance to try again.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    requirements = _requirements()
    good_provider = _keyed_plan({"C02"})
    calls = {"n": 0}

    def fake_anthropic_call(cfg, system, payload, schema, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("Anthropic structured response reached max_tokens")
        return good_provider, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fake_anthropic_call
    try:
        plan, usage, payload = v6_core._call_coarse_locked_planner(
            {"anthropic": {"planner_max_tokens": 100}, "max_validation_retries": 2},
            QUESTION, requirements, COARSE_IDS, _fake_map_doc(),
        )
    finally:
        v6_core._anthropic_call = original
    assert calls["n"] == 2
    v6_core._validate_coarse_locked_plan(plan, QUESTION, requirements, set(COARSE_IDS))


def test_call_coarse_locked_planner_raises_when_every_attempt_hits_max_tokens():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    requirements = _requirements()

    def always_fails(cfg, system, payload, schema, max_tokens):
        raise RuntimeError("Anthropic structured response reached max_tokens")

    original = v6_core._anthropic_call
    v6_core._anthropic_call = always_fails
    try:
        try:
            v6_core._call_coarse_locked_planner(
                {"anthropic": {"planner_max_tokens": 100}, "max_validation_retries": 2},
                QUESTION, requirements, COARSE_IDS, _fake_map_doc(),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError to propagate when no plan was ever produced")
    finally:
        v6_core._anthropic_call = original


def test_locked_coarse_ids_returns_only_selected():
    requirements = _requirements()
    plan = _project_coarse_locked_plan(_keyed_plan({"C01", "C03"}), requirements)
    row = plan["requirement_plans"][0]
    assert _locked_coarse_ids(row) == {"C01", "C03"}


def test_project_plan_round_trips_nested_coarse_judgments():
    requirements = _requirements()
    plan = _project_coarse_locked_plan(_keyed_plan({"C02"}), requirements)
    row = plan["requirement_plans"][0]
    by_id = {j["coarse_id"]: j for j in row["coarse_judgments"]}
    assert by_id["C02"]["selected"] is True
    assert by_id["C01"]["selected"] is False


def _ranking_fixture(count):
    hierarchy = {"medium_nodes": [
        {"medium_id": f"M{i}", "start_sec": float(i * 45), "end_sec": float((i + 1) * 45), "source_fine_ids": [f"F{i}"]}
        for i in range(count)
    ]}
    lexical = [{"lexical_source": "vlm_caption", "lexical_text": "kitchen"} for _ in hierarchy["medium_nodes"]]
    requirement_plan = {"requirement_id": "q1::option_a", "search_description": "kitchen activity", "query_variants": ["kitchen activity"]}
    cfg = {"ranking": {"visual_weight": 1.0, "lexical_weight": 0.0}}
    encoder = _StubEncoder([1.0, 0.0])
    return hierarchy, lexical, requirement_plan, cfg, encoder


def test_rank_within_locked_coarse_excludes_higher_scoring_medium_outside_lock():
    hierarchy, lexical, requirement_plan, cfg, encoder = _ranking_fixture(4)
    parent = {"M0": "C_IN", "M1": "C_IN", "M2": "C_OUT", "M3": "C_OUT"}
    medium_embeddings = np.array([[0.5, 0.0], [0.3, 0.0], [0.9, 0.0], [0.1, 0.0]])

    rows, _, _ = _rank_within_locked_coarse(cfg, QUESTION, requirement_plan, hierarchy, lexical, medium_embeddings, parent, encoder, {"C_IN"})

    medium_ids = {row["medium_id"] for row in rows}
    assert medium_ids == {"M0", "M1"}
    assert rows[0]["medium_id"] == "M0"


def test_rank_within_locked_coarse_keeps_every_locked_medium_no_top_k_truncation():
    hierarchy, lexical, requirement_plan, cfg, encoder = _ranking_fixture(5)
    parent = {f"M{i}": "C_IN" for i in range(5)}
    medium_embeddings = np.array([[float(i) / 10.0, 0.0] for i in range(5)])

    rows, _, _ = _rank_within_locked_coarse(cfg, QUESTION, requirement_plan, hierarchy, lexical, medium_embeddings, parent, encoder, {"C_IN"})

    assert len(rows) == 5


def test_rank_within_locked_coarse_raises_when_lock_has_no_candidates():
    hierarchy, lexical, requirement_plan, cfg, encoder = _ranking_fixture(1)
    parent = {"M0": "C_OUT"}
    medium_embeddings = np.array([[0.5, 0.0]])
    try:
        _rank_within_locked_coarse(cfg, QUESTION, requirement_plan, hierarchy, lexical, medium_embeddings, parent, encoder, {"C_IN"})
    except ValueError:
        pass
    else:
        raise AssertionError("empty candidate set did not raise")


EVIDENCE = [{"evidence_id": "E1"}, {"evidence_id": "E2"}]
EXCLUDED_COARSE_IDS = {"C01", "C02"}


def _investigation(overrides=None):
    overrides = overrides or {}
    value = {
        "question_id": QUESTION["question_id"], "investigation_status": "unresolved",
        "established_facts": "", "gap_reason": "map does not show a clear boundary",
        "requested_coarse_ids": [], "cited_evidence_ids": [],
    }
    value.update(overrides)
    if "atomic_facts" not in overrides:
        # Auto-derive a matching atomic_facts entry so existing tests that only set established_facts
        # don't all need updating for the new required field -- tests that specifically exercise
        # atomic_facts pass it explicitly via overrides instead.
        value["atomic_facts"] = (
            [{"text": value["established_facts"], "evidence_ids": list(value["cited_evidence_ids"])}]
            if value["established_facts"] else []
        )
    if "discriminator_findings" not in overrides:
        # Same rationale: existing tests predate the discriminator checklist and don't supply any
        # discriminators, so an empty findings list is always valid for them (coverage is only checked
        # when discriminators is non-empty). Tests exercising discriminator_findings pass it explicitly.
        value["discriminator_findings"] = []
    return value


def test_shared_investigation_schema_limits_requested_coarse_to_excluded_set():
    text = repr(_shared_investigation_schema(QUESTION, EVIDENCE, list(EXCLUDED_COARSE_IDS)))
    assert "C01" in text
    assert "C02" in text


def test_shared_investigation_schema_drops_enum_when_nothing_excluded():
    # Same Anthropic empty-enum/const limitation as everywhere else in this module: falls back to an
    # unconstrained array; _call_shared_investigation clips hallucinated ids after the call instead.
    schema = _shared_investigation_schema(QUESTION, EVIDENCE, [])
    unit = schema["properties"]["requested_coarse_ids"]
    assert unit == {"type": "array", "items": {"type": "string"}}


def test_shared_investigation_schema_drops_evidence_enum_when_evidence_empty():
    schema = _shared_investigation_schema(QUESTION, [], list(EXCLUDED_COARSE_IDS))
    unit = schema["properties"]["cited_evidence_ids"]
    assert unit == {"type": "array", "items": {"type": "string"}}


def test_validate_shared_investigation_accepts_well_formed_unresolved():
    _validate_shared_investigation(_investigation(), QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)


def test_validate_shared_investigation_rejects_requested_coarse_outside_excluded_set():
    value = _investigation({"requested_coarse_ids": ["C99"]})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-scope requested coarse id passed validation")


def test_validate_shared_investigation_resolved_requires_facts_and_no_further_request():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "", "cited_evidence_ids": ["E1"],
    })
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("resolved investigation with empty established_facts passed validation")

    value2 = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": ["E1"], "requested_coarse_ids": ["C01"],
    })
    try:
        _validate_shared_investigation(value2, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("resolved investigation that still requests more evidence passed validation")


def test_validate_shared_investigation_resolved_requires_citation_when_evidence_exists():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": [],
    })
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("resolved investigation with zero cited evidence passed validation")


def test_validate_shared_investigation_resolved_rejects_empty_evidence_unlike_old_per_option_exception():
    # Deliberate difference from the old per-option claim validator: there is no "this option's text is
    # categorically off-topic" escape hatch here. Zero evidence (Coarse AND Medium fallback both empty)
    # must never legitimately support "resolved" -- this is exactly the false-refutation-from-silence
    # bug confirmed on ab93e55b's option C (0/41 Coarse regions selected, refuted from inference alone).
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "nothing happens here",
        "cited_evidence_ids": [],
    })
    try:
        _validate_shared_investigation(value, QUESTION, [], EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("resolved investigation with zero evidence at all passed validation")


def test_validate_shared_investigation_resolved_with_real_citation_passes():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": ["E1"],
    })
    _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)


def test_validate_shared_investigation_unresolved_requires_gap_reason():
    value = _investigation({"gap_reason": "   "})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("unresolved investigation with empty gap_reason passed validation")


def test_validate_shared_investigation_rejects_unknown_evidence_id():
    value = _investigation({"cited_evidence_ids": ["E99"]})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown evidence id passed validation")


def test_validate_shared_investigation_rejects_self_contradictory_facts():
    value = _investigation({
        "investigation_status": "resolved",
        "established_facts": "Actually this is refuted, not supported, by the evidence.",
        "cited_evidence_ids": ["E1"],
    })
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("self-contradictory established_facts passed validation")


def test_downgrade_noncompliant_investigation_fixes_missing_citation():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": [],
    })
    downgraded = _downgrade_noncompliant_investigation(value, EVIDENCE)
    assert downgraded["investigation_status"] == "unresolved"
    assert downgraded["established_facts"] == ""
    assert downgraded["gap_reason"]
    _validate_shared_investigation(downgraded, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)


def test_downgrade_noncompliant_investigation_fixes_resolved_with_zero_evidence():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "nothing happens here",
        "cited_evidence_ids": [],
    })
    downgraded = _downgrade_noncompliant_investigation(value, [])
    assert downgraded["investigation_status"] == "unresolved"
    _validate_shared_investigation(downgraded, QUESTION, [], EXCLUDED_COARSE_IDS)


def test_downgrade_noncompliant_investigation_leaves_compliant_result_untouched():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": ["E1"],
    })
    downgraded = _downgrade_noncompliant_investigation(value, EVIDENCE)
    assert downgraded == value


MEDIUM_EVIDENCE = [
    {"evidence_id": "medium_detail::q1::shared::M001"},
    {"evidence_id": "medium_detail::q1::shared::M009"},
    {"evidence_id": "semantic_coarse::q1::shared::C01"},
]


def test_evidence_ids_mentioned_in_text_matches_real_ids_only():
    # Real case (70f2a750, 2026-08-07): established_facts mentioned M009 inline ("chopping red pepper
    # and mushrooms (M009, M010)") but M010 has no matching evidence entry here -- only M009 should
    # resolve to a real evidence_id; an unrelated-looking token must not spuriously match.
    text = "Chopping vegetables (M009, M010) near the counter, unrelated M999 mention, see C01 too."
    result = _evidence_ids_mentioned_in_text(text, MEDIUM_EVIDENCE)
    assert result == {"medium_detail::q1::shared::M009", "semantic_coarse::q1::shared::C01"}


def test_evidence_ids_mentioned_in_text_empty_when_nothing_matches():
    assert _evidence_ids_mentioned_in_text("no ids here at all", MEDIUM_EVIDENCE) == set()


def test_validate_shared_investigation_rejects_established_facts_mentioning_uncited_evidence():
    # Real bug (70f2a750, 2026-08-07): established_facts mentioned M009 inline (the detail that
    # distinguished two answer options) but cited_evidence_ids never included it -- a resolved
    # investigation must not be allowed to silently under-cite what it actually drew on.
    value = _investigation({
        "investigation_status": "resolved",
        "established_facts": "The wearer chops vegetables (M009) then cooks.",
        "cited_evidence_ids": ["medium_detail::q1::shared::M001"],
    })
    try:
        _validate_shared_investigation(value, QUESTION, MEDIUM_EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("resolved investigation under-citing a mentioned evidence id passed validation")


def test_validate_shared_investigation_accepts_established_facts_when_all_mentions_are_cited():
    value = _investigation({
        "investigation_status": "resolved",
        "established_facts": "The wearer chops vegetables (M009) then cooks.",
        "cited_evidence_ids": ["medium_detail::q1::shared::M009"],
    })
    _validate_shared_investigation(value, QUESTION, MEDIUM_EVIDENCE, EXCLUDED_COARSE_IDS)


def test_downgrade_noncompliant_investigation_fixes_undercited_established_facts():
    value = _investigation({
        "investigation_status": "resolved",
        "established_facts": "The wearer chops vegetables (M009) then cooks.",
        "cited_evidence_ids": ["medium_detail::q1::shared::M001"],
    })
    downgraded = _downgrade_noncompliant_investigation(value, MEDIUM_EVIDENCE)
    assert downgraded["investigation_status"] == "unresolved"
    assert downgraded["established_facts"] == ""
    assert downgraded["gap_reason"]
    _validate_shared_investigation(downgraded, QUESTION, MEDIUM_EVIDENCE, EXCLUDED_COARSE_IDS)


def _hierarchy_for_fallback():
    hierarchy = {"medium_nodes": [
        {"medium_id": "M1", "start_sec": 0.0, "end_sec": 45.0, "source_fine_ids": ["F1"]},
        {"medium_id": "M2", "start_sec": 45.0, "end_sec": 90.0, "source_fine_ids": ["F2"]},
    ]}
    return hierarchy


def test_medium_lexical_fallback_scores_and_ranks_by_query_overlap(monkeypatch=None):
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    hierarchy = _hierarchy_for_fallback()

    def fake_lexical_rows(side, projections, captions):
        return [
            {"lexical_source": "vlm_caption", "lexical_text": "grabbing tabs and throwing them on the table"},
            {"lexical_source": "vlm_caption", "lexical_text": "assembling a puzzle on a table"},
        ]

    original = v6_core._lexical_rows
    v6_core._lexical_rows = fake_lexical_rows
    try:
        evidence = v6_core._medium_lexical_fallback(
            "r3_2", QUESTION, ["grabbing tabs box throwing table"], hierarchy, [], [],
        )
    finally:
        v6_core._lexical_rows = original
    assert evidence
    assert evidence[0]["evidence_type"] == "medium_lexical_fallback"
    assert evidence[0]["evidence_id"].endswith("M1")  # the higher lexical-overlap Medium ranks first


def test_medium_lexical_fallback_returns_empty_when_nothing_matches():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    hierarchy = _hierarchy_for_fallback()

    def fake_lexical_rows(side, projections, captions):
        return [{"lexical_source": "vlm_caption", "lexical_text": "zzz qqq xyz"} for _ in hierarchy["medium_nodes"]]

    original = v6_core._lexical_rows
    v6_core._lexical_rows = fake_lexical_rows
    try:
        evidence = v6_core._medium_lexical_fallback("r3_2", QUESTION, ["grabbing tabs"], hierarchy, [], [])
    finally:
        v6_core._lexical_rows = original
    assert evidence == []


def test_excluded_judgments_union_keeps_first_seen_reason_across_options():
    requirements = _requirements()
    plan = {"requirement_plans": [
        {"requirement_id": requirements[0]["requirement_id"], "coarse_judgments": [
            {"coarse_id": "C01", "selected": False, "reason": "option A: no mention"},
            {"coarse_id": "C02", "selected": True, "reason": "option A: matches"},
        ]},
        {"requirement_id": requirements[1]["requirement_id"], "coarse_judgments": [
            {"coarse_id": "C01", "selected": False, "reason": "option B: also no mention"},
            {"coarse_id": "C02", "selected": False, "reason": "option B: does not apply"},
        ]},
    ]}
    result = _excluded_judgments_union(plan, locked={"C02"})
    assert [row["coarse_id"] for row in result] == ["C01"]
    assert result[0]["reason"] == "option A: no mention"


def test_shared_search_plan_unions_query_variants_without_duplicates():
    plan = {"requirement_plans": [
        {"search_description": "find X", "query_variants": ["find X", "locate X"], "modality_strategy": "visual"},
        {"search_description": "find X", "query_variants": ["locate X", "spot X"], "modality_strategy": "visual"},
    ]}
    result = _shared_search_plan(plan, "q1::shared")
    assert result["requirement_id"] == "q1::shared"
    assert result["query_variants"] == ["find X", "locate X", "spot X"]


def test_shared_investigation_schema_requires_atomic_facts():
    text = repr(_shared_investigation_schema(QUESTION, EVIDENCE, list(EXCLUDED_COARSE_IDS)))
    assert "atomic_facts" in text


def test_validate_shared_investigation_rejects_empty_atomic_facts_when_resolved():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": ["E1"], "atomic_facts": [],
    })
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("resolved investigation with empty atomic_facts passed validation")


def test_validate_shared_investigation_rejects_atomic_fact_citing_unknown_evidence():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": ["E1"], "atomic_facts": [{"text": "organizing took longer", "evidence_ids": ["E99"]}],
    })
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)
    except ValueError:
        pass
    else:
        raise AssertionError("atomic fact citing unknown evidence passed validation")


def test_downgrade_noncompliant_investigation_fixes_missing_atomic_facts():
    value = _investigation({
        "investigation_status": "resolved", "established_facts": "organizing took longer",
        "cited_evidence_ids": ["E1"], "atomic_facts": [],
    })
    downgraded = _downgrade_noncompliant_investigation(value, EVIDENCE)
    assert downgraded["investigation_status"] == "unresolved"
    assert downgraded["atomic_facts"] == []
    _validate_shared_investigation(downgraded, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)


def test_downgrade_noncompliant_investigation_fixes_self_contradiction():
    # Real crash (9a7a189e r1_av, 2026-08-08, all-4-fixes 10-video run): this downgrade path only ever
    # handled missing-grounds failures -- a self-contradictory resolved investigation (established_facts
    # arguing the opposite of investigation_status=resolved) passed through untouched, so
    # _validate_shared_investigation raised the identical error a second time and it propagated uncaught,
    # crashing run_live. This has otherwise-valid grounds (real citation, real atomic_facts) so only the
    # self-contradiction condition should be what triggers the downgrade here.
    value = _investigation({
        "investigation_status": "resolved",
        "established_facts": "Organizing took longer. Correction: this is refuted, not supported, by the evidence.",
        "cited_evidence_ids": ["E1"], "atomic_facts": [{"text": "Organizing took longer.", "evidence_ids": ["E1"]}],
    })
    downgraded = _downgrade_noncompliant_investigation(value, EVIDENCE)
    assert downgraded["investigation_status"] == "unresolved"
    assert downgraded["established_facts"] == ""
    assert downgraded["gap_reason"]
    _validate_shared_investigation(downgraded, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS)


def test_call_shared_investigation_never_crashes_on_persistent_self_contradiction():
    # End-to-end regression for the actual crash path, per the user's explicit ask: self-contradiction ->
    # every retry attempt fails the same way -> downgrade -> the call must return a valid unresolved
    # result and the process must continue, never raise past _call_shared_investigation.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    def always_self_contradictory(cfg, system, payload, schema, max_tokens):
        provider = {
            "question_id": QUESTION["question_id"], "investigation_status": "resolved",
            "established_facts": "Organizing took longer. Correction: this is refuted, not supported, by the evidence.",
            "atomic_facts": [{"text": "Organizing took longer.", "evidence_ids": ["E1"]}],
            "discriminator_findings": [],
            "gap_reason": "", "requested_coarse_ids": [], "cited_evidence_ids": ["E1"],
        }
        return provider, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = always_self_contradictory
    try:
        result, usage = v6_core._call_shared_investigation(
            {"anthropic": {"claim_max_tokens": 400}, "max_validation_retries": 2},
            QUESTION, EVIDENCE, [], round_number=0,
        )
    finally:
        v6_core._anthropic_call = original
    assert result["investigation_status"] == "unresolved"
    assert result["established_facts"] == ""


DECISIVE_REQUIREMENTS = [
    {"requirement_id": "q1::option_a", "description": "Option text: Kitchen, Living Room, Bathroom"},
    {"requirement_id": "q1::option_c", "description": "Option text: Kitchen, Living Room, Bedroom"},
]


def test_is_decisive_fact_true_when_fact_touches_some_but_not_all_options():
    # "Bedroom" appears in option C's wording but not option A's -- this fact could change which
    # option wins, so it's worth verifying.
    assert _is_decisive_fact("The wearer organized items in the Bedroom.", DECISIVE_REQUIREMENTS) is True


def test_is_decisive_fact_false_when_fact_touches_all_options_equally():
    assert _is_decisive_fact("The wearer spent time in the Kitchen.", DECISIVE_REQUIREMENTS) is False


def test_is_decisive_fact_false_when_fact_touches_no_option():
    assert _is_decisive_fact("Background music played throughout.", DECISIVE_REQUIREMENTS) is False


FACT_VERIFICATION_EVIDENCE = [{"evidence_id": "semantic_coarse::q1::shared::C17", "source_content": "Ego organizes a desk in a messy room."}]


def test_verify_decisive_facts_rejects_overclaim_locally_no_api_call():
    # Real case (824e7896, 2026-08-08): cited text only ever says "a messy room"; a claimed fact of
    # "Bedroom" has zero lexical overlap with it and must be flagged for free, no API call spent.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not call the API for a zero-overlap overclaim")

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fail_if_called
    try:
        atomic_facts = [{"text": "The wearer organized items in the Bedroom.", "evidence_ids": ["semantic_coarse::q1::shared::C17"]}]
        results, calls = v6_core._verify_decisive_facts(
            {"anthropic": {"claim_max_tokens": 400}}, atomic_facts, DECISIVE_REQUIREMENTS, FACT_VERIFICATION_EVIDENCE,
        )
    finally:
        v6_core._anthropic_call = original
    assert calls == []
    assert results[0]["checked"] is True
    assert results[0]["verified"] is False


def test_verify_decisive_facts_skips_nondecisive_facts_without_checking():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not call the API for a non-decisive fact")

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fail_if_called
    try:
        atomic_facts = [{"text": "The wearer spent time in the Kitchen.", "evidence_ids": ["semantic_coarse::q1::shared::C17"]}]
        results, calls = v6_core._verify_decisive_facts(
            {"anthropic": {"claim_max_tokens": 400}}, atomic_facts, DECISIVE_REQUIREMENTS, FACT_VERIFICATION_EVIDENCE,
        )
    finally:
        v6_core._anthropic_call = original
    assert calls == []
    assert results[0]["checked"] is False
    assert results[0]["verified"] is True


def test_verify_decisive_facts_calls_api_when_lexical_overlap_is_ambiguous():
    # Decisive (mentions "Bedroom", a distinguishing term) AND shares some wording ("organizes", "desk")
    # with the cited text without being an exact match -- not confidently zero, so this must escalate to
    # the small verification call rather than being silently passed or failed by the local check alone.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    def fake_anthropic_call(cfg, system, payload, schema, max_tokens):
        return {"entailed": True, "reason": "matches"}, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fake_anthropic_call
    try:
        atomic_facts = [{"text": "The wearer organizes a desk in the Bedroom.", "evidence_ids": ["semantic_coarse::q1::shared::C17"]}]
        results, calls = v6_core._verify_decisive_facts(
            {"anthropic": {"claim_max_tokens": 400}}, atomic_facts, DECISIVE_REQUIREMENTS, FACT_VERIFICATION_EVIDENCE,
        )
    finally:
        v6_core._anthropic_call = original
    assert len(calls) == 1
    assert results[0]["checked"] is True
    assert results[0]["verified"] is True


REAL_824E7896_REQUIREMENTS = [
    {"requirement_id": "824e7896_17_8::option_a", "description": "Option text: Kitchen, living room, bathroom"},
    {"requirement_id": "824e7896_17_8::option_b", "description": "Option text: Kitchen, Living Room"},
    {"requirement_id": "824e7896_17_8::option_c", "description": "Option text: Kitchen, Living Room, Bedroom"},
    {"requirement_id": "824e7896_17_8::option_d", "description": "Option text: Kitchen, Living Room, Dining Room"},
    {"requirement_id": "824e7896_17_8::option_e", "description": "Option text: Kitchen, Living Room, Office"},
]
REAL_824E7896_EVIDENCE = [{
    "evidence_id": "medium_detail::824e7896_17_8::shared::M017",
    "source_content": "Ego is organizing a desk in a messy room.",
}]


def test_is_decisive_fact_not_masked_by_a_word_shared_across_every_option():
    # Real bug caught by a zero-API replay against the actual historical 824e7896 artifact (2026-08-08):
    # the fact's word set as a whole intersects EVERY option's word set once "room" is in the mix (all 5
    # options mention "Living Room"), which masked the one genuinely decisive word ("bedroom", present
    # only in option C). Must check per-word, not per-fact-as-a-whole.
    text = "The camera wearer interacted with objects in a Messy Room/Bedroom, organizing a desk (720-765s)."
    assert _is_decisive_fact(text, REAL_824E7896_REQUIREMENTS) is True


def test_verify_decisive_facts_regression_824e7896_bedroom_overclaim():
    # Deterministic regression test built from the actual historical failure, not a synthetic case: this
    # is the real cited Medium caption for the region the original (pre-fix) established_facts cited to
    # support asserting "a Messy Room/Bedroom" -- the real caption only ever says "a messy room". Lexical
    # overlap is non-zero here (the fact echoes "organizing a desk"/"messy room" from its own source), so
    # this must escalate to the small semantic-verification call rather than being free-rejected locally
    # -- the semantic tier is mocked (zero API cost) with the verdict a competent judge should give, to
    # confirm the full pipeline (decisive detection -> escalation -> verified=False) is wired correctly.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    def mock_semantic_verdict(cfg, system, payload, schema, max_tokens):
        return (
            {"entailed": False, "reason": "The cited text only says 'a messy room'; it never states or implies 'Bedroom' specifically."},
            {"provider": "anthropic", "output_tokens": 1},
        )

    original = v6_core._anthropic_call
    v6_core._anthropic_call = mock_semantic_verdict
    try:
        atomic_facts = [{
            "text": "The camera wearer interacted with objects in a Messy Room/Bedroom, organizing a desk (720-765s).",
            "evidence_ids": ["medium_detail::824e7896_17_8::shared::M017"],
        }]
        results, calls = v6_core._verify_decisive_facts(
            {"anthropic": {"claim_max_tokens": 400}}, atomic_facts, REAL_824E7896_REQUIREMENTS, REAL_824E7896_EVIDENCE,
        )
    finally:
        v6_core._anthropic_call = original
    assert len(calls) == 1
    assert results[0]["checked"] is True
    assert results[0]["verified"] is False


def test_verified_facts_filter_removes_the_real_bedroom_overclaim_from_mapping_input():
    # End-to-end regression, real historical data (824e7896, 2026-08-08): confirms the fix for codex's
    # first correction -- mapping must never be ABLE to see the overclaimed fact, not merely be told not
    # to use it. Builds verified_facts exactly the way _resolve_requirements does (atomic_facts minus
    # anything checked-and-unverified) and confirms the Bedroom claim is structurally absent.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    atomic_facts = [
        {"text": "The camera wearer visited the Kitchen.", "evidence_ids": ["medium_detail::824e7896_17_8::shared::M008"]},
        {"text": "The camera wearer visited the Living Room.", "evidence_ids": ["medium_detail::824e7896_17_8::shared::M015"]},
        {
            "text": "The camera wearer interacted with objects in a Messy Room/Bedroom, organizing a desk (720-765s).",
            "evidence_ids": ["medium_detail::824e7896_17_8::shared::M017"],
        },
    ]
    evidence = [{
        "evidence_id": "medium_detail::824e7896_17_8::shared::M017",
        "source_content": "Ego is organizing a desk in a messy room.",
    }]

    def mock_semantic_verdict(cfg, system, payload, schema, max_tokens):
        return (
            {"entailed": False, "reason": "only 'a messy room' is stated, not Bedroom"},
            {"provider": "anthropic", "output_tokens": 1},
        )

    original = v6_core._anthropic_call
    v6_core._anthropic_call = mock_semantic_verdict
    try:
        fact_verification, calls = v6_core._verify_decisive_facts(
            {"anthropic": {"claim_max_tokens": 400}}, atomic_facts, REAL_824E7896_REQUIREMENTS, evidence,
        )
    finally:
        v6_core._anthropic_call = original

    unverified_texts = {r["text"] for r in fact_verification if r["checked"] and not r["verified"]}
    verified_facts = [f for f in atomic_facts if f["text"] not in unverified_texts]
    verified_texts = {f["text"] for f in verified_facts}
    assert "The camera wearer interacted with objects in a Messy Room/Bedroom, organizing a desk (720-765s)." not in verified_texts
    assert verified_texts == {"The camera wearer visited the Kitchen.", "The camera wearer visited the Living Room."}


def test_call_option_mapping_payload_only_includes_verified_facts():
    # Codex's correction (2026-08-08): mapping must not have any side channel back to raw evidence --
    # no established_facts, no full_evidence. Its entire knowledge is the verified_facts list.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    captured = {}

    def fake_anthropic_call(cfg, system, payload, schema, max_tokens):
        captured["payload"] = payload
        return _mapping(), {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fake_anthropic_call
    try:
        verified_facts = [{"text": "Kitchen was visited.", "evidence_ids": ["E2"]}]
        v6_core._call_option_mapping(
            {"anthropic": {"claim_max_tokens": 400}, "max_validation_retries": 2}, QUESTION, _requirements(),
            verified_facts,
        )
    finally:
        v6_core._anthropic_call = original
    assert captured["payload"]["verified_facts"] == verified_facts
    assert "established_facts" not in captured["payload"]
    assert "full_evidence" not in captured["payload"]
    assert "unverified_facts" not in captured["payload"]


MAPPING_FACT_TEXTS = ["fact A supports this", "fact B supports this"]


def _mapping(overrides_by_rid=None):
    overrides_by_rid = overrides_by_rid or {}
    requirements = _requirements()
    rows = []
    for row in requirements:
        rid = row["requirement_id"]
        unit = {
            "requirement_id": rid, "status": "unresolved", "used_facts": [],
            "reason": "not enough distinguishing detail",
        }
        unit.update(overrides_by_rid.get(rid, {}))
        rows.append(unit)
    return {"question_id": QUESTION["question_id"], "option_assessments": rows}


def test_option_mapping_schema_covers_every_requirement():
    requirements = _requirements()
    text = repr(_option_mapping_schema(requirements, MAPPING_FACT_TEXTS))
    for row in requirements:
        assert row["requirement_id"] in text


def test_validate_option_mapping_accepts_well_formed_result():
    requirements = _requirements()
    _validate_option_mapping(_mapping(), QUESTION, requirements, MAPPING_FACT_TEXTS)


def test_validate_option_mapping_rejects_missing_requirement():
    requirements = _requirements()
    value = _mapping()
    value["option_assessments"] = value["option_assessments"][:-1]
    try:
        _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)
    except ValueError:
        pass
    else:
        raise AssertionError("missing requirement in mapping passed validation")


def test_validate_option_mapping_rejects_empty_reason():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {"reason": "   "}})
    try:
        _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)
    except ValueError:
        pass
    else:
        raise AssertionError("empty mapping reason passed validation")


def test_validate_option_mapping_rejects_self_contradictory_reason():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "refuted", "used_facts": ["fact A supports this"],
        "reason": "This option is supported, not refuted, by established_facts.",
    }})
    try:
        _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)
    except ValueError:
        pass
    else:
        raise AssertionError("self-contradictory mapping reason passed validation")


def test_validate_option_mapping_rejects_supported_without_used_facts():
    # Codex's correction (2026-08-08): used_facts is the citation trail for the option's own conclusion
    # -- a supported/refuted verdict with nothing in used_facts has no traceable basis.
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "supported", "used_facts": [], "reason": "matches the facts",
    }})
    try:
        _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)
    except ValueError:
        pass
    else:
        raise AssertionError("supported mapping with empty used_facts passed validation")


def test_validate_option_mapping_rejects_unknown_used_fact():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "supported", "used_facts": ["a fact that was never shown to mapping"], "reason": "matches",
    }})
    try:
        _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)
    except ValueError:
        pass
    else:
        raise AssertionError("used_facts referencing an unknown fact passed validation")


def test_validate_option_mapping_allows_multiple_supported():
    # Deliberately NOT enforced here: if verified_facts genuinely does not distinguish between two
    # options, the mapping call may legitimately mark both supported -- the existing _finalize_answer
    # AMBIGUOUS_SUPPORT_CAVEAT_MARKER check downstream is what prevents this from being called
    # "confirmed", not a hard rejection at the mapping-validation layer.
    requirements = _requirements()
    value = _mapping({
        requirements[0]["requirement_id"]: {
            "status": "supported", "used_facts": ["fact A supports this"], "reason": "matches verified_facts",
        },
        requirements[1]["requirement_id"]: {
            "status": "supported", "used_facts": ["fact B supports this"], "reason": "also matches verified_facts",
        },
    })
    _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)


def test_repair_option_mapping_coverage_fills_missing_requirement():
    requirements = _requirements()
    value = _mapping()
    value["option_assessments"] = value["option_assessments"][:-1]
    repaired = _repair_option_mapping_coverage(value, requirements, MAPPING_FACT_TEXTS)
    _validate_option_mapping(repaired, QUESTION, requirements, MAPPING_FACT_TEXTS)
    assert len(repaired["option_assessments"]) == len(requirements)


def test_repair_option_mapping_coverage_fixes_self_contradiction():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "refuted", "used_facts": ["fact A supports this"],
        "reason": "This option is supported, not refuted, by established_facts.",
    }})
    repaired = _repair_option_mapping_coverage(value, requirements, MAPPING_FACT_TEXTS)
    _validate_option_mapping(repaired, QUESTION, requirements, MAPPING_FACT_TEXTS)
    fixed = next(r for r in repaired["option_assessments"] if r["requirement_id"] == requirements[0]["requirement_id"])
    assert fixed["status"] == "unresolved"


def test_repair_option_mapping_coverage_fixes_missing_used_facts():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "supported", "used_facts": [], "reason": "matches",
    }})
    repaired = _repair_option_mapping_coverage(value, requirements, MAPPING_FACT_TEXTS)
    _validate_option_mapping(repaired, QUESTION, requirements, MAPPING_FACT_TEXTS)
    fixed = next(r for r in repaired["option_assessments"] if r["requirement_id"] == requirements[0]["requirement_id"])
    assert fixed["status"] == "unresolved"


def test_reason_is_closed_world_refutation_catches_the_real_70f2a750_phrase():
    # Real bug: mapping refuted option E with "...these are not explicitly characterized as 'waste
    # disposal' in the established facts, making this addition unsupported by the evidence" --
    # treating its own summary's silence as a positive contradiction.
    text = "These are not explicitly characterized as 'waste disposal' in the established facts."
    assert _reason_is_closed_world_refutation("refuted", text) is True
    assert _reason_is_closed_world_refutation("unresolved", text) is False
    assert _reason_is_closed_world_refutation("refuted", "The evidence directly shows the opposite occurred.") is False


def test_validate_option_mapping_rejects_closed_world_refutation():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "refuted", "used_facts": ["fact A supports this"],
        "reason": "This is not mentioned anywhere in the established facts.",
    }})
    try:
        _validate_option_mapping(value, QUESTION, requirements, MAPPING_FACT_TEXTS)
    except ValueError:
        pass
    else:
        raise AssertionError("closed-world refutation (silence, not contradiction) passed validation")


def test_repair_option_mapping_coverage_fixes_closed_world_refutation():
    requirements = _requirements()
    value = _mapping({requirements[0]["requirement_id"]: {
        "status": "refuted", "used_facts": ["fact A supports this"],
        "reason": "This is not mentioned anywhere in the established facts.",
    }})
    repaired = _repair_option_mapping_coverage(value, requirements, MAPPING_FACT_TEXTS)
    _validate_option_mapping(repaired, QUESTION, requirements, MAPPING_FACT_TEXTS)
    fixed = next(r for r in repaired["option_assessments"] if r["requirement_id"] == requirements[0]["requirement_id"])
    assert fixed["status"] == "unresolved"


def test_claim_to_assessment_never_maps_refuted_to_supported():
    # This is the exact bug real traffic hit: option_d's claim_text said "Option D is incorrect" but
    # the old resolved:bool design still labeled the assessment status "supported". option_status
    # must carry the polarity itself so this can't happen.
    refuted = {"requirement_id": "q1::option_d", "option_status": "refuted", "claim_text": "Option D is incorrect.", "cited_evidence_ids": ["E1"]}
    assessment = _claim_to_assessment(refuted)
    assert assessment["status"] != "supported"
    assert assessment["status"] == "not_found"


def test_claim_to_assessment_supported_and_unresolved_map_correctly():
    supported = {"requirement_id": "q1::option_a", "option_status": "supported", "claim_text": "A is right.", "cited_evidence_ids": ["E1"]}
    assert _claim_to_assessment(supported)["status"] == "supported"
    unresolved = {"requirement_id": "q1::option_b", "option_status": "unresolved", "gap_reason": "not enough evidence", "cited_evidence_ids": []}
    assert _claim_to_assessment(unresolved)["status"] == "uncertain"


def test_finalize_answer_confirmed_when_selected_option_is_supported():
    answer = {"selected_option_id": "A", "caveats": [], "supporting_evidence_ids": ["E1"]}
    per_requirement_claim = {"q1::option_a": {"option_status": "supported"}}
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["final_status"] == "confirmed"
    assert result["caveats"] == []


def test_finalize_answer_downgrades_confirmed_with_zero_citation():
    # 819c8af7 had all 5 options refuted, so _final had to pick one anyway; if _final had labeled that
    # pick "confirmed" while citing nothing, the same ungrounded-claim failure this design exists to
    # close would resurface one layer up, past _validate_claim's citation rule for individual options.
    answer = {"selected_option_id": "A", "caveats": [], "supporting_evidence_ids": []}
    per_requirement_claim = {"q1::option_a": {"option_status": "supported"}}
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["final_status"] == "budget_exhausted_guess"
    assert any(BUDGET_EXHAUSTED_CAVEAT_MARKER in row for row in result["caveats"])


def test_finalize_answer_confirmed_when_uniquely_supported_among_others():
    answer = {"selected_option_id": "A", "caveats": [], "supporting_evidence_ids": ["E1"]}
    per_requirement_claim = {
        "q1::option_a": {"option_status": "supported"},
        "q1::option_b": {"option_status": "refuted"},
        "q1::option_c": {"option_status": "unresolved"},
    }
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["final_status"] == "confirmed"
    assert result["caveats"] == []


def test_finalize_answer_downgrades_when_another_option_also_supported():
    # Real traffic (10-video pilot, 2026-08-07): 7 of 13 confirmed finals had more than one option
    # independently marked supported (e.g. a workout question where both "Squats" and "Sprinting
    # Drills" ended up supported) -- selecting one and calling it "confirmed" asserts an exclusivity
    # the investigation never actually established.
    answer = {"selected_option_id": "A", "caveats": [], "supporting_evidence_ids": ["E1"]}
    per_requirement_claim = {
        "q1::option_a": {"option_status": "supported"},
        "q1::option_b": {"option_status": "supported"},
    }
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["final_status"] == "budget_exhausted_guess"
    assert any(AMBIGUOUS_SUPPORT_CAVEAT_MARKER in row for row in result["caveats"])
    assert any("B" in row for row in result["caveats"] if AMBIGUOUS_SUPPORT_CAVEAT_MARKER in row)


def test_finalize_answer_does_not_duplicate_ambiguous_marker():
    existing = f"{AMBIGUOUS_SUPPORT_CAVEAT_MARKER}: already noted"
    answer = {"selected_option_id": "A", "caveats": [existing], "supporting_evidence_ids": ["E1"]}
    per_requirement_claim = {
        "q1::option_a": {"option_status": "supported"},
        "q1::option_b": {"option_status": "supported"},
    }
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["caveats"] == [existing]


def test_finalize_answer_forces_marker_when_unresolved():
    answer = {"selected_option_id": "A", "caveats": []}
    per_requirement_claim = {"q1::option_a": {"option_status": "unresolved"}}
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["final_status"] == "budget_exhausted_guess"
    assert any(BUDGET_EXHAUSTED_CAVEAT_MARKER in row for row in result["caveats"])


def test_finalize_answer_does_not_duplicate_existing_marker():
    existing = f"{BUDGET_EXHAUSTED_CAVEAT_MARKER}: already noted"
    answer = {"selected_option_id": "A", "caveats": [existing]}
    per_requirement_claim = {"q1::option_a": {"option_status": "unresolved"}}
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["caveats"] == [existing]


def test_finalize_answer_rejects_refuted_selection_when_better_exists():
    answer = {"selected_option_id": "A", "caveats": []}
    per_requirement_claim = {
        "q1::option_a": {"option_status": "refuted"},
        "q1::option_b": {"option_status": "supported"},
    }
    try:
        _finalize_answer(answer, QUESTION, per_requirement_claim)
    except ValueError:
        pass
    else:
        raise AssertionError("selecting a refuted option while a supported one exists passed validation")


def test_finalize_answer_allows_refuted_selection_when_nothing_better_exists():
    # Still a forced guess (HourVideo is multiple-choice, an answer must be submitted either way),
    # but should not raise just because every option ended up refuted or unresolved.
    answer = {"selected_option_id": "A", "caveats": []}
    per_requirement_claim = {
        "q1::option_a": {"option_status": "refuted"},
        "q1::option_b": {"option_status": "unresolved"},
    }
    result = _finalize_answer(answer, QUESTION, per_requirement_claim)
    assert result["final_status"] == "budget_exhausted_guess"


FINE_IDS = ["F1", "F2"]
BATCH_RIDS = ["q1::option_a", "q1::option_b"]


def _batch_execution(status_by_rid=None, supporting_by_rid=None):
    status_by_rid = status_by_rid or {rid: "confirmed" for rid in BATCH_RIDS}
    supporting_by_rid = supporting_by_rid or {rid: ["F1"] for rid in BATCH_RIDS}
    return {
        "observations": [
            {"fine_id": fid, "finding": "kitchen visible", "visible_actions": [], "visible_objects": [], "uncertainty_notes": []}
            for fid in FINE_IDS
        ],
        "claim_assessments": {
            rid: {"claim_status": status_by_rid[rid], "supporting_fine_ids": supporting_by_rid[rid], "rationale": "matches the claim"}
            for rid in BATCH_RIDS
        },
    }


def test_batch_claim_execution_schema_has_no_final_answer_field():
    text = repr(_batch_claim_execution_schema(FINE_IDS, BATCH_RIDS))
    assert "claim_status" in text
    assert "selected_option_id" not in text
    assert "answer" not in text


def test_validate_batch_claim_execution_accepts_well_formed_result():
    _validate_batch_claim_execution(_batch_execution(), FINE_IDS, BATCH_RIDS)


def test_validate_batch_claim_execution_rejects_fine_coverage_mismatch():
    value = _batch_execution()
    value["observations"] = value["observations"][:1]
    try:
        _validate_batch_claim_execution(value, FINE_IDS, BATCH_RIDS)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete Fine coverage passed validation")


def test_validate_batch_claim_execution_confirmed_requires_supporting_fine():
    value = _batch_execution(supporting_by_rid={"q1::option_a": [], "q1::option_b": ["F1"]})
    try:
        _validate_batch_claim_execution(value, FINE_IDS, BATCH_RIDS)
    except ValueError:
        pass
    else:
        raise AssertionError("confirmed with no supporting Fine passed validation")


def test_validate_batch_claim_execution_rejects_requirement_coverage_mismatch():
    value = _batch_execution()
    del value["claim_assessments"]["q1::option_b"]
    try:
        _validate_batch_claim_execution(value, FINE_IDS, BATCH_RIDS)
    except ValueError:
        pass
    else:
        raise AssertionError("missing requirement assessment passed validation")


def test_dedupe_fine_rows_merges_shared_image_and_tracks_both_requirements():
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    image = tmp_dir / "frame.jpg"
    image.write_bytes(b"same-bytes")
    shared = {"fine_id": "F1", "timestamp_sec": 1.0, "source_frame_path": str(image)}
    only_b = {"fine_id": "F2", "timestamp_sec": 2.0, "source_frame_path": str(image)}
    fine_rows_by_rid = {
        "q1::option_a": [shared],
        "q1::option_b": [dict(shared, fine_id="F1"), only_b],
    }
    unique, targets = _dedupe_fine_rows(fine_rows_by_rid)
    # F1 is the identical image content under both requirements -> one entry, both requirements listed.
    assert len(unique) == 1
    digest = unique[0]["image_sha256"]
    assert set(targets[digest]) == {"q1::option_a", "q1::option_b"}


def test_write_run_manifest_records_config_and_core_module_hash():
    import json
    import tempfile

    root = Path(tempfile.mkdtemp())
    config_path = root / "config.json"
    config_path.write_text(json.dumps({"max_claim_rounds": 3, "review_contract_version": "test"}), encoding="utf-8")
    cfg = {"output_root": "outputs/experiments/test_run", "max_claim_rounds": 3, "review_contract_version": "test"}

    manifest = _write_run_manifest(root, cfg, config_path, video_uid="testuid001")

    assert manifest["video_uid_filter"] == "testuid001"
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["core_module_sha256"]) == 64
    assert (root / cfg["output_root"] / "run_manifest_latest.json").is_file()


def test_claim_text_contradicts_status_catches_the_real_reversal_seen_on_819c8af7():
    text = "Option D states 'Using a wrench on the switchboard' ... Option D is supported, not refuted."
    assert _claim_text_contradicts_status("refuted", text) is True
    assert _claim_text_contradicts_status("supported", text) is False


def test_claim_text_contradicts_status_symmetric_case():
    text = "The evidence shows this option is refuted; it is incorrect."
    assert _claim_text_contradicts_status("supported", text) is True
    assert _claim_text_contradicts_status("refuted", text) is False


def test_claim_text_contradicts_status_consistent_text_passes():
    assert _claim_text_contradicts_status("refuted", "The evidence directly contradicts this option.") is False
    assert _claim_text_contradicts_status("supported", "The evidence directly confirms this option.") is False


def _fine_row(fine_id, medium_id, medium_rank):
    return {"fine_id": fine_id, "medium_id": medium_id, "timestamp_sec": float(medium_rank), "medium_rank": medium_rank}


def test_allocate_fine_quota_by_coarse_guarantees_every_region_gets_at_least_one():
    # 3 Coarse regions, cap=3: C_LOW's only candidate must survive even though C_HIGH has enough
    # high-ranked candidates to fill the whole cap on score alone -- this is exactly the coverage-loss
    # class of bug Coarse-locking was built to fix at the Medium level, now guarded at the Fine level.
    parent = {"M1": "C_HIGH", "M2": "C_HIGH", "M3": "C_HIGH", "M4": "C_HIGH", "M5": "C_LOW", "M6": "C_MID"}
    fine_rows = [
        _fine_row("F1", "M1", 1), _fine_row("F2", "M2", 2), _fine_row("F3", "M3", 3), _fine_row("F4", "M4", 4),
        _fine_row("F5", "M5", 5), _fine_row("F6", "M6", 6),
    ]
    result = _allocate_fine_quota_by_coarse(fine_rows, parent, cap=3)
    regions = {parent[row["medium_id"]] for row in result}
    assert regions == {"C_HIGH", "C_LOW", "C_MID"}
    assert len(result) == 3


def test_allocate_fine_quota_by_coarse_spends_leftover_on_highest_rank():
    parent = {"M1": "C_A", "M2": "C_A", "M3": "C_B"}
    fine_rows = [_fine_row("F1", "M1", 1), _fine_row("F2", "M2", 2), _fine_row("F3", "M3", 3)]
    result = _allocate_fine_quota_by_coarse(fine_rows, parent, cap=2)
    ids = {row["fine_id"] for row in result}
    assert "F3" in ids  # C_B's one candidate must be included (base quota)
    assert "F1" in ids  # the higher-ranked of C_A's two candidates gets the leftover slot, not F2


def test_allocate_fine_quota_by_coarse_no_op_under_cap():
    parent = {"M1": "C_A"}
    fine_rows = [_fine_row("F1", "M1", 1)]
    result = _allocate_fine_quota_by_coarse(fine_rows, parent, cap=20)
    assert result == fine_rows


def _sha_row(fine_id, sha, medium_rank):
    return {"fine_id": fine_id, "image_sha256": sha, "timestamp_sec": float(medium_rank), "medium_rank": medium_rank}


def test_allocate_fine_quota_by_requirement_caps_a_batch_within_each_ones_own_limit():
    # Real traffic: 4 requirements each within their own 20-image cap still deduped to 29 unique
    # images in one batch, and that batch got HTTP 400. The per-claim cap alone does not bound the
    # batch total once cross-option dedup merges several requirements' pools together.
    unique_fines = [_sha_row(f"F{i}", f"sha{i}", i) for i in range(29)]
    targets = {f"sha{i}": [f"option_{i % 4}"] for i in range(29)}
    result = _allocate_fine_quota_by_requirement(unique_fines, targets, cap=16)
    assert len(result) <= 16
    served = {targets[row["image_sha256"]][0] for row in result}
    assert served == {"option_0", "option_1", "option_2", "option_3"}


def test_allocate_fine_quota_by_requirement_no_op_under_cap():
    unique_fines = [_sha_row("F1", "sha1", 1)]
    targets = {"sha1": ["option_a"]}
    result = _allocate_fine_quota_by_requirement(unique_fines, targets, cap=16)
    assert result == unique_fines


# ============================================================================
# Discriminator checklist (task 37, part A): _discriminator_schema /
# _validate_discriminators, _call_discriminator_extraction's retry/graceful-
# degradation contract, and the discriminator_findings -> verified_facts
# integration path (_discriminator_findings_as_facts + _verify_decisive_facts'
# force_decisive_texts). No live API calls anywhere in this section.
# ============================================================================

DISCRIMINATORS = [
    {"text": "Does the sequence include dampening the wall surface?", "option_ids_with_unique_clause": ["A", "B"]},
]


def test_discriminator_schema_constrains_distinguishes_to_known_option_ids():
    schema = _discriminator_schema(_requirements())
    item = schema["properties"]["discriminators"]["items"]
    assert item["properties"]["option_ids_with_unique_clause"]["items"]["enum"] == ["A", "B"]


def test_validate_discriminators_accepts_well_formed():
    value = {"question_id": QUESTION["question_id"], "discriminators": DISCRIMINATORS}
    _validate_discriminators(value, QUESTION, _requirements())


def test_validate_discriminators_rejects_question_mismatch():
    value = {"question_id": "wrong", "discriminators": DISCRIMINATORS}
    try:
        _validate_discriminators(value, QUESTION, _requirements())
    except ValueError:
        pass
    else:
        raise AssertionError("question_id mismatch passed validation")


def test_validate_discriminators_rejects_empty_list():
    value = {"question_id": QUESTION["question_id"], "discriminators": []}
    try:
        _validate_discriminators(value, QUESTION, _requirements())
    except ValueError:
        pass
    else:
        raise AssertionError("empty discriminators list passed validation")


def test_validate_discriminators_rejects_blank_text():
    value = {"question_id": QUESTION["question_id"], "discriminators": [{"text": "   ", "option_ids_with_unique_clause": ["A", "B"]}]}
    try:
        _validate_discriminators(value, QUESTION, _requirements())
    except ValueError:
        pass
    else:
        raise AssertionError("blank discriminator text passed validation")


def test_validate_discriminators_rejects_unknown_option_id():
    value = {"question_id": QUESTION["question_id"], "discriminators": [{"text": "does X happen?", "option_ids_with_unique_clause": ["A", "Z"]}]}
    try:
        _validate_discriminators(value, QUESTION, _requirements())
    except ValueError:
        pass
    else:
        raise AssertionError("unknown option_id passed validation")


def test_validate_discriminators_accepts_single_option_unique_clause():
    # Real live-call finding (2026-08-09, 06638e64): a discriminator whose detail belongs to exactly
    # one option ("does dampen the wall surface?" -> only option B mentions it) is the common, correct
    # shape -- requiring >= 2 ids used to reject this, which rejected 5 of 8 real discriminators a live
    # call correctly extracted, including this exact wall-dampening one. See DISCRIMINATOR_SYSTEM.
    value = {"question_id": QUESTION["question_id"], "discriminators": [{"text": "does X happen?", "option_ids_with_unique_clause": ["A"]}]}
    _validate_discriminators(value, QUESTION, _requirements())


def test_validate_discriminators_rejects_empty_option_id_list():
    # The one genuinely meaningless case: a discriminator that marks no option at all.
    value = {"question_id": QUESTION["question_id"], "discriminators": [{"text": "does X happen?", "option_ids_with_unique_clause": []}]}
    try:
        _validate_discriminators(value, QUESTION, _requirements())
    except ValueError:
        pass
    else:
        raise AssertionError("empty option_ids_with_unique_clause list passed validation")


def test_call_discriminator_extraction_retries_past_a_transient_error():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    requirements = _requirements()
    good_provider = {"question_id": QUESTION["question_id"], "discriminators": DISCRIMINATORS}
    calls = {"n": 0}

    def fake_anthropic_call(cfg, system, payload, schema, max_tokens):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient provider error")
        return good_provider, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fake_anthropic_call
    try:
        discriminators, discriminator_calls = v6_core._call_discriminator_extraction(
            {"anthropic": {"claim_max_tokens": 100}, "max_validation_retries": 2}, QUESTION, requirements,
        )
    finally:
        v6_core._anthropic_call = original
    assert calls["n"] == 2
    assert discriminators == DISCRIMINATORS
    assert len(discriminator_calls) == 1  # the first, RuntimeError attempt logs no usage


def test_call_discriminator_extraction_degrades_to_empty_list_when_every_attempt_fails_validation():
    # Deliberately different contract from the planner: a discriminator-extraction failure must never
    # raise and block the whole question, only degrade to "no discriminators" (today's pre-V6.2
    # behavior) -- see the "Strictly additive" comment above _call_discriminator_extraction.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    requirements = _requirements()
    bad_provider = {"question_id": "wrong_question_id", "discriminators": DISCRIMINATORS}

    def fake_anthropic_call(cfg, system, payload, schema, max_tokens):
        return bad_provider, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fake_anthropic_call
    try:
        discriminators, discriminator_calls = v6_core._call_discriminator_extraction(
            {"anthropic": {"claim_max_tokens": 100}, "max_validation_retries": 2}, QUESTION, requirements,
        )
    finally:
        v6_core._anthropic_call = original
    assert discriminators == []
    assert len(discriminator_calls) == 2


def test_call_discriminator_extraction_degrades_to_empty_list_when_every_attempt_raises():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    requirements = _requirements()

    def always_fails(cfg, system, payload, schema, max_tokens):
        raise RuntimeError("provider unavailable")

    original = v6_core._anthropic_call
    v6_core._anthropic_call = always_fails
    try:
        discriminators, discriminator_calls = v6_core._call_discriminator_extraction(
            {"anthropic": {"claim_max_tokens": 100}, "max_validation_retries": 2}, QUESTION, requirements,
        )
    finally:
        v6_core._anthropic_call = original
    assert discriminators == []
    assert discriminator_calls == []


def test_call_discriminator_extraction_dedupes_option_ids_with_unique_clause():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    requirements = _requirements()
    dupe_provider = {
        "question_id": QUESTION["question_id"],
        "discriminators": [{"text": "does X happen?", "option_ids_with_unique_clause": ["A", "A", "B"]}],
    }

    def fake_anthropic_call(cfg, system, payload, schema, max_tokens):
        return dupe_provider, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._anthropic_call
    v6_core._anthropic_call = fake_anthropic_call
    try:
        discriminators, _ = v6_core._call_discriminator_extraction(
            {"anthropic": {"claim_max_tokens": 100}, "max_validation_retries": 2}, QUESTION, requirements,
        )
    finally:
        v6_core._anthropic_call = original
    assert discriminators[0]["option_ids_with_unique_clause"] == ["A", "B"]


def test_shared_investigation_schema_discriminator_field_enum_matches_supplied_texts():
    schema = _shared_investigation_schema(QUESTION, EVIDENCE, [], DISCRIMINATORS)
    item = schema["properties"]["discriminator_findings"]["items"]
    assert item["properties"]["discriminator"]["enum"] == [DISCRIMINATORS[0]["text"]]


def test_shared_investigation_schema_discriminator_field_unconstrained_when_none_supplied():
    schema = _shared_investigation_schema(QUESTION, EVIDENCE, [])
    item = schema["properties"]["discriminator_findings"]["items"]
    assert item["properties"]["discriminator"] == {"type": "string"}


def _finding(discriminator_text, finding="the wall was not dampened", evidence_ids=("E1",)):
    return {"discriminator": discriminator_text, "finding": finding, "evidence_ids": list(evidence_ids)}


def test_validate_shared_investigation_accepts_full_discriminator_coverage():
    value = _investigation({"discriminator_findings": [_finding(DISCRIMINATORS[0]["text"])]})
    _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)


def test_validate_shared_investigation_discriminator_coverage_holds_even_while_unresolved():
    # Discriminator findings are usable independent of investigation_status -- see the module comment
    # above SHARED_INVESTIGATION_SYSTEM's discriminator_findings instructions.
    value = _investigation({
        "investigation_status": "unresolved", "gap_reason": "not enough to resolve the whole question",
        "discriminator_findings": [_finding(DISCRIMINATORS[0]["text"])],
    })
    _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)


def test_validate_shared_investigation_rejects_missing_discriminator_coverage():
    value = _investigation({"discriminator_findings": []})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)
    except ValueError:
        pass
    else:
        raise AssertionError("missing discriminator coverage passed validation")


def test_validate_shared_investigation_rejects_discriminator_finding_for_unknown_text():
    value = _investigation({"discriminator_findings": [_finding("a discriminator nobody asked for")]})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)
    except ValueError:
        pass
    else:
        raise AssertionError("finding for an unsupplied discriminator passed validation")


def test_validate_shared_investigation_rejects_blank_finding_text():
    value = _investigation({"discriminator_findings": [_finding(DISCRIMINATORS[0]["text"], finding="  ")]})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)
    except ValueError:
        pass
    else:
        raise AssertionError("blank finding text passed validation")


def test_validate_shared_investigation_rejects_finding_citing_unknown_evidence():
    value = _investigation({"discriminator_findings": [_finding(DISCRIMINATORS[0]["text"], evidence_ids=("E99",))]})
    try:
        _validate_shared_investigation(value, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)
    except ValueError:
        pass
    else:
        raise AssertionError("finding citing unknown evidence passed validation")


def test_downgrade_noncompliant_investigation_fills_missing_discriminator_coverage():
    value = _investigation({"discriminator_findings": []})
    downgraded = _downgrade_noncompliant_investigation(value, EVIDENCE, DISCRIMINATORS)
    assert len(downgraded["discriminator_findings"]) == 1
    assert downgraded["discriminator_findings"][0]["discriminator"] == DISCRIMINATORS[0]["text"]
    assert downgraded["discriminator_findings"][0]["evidence_ids"] == []
    _validate_shared_investigation(downgraded, QUESTION, EVIDENCE, EXCLUDED_COARSE_IDS, DISCRIMINATORS)


def test_downgrade_noncompliant_investigation_preserves_existing_discriminator_finding():
    value = _investigation({"discriminator_findings": [_finding(DISCRIMINATORS[0]["text"], finding="real finding")]})
    downgraded = _downgrade_noncompliant_investigation(value, EVIDENCE, DISCRIMINATORS)
    assert downgraded["discriminator_findings"][0]["finding"] == "real finding"


def test_discriminator_findings_as_facts_formats_text_and_keeps_evidence():
    findings = [_finding(DISCRIMINATORS[0]["text"], finding="water went into the mixing bucket, not the wall")]
    facts, decisive_texts = _discriminator_findings_as_facts(findings)
    assert facts == [{
        "text": f"{DISCRIMINATORS[0]['text']}: water went into the mixing bucket, not the wall",
        "evidence_ids": ["E1"],
    }]
    assert decisive_texts == frozenset({facts[0]["text"]})


def test_discriminator_findings_as_facts_drops_placeholder_findings_with_no_evidence():
    findings = [_finding(
        DISCRIMINATORS[0]["text"], finding="Insufficient evidence to address this discriminator.", evidence_ids=(),
    )]
    facts, decisive_texts = _discriminator_findings_as_facts(findings)
    assert facts == []
    assert decisive_texts == frozenset()


def test_verify_decisive_facts_forces_a_discriminator_fact_through_verification():
    # A discriminator-derived fact is decisive by construction (extracted specifically because it
    # distinguishes options) -- _is_decisive_fact's word-overlap-with-option-text heuristic would not
    # necessarily catch it, so force_decisive_texts must route it through the lexical-then-API cascade
    # regardless of what the heuristic alone would conclude.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    evidence = [{"evidence_id": "E1", "source_content": "spraying water into the mixing bucket"}]
    fact_text = "water went into the mixing bucket, not the wall"
    facts = [{"text": fact_text, "evidence_ids": ["E1"]}]
    requirements = _requirements()  # option text is "alpha"/"beta" -- shares no vocabulary with this fact

    calls = {"n": 0}

    def fake_call_fact_verification(cfg, text, cited_text):
        calls["n"] += 1
        return {"entailed": False, "reason": "cited text describes the bucket, not the wall"}, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._call_fact_verification
    v6_core._call_fact_verification = fake_call_fact_verification
    try:
        results, verification_calls = v6_core._verify_decisive_facts(
            {}, facts, requirements, evidence, force_decisive_texts=frozenset({fact_text}),
        )
    finally:
        v6_core._call_fact_verification = original

    assert calls["n"] == 1
    assert len(verification_calls) == 1
    assert results[0]["checked"] is True
    assert results[0]["verified"] is False


def test_verify_decisive_facts_skips_the_same_fact_without_force_decisive_texts():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    evidence = [{"evidence_id": "E1", "source_content": "spraying water into the mixing bucket"}]
    fact_text = "water went into the mixing bucket, not the wall"
    facts = [{"text": fact_text, "evidence_ids": ["E1"]}]
    requirements = _requirements()

    calls = {"n": 0}

    def fake_call_fact_verification(cfg, text, cited_text):
        calls["n"] += 1
        return {"entailed": True, "reason": ""}, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._call_fact_verification
    v6_core._call_fact_verification = fake_call_fact_verification
    try:
        results, verification_calls = v6_core._verify_decisive_facts({}, facts, requirements, evidence)
    finally:
        v6_core._call_fact_verification = original

    assert calls["n"] == 0
    assert verification_calls == []
    assert results[0]["checked"] is False
    assert results[0]["verified"] is True


def test_discriminator_findings_flow_into_verified_facts_without_api():
    # Composes _discriminator_findings_as_facts -> _verify_decisive_facts exactly as _resolve_requirements
    # does (core.py:1319-1331), with a stubbed fact-verification call standing in for the one real API
    # round-trip -- this is the integration path task 37 part A asks to cover, without needing the full
    # hierarchy/embeddings fixtures a real _resolve_requirements call would require.
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    evidence = [{"evidence_id": "E1", "source_content": "spraying water into the mixing bucket"}]
    findings = [_finding(
        DISCRIMINATORS[0]["text"], finding="water went into the mixing bucket, not the wall", evidence_ids=("E1",),
    )]
    discriminator_facts, force_decisive_texts = v6_core._discriminator_findings_as_facts(findings)
    requirements = _requirements()

    def fake_call_fact_verification(cfg, text, cited_text):
        return {"entailed": True, "reason": "matches"}, {"provider": "anthropic", "output_tokens": 1}

    original = v6_core._call_fact_verification
    v6_core._call_fact_verification = fake_call_fact_verification
    try:
        fact_verification, calls = v6_core._verify_decisive_facts(
            {}, discriminator_facts, requirements, evidence, force_decisive_texts,
        )
    finally:
        v6_core._call_fact_verification = original

    verified_texts = {row["text"] for row in fact_verification if row["verified"]}
    assert verified_texts == {discriminator_facts[0]["text"]}
    assert len(calls) == 1


# ============================================================================
# Discriminator checklist (task 37, part B): zero-API replay against the real
# 06638e64-21ea-4065-8d8e-919b0aaf4538_5_12 evidence that motivated the
# discriminator checklist in the first place (see the comment above
# DISCRIMINATOR_SYSTEM). Evidence text (C05's navigation_summary, M012/M013
# captions) and the discriminator wording are copied verbatim from
# outputs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1/
# cases/06638e64-21ea-4065-8d8e-919b0aaf4538/r3_2/shared_investigation.
# pre_fix123.json -- the real run where established_facts stated "the wall
# was not dampened before plaster application based on the available
# evidence" and then, in the same paragraph, still said the sequence
# "matches option B most closely" (option B's only difference from gold
# option A is that it adds wall-dampening). Every LLM call is stubbed here;
# this only exercises the plumbing between them, not judgment quality
# (that's task 37 part C, the live paid check).
# ============================================================================

REAL_06638E64_QUESTION = {
    "question_id": "06638e64-21ea-4065-8d8e-919b0aaf4538_5_12",
    "question_text": "What steps did the camera wearer take before applying quick setting plaster cement to the wall?",
    "answer_options": [
        {"option_id": "A", "text": "Organize the work area, adjust buckets, pick up various tools and materials including a tile, scrapers, and tape, prepare quick setting plaster cement for application"},
        {"option_id": "B", "text": "Organize the work area, adjust buckets, pick up various tools and materials including a tile, scrapers, and tape, prepare quick setting plaster cement for application, and dampen the wall surface"},
        {"option_id": "C", "text": "Organize the work area, adjust buckets, pick up various tools and materials including a tile, scrapers, and mesh, mix quick setting plaster cement for application, and inspect the wall for cracks"},
        {"option_id": "D", "text": "Organize the work area, adjust containers, pick up various tools and materials including a tile, spatulas, and adhesive, prepare quick setting plaster cement for application"},
        {"option_id": "E", "text": "Organize the work area, adjust buckets, pick up various tools and materials including a tile and tape, prepare quick setting plaster cement for application"},
    ],
}
# Real gold answer is A. This is the exact worked example in DISCRIMINATOR_SYSTEM's own docstring.
REAL_06638E64_DISCRIMINATOR = {
    "text": "Does the sequence include dampening the wall surface?", "option_ids_with_unique_clause": ["A", "B"],
}
REAL_06638E64_MAP_DOC = {
    "coarse_regions": [{
        "coarse_id": "C05",
        "navigation_summary": "Plaster application phase - mixing cement/plaster in bucket and applying to wall with trowel",
        "start_sec": 495.0, "end_sec": 585.0, "source_medium_ids": ["M012", "M013"],
    }],
}
REAL_06638E64_HIERARCHY = {
    "medium_nodes": [
        {"medium_id": "M012", "start_sec": 495.0, "end_sec": 540.0, "source_fine_ids": []},
        {"medium_id": "M013", "start_sec": 540.0, "end_sec": 585.0, "source_fine_ids": []},
    ],
}
REAL_06638E64_CAPTIONS = [
    {"qwen_caption": "Ego opens a van door, grabs a bag of cement, and mixes it with water in a bucket."},
    {"qwen_caption": "Ego is mixing plaster in a bucket using a tool while standing in a room with a window."},
]


def _real_06638e64_plan():
    requirements = option_requirements(REAL_06638E64_QUESTION)

    def unit():
        return {
            "search_description": "steps taken before applying quick setting plaster cement",
            "query_variants": ["preparation before applying plaster"], "modality_strategy": "visual",
            "coarse_judgments": [{"coarse_id": "C05", "selected": True, "reason": "the plaster-application phase"}],
        }

    return {
        "question_id": REAL_06638E64_QUESTION["question_id"],
        "requirement_plans": [{"requirement_id": row["requirement_id"], **unit()} for row in requirements],
    }


def test_discriminator_checklist_replay_against_real_06638e64_evidence_without_api():
    from experiments.hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_2 import core as v6_core

    question = REAL_06638E64_QUESTION
    requirements = option_requirements(question)
    plan = _real_06638e64_plan()
    map_doc = REAL_06638E64_MAP_DOC
    hierarchy = REAL_06638E64_HIERARCHY
    parent = v6_core._parent_map(map_doc)

    # The real cited evidence text, reproduced exactly -- none of it mentions wall-dampening, which is
    # exactly the gap the real buggy run papered over.
    real_evidence = v6_core._map_level_evidence(
        "r3_2", f"{question['question_id']}::shared", {"C05"}, map_doc, hierarchy, [], REAL_06638E64_CAPTIONS, parent,
    )
    real_evidence_ids = [row["evidence_id"] for row in real_evidence]
    assert len(real_evidence_ids) == 3
    assert all("dampen" not in row["source_content"].lower() for row in real_evidence)

    discriminator_fact_finding = (
        "The evidence describes mixing cement/plaster in a bucket and applying it with a trowel; "
        "it does not describe dampening the wall."
    )
    investigation_calls = {"n": 0}

    def fake_shared_investigation(cfg, question_arg, evidence_arg, excluded_judgments, round_number, discriminators=()):
        investigation_calls["n"] += 1
        assert round_number == 0
        assert {row["evidence_id"] for row in evidence_arg} == set(real_evidence_ids)
        return {
            "question_id": question_arg["question_id"], "investigation_status": "resolved",
            "established_facts": (
                "The camera wearer mixed cement and plaster in a bucket and applied it with a trowel; "
                "no wall-dampening step is described in the available evidence."
            ),
            "atomic_facts": [{
                "text": "The camera wearer mixed cement and plaster in a bucket before applying it with a trowel.",
                "evidence_ids": real_evidence_ids[1:],
            }],
            "discriminator_findings": [{
                "discriminator": REAL_06638E64_DISCRIMINATOR["text"],
                "finding": discriminator_fact_finding, "evidence_ids": real_evidence_ids,
            }],
            "gap_reason": "", "requested_coarse_ids": [], "cited_evidence_ids": real_evidence_ids,
        }, {"provider": "anthropic", "output_tokens": 1}

    verification_calls = []

    def fake_call_fact_verification(cfg, text, cited_text):
        verification_calls.append(text)
        return {"entailed": True, "reason": "matches the cited captions"}, {"provider": "anthropic", "output_tokens": 1}

    mapping_calls = []

    def fake_call_option_mapping(cfg, question_arg, requirements_arg, verified_facts):
        mapping_calls.append(verified_facts)
        # A real option-mapping call would decide B here; this stub only needs to prove the plumbing
        # delivered the discriminator's honest finding as one of the facts it had to work with.
        return {"question_id": question_arg["question_id"], "option_assessments": [
            {"requirement_id": row["requirement_id"], "status": "unresolved", "used_facts": [], "reason": "stub"}
            for row in requirements_arg
        ]}, {"provider": "anthropic", "output_tokens": 1}

    original_investigation = v6_core._call_shared_investigation
    original_verification = v6_core._call_fact_verification
    original_mapping = v6_core._call_option_mapping
    v6_core._call_shared_investigation = fake_shared_investigation
    v6_core._call_fact_verification = fake_call_fact_verification
    v6_core._call_option_mapping = fake_call_option_mapping
    try:
        claims, evidence_by_rid, calls, investigation, fact_verification = v6_core._resolve_requirements(
            cfg={}, question=question, requirements=requirements, plan=plan, side="r3_2", map_doc=map_doc,
            hierarchy=hierarchy, projections=[], captions=REAL_06638E64_CAPTIONS, medium_embeddings=None,
            fine_by_id={}, fine_embeddings=None, row_by_id={}, encoder=None, parent=parent, max_rounds=3,
            discriminators=[REAL_06638E64_DISCRIMINATOR],
        )
    finally:
        v6_core._call_shared_investigation = original_investigation
        v6_core._call_fact_verification = original_verification
        v6_core._call_option_mapping = original_mapping

    # 1. Resolved on round 0 -- no retrieval, no image transmission was ever needed.
    assert investigation_calls["n"] == 1
    assert investigation["investigation_status"] == "resolved"

    # 2. skip-heuristic bypassed: the discriminator finding is decisive by construction and went
    #    through fact verification even though _is_decisive_fact's own word-overlap heuristic has
    #    nothing in this fact's wording that matches any option's text.
    discriminator_fact_text = f"{REAL_06638E64_DISCRIMINATOR['text']}: {discriminator_fact_finding}"
    assert discriminator_fact_text in verification_calls

    # 3. verification-gate routing: it was actually checked (not silently skipped) and passed.
    assert any(row["text"] == discriminator_fact_text and row["checked"] and row["verified"] for row in fact_verification)

    # 4. merge-into-verified_facts: option mapping was handed this fact, with its real evidence
    #    citations, not just the (much vaguer) atomic_facts breakdown.
    assert len(mapping_calls) == 1
    verified_texts = {f["text"] for f in mapping_calls[0]}
    assert discriminator_fact_text in verified_texts
    discriminator_entry = next(f for f in mapping_calls[0] if f["text"] == discriminator_fact_text)
    assert discriminator_entry["evidence_ids"] == real_evidence_ids

    # 5. the shared result was fanned back out to every one of the 5 real answer options.
    assert set(claims.keys()) == {row["requirement_id"] for row in requirements}


# ============================================================================
# Discriminator checklist (task 37, part C follow-up, 2026-08-09): a real,
# paid Anthropic call against REAL_06638E64_QUESTION returned 8 discriminators
# where 5 of 8 named only one option each (e.g. the wall-dampening one this
# whole mechanism was built for named only "B") -- the >= 2 rule then in
# effect rejected all 5, and both retry attempts failed the same way, so
# _call_discriminator_extraction degraded to []. That live call's raw
# provider response is captured verbatim below and replayed against the
# fixed validator with no further API spend: option_ids_with_unique_clause
# now means "this option has a clause none of the others do" for a single
# id, so this exact real response must now pass in full.
# ============================================================================

REAL_LIVE_06638E64_DISCRIMINATOR_RESPONSE = {
    "question_id": "06638e64-21ea-4065-8d8e-919b0aaf4538_5_12",
    "discriminators": [
        {"text": "Does the camera wearer dampen the wall surface before applying the plaster cement?", "option_ids_with_unique_clause": ["B"]},
        {"text": "Does the camera wearer pick up scrapers as part of the tool preparation?", "option_ids_with_unique_clause": ["A", "B", "C"]},
        {"text": "Does the camera wearer pick up mesh as part of the materials?", "option_ids_with_unique_clause": ["C"]},
        {"text": "Does the camera wearer inspect the wall for cracks?", "option_ids_with_unique_clause": ["C"]},
        {"text": "Does the camera wearer pick up spatulas rather than scrapers?", "option_ids_with_unique_clause": ["D"]},
        {"text": "Does the camera wearer pick up adhesive as part of the materials?", "option_ids_with_unique_clause": ["D"]},
        {"text": "Does the camera wearer adjust containers rather than buckets?", "option_ids_with_unique_clause": ["D"]},
        {"text": "Does the camera wearer pick up scrapers as a tool?", "option_ids_with_unique_clause": ["A", "B", "C"]},
    ],
}


def test_real_live_06638e64_discriminator_response_now_validates():
    _validate_discriminators(REAL_LIVE_06638E64_DISCRIMINATOR_RESPONSE, REAL_06638E64_QUESTION, option_requirements(REAL_06638E64_QUESTION))


def test_real_live_06638e64_response_would_have_been_rejected_under_the_old_rule():
    # Documents the regression this fix closes: under the old "distinguishes at least two options"
    # rule, 6 of these 8 real discriminators -- including the wall-dampening one -- were invalid.
    single_option_count = sum(
        1 for row in REAL_LIVE_06638E64_DISCRIMINATOR_RESPONSE["discriminators"]
        if len(row["option_ids_with_unique_clause"]) < 2
    )
    assert single_option_count == 6
    wall_dampening = next(
        row for row in REAL_LIVE_06638E64_DISCRIMINATOR_RESPONSE["discriminators"] if "dampen" in row["text"].lower()
    )
    assert wall_dampening["option_ids_with_unique_clause"] == ["B"]
