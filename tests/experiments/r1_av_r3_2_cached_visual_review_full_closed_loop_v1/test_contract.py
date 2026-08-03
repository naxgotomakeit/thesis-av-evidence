from experiments.r1_av_r3_2_cached_visual_review_full_closed_loop_v1.core import normalize_display_text, validate_packets


def test_display_normalization_is_ascii_and_semantics_preserving():
    value, count = normalize_display_text("597\u2013605 and 613\u2011618")
    assert value == "597-605 and 613-618"
    assert count == 2


def test_packet_validation_preserves_requirement_order():
    scoped = {
        side: {"q": {"in_scope_requirement_assessments": [{"requirement_id": "r"}]}}
        for side in ("r1_av", "r3_2")
    }
    updated = {
        side: {"q": {"requirement_assessments": [{"requirement_id": "r"}]}}
        for side in ("r1_av", "r3_2")
    }
    packets = {
        side: [{"question_id": "q", "requirement_assessments": [{"requirement_id": "r"}]}]
        for side in ("r1_av", "r3_2")
    }
    # The production order is deliberately patched for this minimal unit fixture.
    import experiments.r1_av_r3_2_cached_visual_review_full_closed_loop_v1.core as core
    old = core.QUESTION_ORDER
    core.QUESTION_ORDER = ["q"]
    try:
        assert validate_packets(scoped, updated, packets) == []
        packets["r1_av"][0]["requirement_assessments"] = []
        assert validate_packets(scoped, updated, packets)
    finally:
        core.QUESTION_ORDER = old
