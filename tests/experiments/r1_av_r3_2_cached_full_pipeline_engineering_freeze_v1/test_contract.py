def test_freeze_scope_is_engineering_only():
    frozen = {
        "freeze_kind": "engineering_test_snapshot",
        "manual_answer_correctness": "not_frozen_user_elected_non_blocking",
        "hourvideo_performance": "not_tested",
    }
    assert frozen["freeze_kind"] == "engineering_test_snapshot"
    assert frozen["manual_answer_correctness"].startswith("not_frozen")
    assert frozen["hourvideo_performance"] == "not_tested"

