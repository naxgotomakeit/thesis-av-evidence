from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.live import option_requirements, review_scope


def question():
    return {"question_id": "video_1", "question_text": "What happened?", "answer_options": [{"option_id": "A", "text": "One"}, {"option_id": "B", "text": "Two"}]}


def test_one_requirement_per_option_in_order():
    rows = option_requirements(question())
    assert [row["requirement_id"] for row in rows] == ["video_1::option_a", "video_1::option_b"]
    assert [row["option_id"] for row in rows] == ["A", "B"]


def test_review_scope_is_question_and_option_specific():
    assert review_scope("video_1", "A") != review_scope("video_2", "A")
    assert review_scope("video_1", "A") != review_scope("video_1", "B")


def test_requirements_do_not_contain_gold_or_reference_fields():
    text = repr(option_requirements(question())).lower()
    assert "gold" not in text
    assert "reference_timestamp" not in text
