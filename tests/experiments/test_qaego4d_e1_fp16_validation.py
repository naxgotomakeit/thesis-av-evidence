from src.experiments.qaego4d_e1_fp16_validation.runner import _normalise_text


def test_normalised_open_text_ignores_case_punctuation_and_spacing() -> None:
    assert _normalise_text("A spoon.") == _normalise_text("  a spoon ")


def test_normalised_open_text_preserves_word_changes() -> None:
    assert _normalise_text("on the table") != _normalise_text("on the floor")
