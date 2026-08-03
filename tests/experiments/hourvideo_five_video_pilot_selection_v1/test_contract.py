import unittest

from experiments.hourvideo_five_video_pilot_selection_v1.core import (
    contains_protected_evaluation_key,
    validate_selection_specs,
)


def _spec(index: int) -> dict:
    uid = f"video-{index}"
    return {
        "video_uid": uid,
        "selection_category": f"category-{index}",
        "primary_question_id": f"{uid}_question-{index}",
        "selection_rationale": "Question-text-aligned selection rationale.",
    }


class FiveVideoSelectionContractTests(unittest.TestCase):
    def test_five_unique_specs_pass(self) -> None:
        self.assertEqual(validate_selection_specs([_spec(i) for i in range(5)]), [])

    def test_wrong_count_fails(self) -> None:
        self.assertIn(
            "exactly five videos are required",
            validate_selection_specs([_spec(i) for i in range(4)]),
        )

    def test_duplicate_video_fails(self) -> None:
        specs = [_spec(i) for i in range(5)]
        specs[4]["video_uid"] = specs[0]["video_uid"]
        self.assertIn("duplicate video UID", validate_selection_specs(specs))

    def test_question_must_belong_to_video(self) -> None:
        specs = [_spec(i) for i in range(5)]
        specs[2]["primary_question_id"] = "other_question"
        self.assertTrue(
            any(
                error.startswith("question/video mismatch")
                for error in validate_selection_specs(specs)
            )
        )

    def test_task_is_allowed_but_answer_and_timestamp_are_protected(self) -> None:
        self.assertFalse(contains_protected_evaluation_key({"task": "summarization"}))
        self.assertTrue(
            contains_protected_evaluation_key({"correct_answer_label": "A"})
        )
        self.assertTrue(
            contains_protected_evaluation_key({"relevant_timestamps": [1, 2]})
        )


if __name__ == "__main__":
    unittest.main()
