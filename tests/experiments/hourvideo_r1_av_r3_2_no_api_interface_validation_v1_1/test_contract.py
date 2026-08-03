import unittest

from experiments.hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.core import (
    contains_forbidden_key,
    make_no_api_trace,
    make_requirements,
    project_safe_question,
    validate_final_output,
)


def _text_row() -> dict:
    row = {
        "qid": "video_q1",
        "video_uid": "video",
        "question": "What happened?",
        "task": "offline/task",
        "correct_answer_label": "B",
        "relevant_timestamps": "1-2",
        "canary": "protected",
    }
    for index, text in enumerate(["one", "two", "three", "four", "five"], 1):
        row[f"answer_{index}"] = text
    return row


class HourVideoInterfaceContractTests(unittest.TestCase):
    def test_projection_strips_protected_fields(self) -> None:
        projected = project_safe_question(_text_row(), list("ABCDE"))
        self.assertFalse(contains_forbidden_key(projected))
        self.assertEqual(
            list(projected),
            ["question_id", "video_uid", "question_text", "answer_options"],
        )
        self.assertEqual(
            [row["option_id"] for row in projected["answer_options"]],
            list("ABCDE"),
        )

    def test_image_options_remain_portable_asset_references(self) -> None:
        row = _text_row()
        for index in range(1, 6):
            row[f"answer_{index}"] = f"images/{index}.png"
        projected = project_safe_question(row, list("ABCDE"))
        self.assertEqual(
            projected["answer_options"][0],
            {
                "option_id": "A",
                "content_type": "image",
                "asset_ref": "images/1.png",
            },
        )

    def test_mixed_option_modalities_fail(self) -> None:
        row = _text_row()
        row["answer_5"] = "images/5.png"
        with self.assertRaisesRegex(ValueError, "mixed option modalities"):
            project_safe_question(row, list("ABCDE"))

    def test_requirements_are_deterministic_and_option_centric(self) -> None:
        projected = project_safe_question(_text_row(), list("ABCDE"))
        requirements = make_requirements(projected)
        self.assertEqual(len(requirements), 5)
        self.assertEqual(requirements[0]["requirement_id"], "video_q1::option_A")

    def test_r1_r3_question_payloads_match_without_rung_names(self) -> None:
        projected = project_safe_question(_text_row(), list("ABCDE"))
        r1 = make_no_api_trace(projected, "structural_audio_navigation")
        r3 = make_no_api_trace(projected, "audio_visual_semantic_navigation")
        self.assertEqual(r1["shared_question_payload"], r3["shared_question_payload"])
        serialized = str(r1["shared_question_payload"]) + str(r3["shared_question_payload"])
        self.assertNotIn("R1_AV", serialized)
        self.assertNotIn("R3_2", serialized)

    def test_map_capabilities_preserve_evidence_boundary(self) -> None:
        projected = project_safe_question(_text_row(), list("ABCDE"))
        r1 = make_no_api_trace(projected, "structural_audio_navigation")
        r3 = make_no_api_trace(projected, "audio_visual_semantic_navigation")
        self.assertFalse(
            r1["map_binding"]["semantic_coarse_may_be_sufficiency_evidence"]
        )
        self.assertTrue(
            r3["map_binding"]["semantic_coarse_may_be_sufficiency_evidence"]
        )
        self.assertFalse(
            r3["map_binding"]["semantic_coarse_is_reviewed_visual_confirmation"]
        )

    def test_final_output_rejects_invalid_label_and_unknown_references(self) -> None:
        output = {
            "selected_option": "F",
            "answerability": "answer_directly",
            "supporting_requirement_ids": ["unknown"],
            "supporting_evidence_ids": ["unknown"],
        }
        errors = validate_final_output(output, set(), set())
        self.assertIn("selected_option must be A-E", errors)
        self.assertIn("unknown requirement reference", errors)
        self.assertIn("unknown evidence reference", errors)


if __name__ == "__main__":
    unittest.main()
