import json
import unittest

from src.question_planner.task5a import (
    ModelReply, extract_cues, parse_model_json, plan_with_retry,
    safe_question_cases, semantic_diagnostic, validate_plan,
)


def valid_plan():
    return {
        "answer_requirement": {"operation": "describe_sound", "description": "Describe an acoustic property."},
        "anchor_cues": [{"text": "at the start", "modality": "time"}],
        "primary_anchor_modality": "time",
        "resolver_modalities": ["acoustic"],
        "audio_role": "direct_answer",
        "temporal_relation": "during",
        "requires_local_visual_inspection": False,
        "visual_route": "not_required",
        "fallback_route": "Use acoustic evidence near the textual anchor.",
        "planner_confidence": 0.9,
        "rationale": "Time anchors the event. Acoustic evidence resolves the requested property.",
    }


class CueExtractionTests(unittest.TestCase):
    def test_timestamp_parsing(self):
        cues = extract_cues("What happened at 01:03?")
        self.assertEqual(cues["time_cues"], [{"raw_text":"01:03","cue_type":"explicit_timestamp","start_sec":63.0,"end_sec":63.0,"source":"question_text"}])

    def test_time_range_parsing_without_duplicate_singles(self):
        cues = extract_cues("What happened at 00:13–00:14?")
        self.assertEqual(len(cues["time_cues"]), 1)
        self.assertEqual((cues["time_cues"][0]["start_sec"], cues["time_cues"][0]["end_sec"]), (13.0, 14.0))
        cues = extract_cues("How many between 17 and 28 seconds?")
        self.assertEqual((cues["time_cues"][0]["start_sec"], cues["time_cues"][0]["end_sec"]), (17.0, 28.0))

    def test_quoted_phrase_extraction(self):
        cues = extract_cues('Who said “Oh man” after "Wait"?')
        self.assertEqual([x["text"] for x in cues["quoted_phrases"]], ["Oh man", "Wait"])
        self.assertEqual(cues["relative_temporal_expressions"][0]["raw_text"].lower(), "after")

    def test_single_quoted_phrase_exact_regression(self):
        cues = extract_cues("What generated the spoken '10' heard at 00:03–00:04?")
        self.assertEqual(cues["quoted_phrases"], [{"raw_text":"'10'","text":"10","cue_type":"quoted_phrase","source":"question_text"}])

    def test_between_clock_range_is_one_cue(self):
        cues = extract_cues("How many times was it said between 00:17 and 00:28?")
        self.assertEqual(cues["time_cues"], [{"raw_text":"between 00:17 and 00:28","cue_type":"explicit_time_range","start_sec":17.0,"end_sec":28.0,"source":"question_text"}])

    def test_no_time_cue_question(self):
        cues = extract_cues("What object made the soft sound?")
        self.assertEqual(cues["time_cues"], [])
        self.assertEqual(cues["relative_temporal_expressions"], [])


class ValidationAndRetryTests(unittest.TestCase):
    def test_schema_validation(self):
        self.assertEqual(validate_plan(valid_plan()), [])
        bad = valid_plan(); bad["audio_role"] = "always_use_everything"; bad["planner_confidence"] = 2
        self.assertIn("invalid audio_role", validate_plan(bad))
        self.assertIn("planner_confidence must be between 0 and 1", validate_plan(bad))

    def test_invalid_model_json_handling(self):
        with self.assertRaises(json.JSONDecodeError): parse_model_json("```json\n{}\n```")

    def test_retry_behavior(self):
        calls = []
        def request(system, user, number):
            calls.append((number, user))
            text = "not json" if number == 1 else json.dumps(valid_plan())
            return ModelReply(text, 0.1, 10, 10)
        result = plan_with_retry("What sound was heard at the start?", extract_cues("What sound was heard at the start?"), request)
        self.assertTrue(result["retry_required"])
        self.assertEqual(result["validation_status"], "valid")
        self.assertEqual(len(calls), 2)
        self.assertIn("failed strict JSON/schema validation", calls[1][1])

    def test_ground_truth_and_reference_fields_are_discarded(self):
        source = [{"case_id":"x", "question":"What sound?", "answer":"SECRET ANSWER", "provided_context":"SECRET CONTEXT", "provided_timestamp_start":42, "answer_options":["A"]}]
        safe = safe_question_cases(source)
        self.assertEqual(safe, [{"case_id":"x", "question":"What sound?"}])
        serialized = json.dumps(safe)
        self.assertNotIn("SECRET", serialized)
        self.assertNotIn("timestamp", serialized)
        self.assertNotIn("answer", serialized)


class SemanticDiagnosticTests(unittest.TestCase):
    def test_visual_route_preserves_function_despite_anchor_label_issue(self):
        plan = valid_plan()
        plan.update({
            "answer_requirement": {"operation":"identify_object","description":"Identify the object associated with a sound."},
            "primary_anchor_modality":"visual",
            "resolver_modalities":["visual","acoustic"],
            "audio_role":"direct_answer",
            "requires_local_visual_inspection":True,
            "visual_route":"anchor_guided_local_refinement",
        })
        cues = extract_cues("What object made the sound at 00:10?")
        diagnostic = semantic_diagnostic("What object made the sound at 00:10?", cues, plan)
        self.assertEqual(diagnostic["analysis_status"], "questionable")
        self.assertEqual(diagnostic["functional_impact"], "labeling_quality_only")

    def test_missing_required_visual_resolver_may_omit_evidence(self):
        plan = valid_plan()
        plan.update({"answer_requirement":{"operation":"identify_object","description":"Identify an object."}, "resolver_modalities":["acoustic"], "requires_local_visual_inspection":False, "visual_route":"not_required"})
        diagnostic = semantic_diagnostic("What object made the noise?", extract_cues("What object made the noise?"), plan)
        self.assertEqual(diagnostic["analysis_status"], "inconsistent")
        self.assertEqual(diagnostic["functional_impact"], "may_omit_required_evidence")


if __name__ == "__main__": unittest.main()
