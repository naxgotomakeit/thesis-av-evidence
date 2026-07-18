import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_task5c_evidence_sufficiency
from src.retrieval import task5c


def plan(operation="identify_object", modalities=None, local_visual=False):
    return {"answer_requirement": {"operation": operation}, "resolver_modalities": modalities or [], "requires_local_visual_inspection": local_visual}


def cues(phrase=None, interval=None):
    time_cues = [] if interval is None else [{"raw_text": "question time", "start_sec": interval[0], "end_sec": interval[1], "source": "question_text"}]
    quoted = [] if phrase is None else [{"text": phrase, "source": "question_text"}]
    return {"time_cues": time_cues, "quoted_phrases": quoted}


def record(question="question", selected_frames=None, interval=None):
    return {"case_id": "case_x", "question": question, "selected_visual_evidence_frames": selected_frames or [], "anchor_resolution": {"raw_question_intervals": [], "final_search_intervals": [] if interval is None else [{"start_sec": interval[0], "end_sec": interval[1]}]}, "local_visual_refinement": {"executed": bool(selected_frames)}}


def speech(candidate_id, text, start, end):
    return {"candidate_id": candidate_id, "candidate_type": "speech_segment", "modality": "speech", "transcript_text": text, "start_time": start, "end_time": end, "warnings": []}


class Task5CSufficiencyTests(unittest.TestCase):
    def test_a_empty_selected_evidence_is_insufficient_while_execution_can_be_complete(self):
        assessment = task5c.classify_evidence(record(), plan("describe_sound", ["acoustic"]), cues(), [])
        execution_status = "complete"
        self.assertEqual(execution_status, "complete")
        self.assertEqual(assessment["evidence_status"], "insufficient")
        self.assertIn("no_selected_candidates", assessment["sufficiency_reason_codes"])

    def test_b_missing_required_speech_requires_fallback(self):
        assessment = task5c.classify_evidence(record(), plan("identify_source", ["speech"]), cues("hello"), [])
        self.assertEqual(assessment["evidence_status"], "insufficient")
        self.assertTrue(assessment["fallback_required"])

    def test_c_required_visual_inspection_without_canonical_frames_is_insufficient(self):
        assessment = task5c.classify_evidence(record(), plan("identify_object", ["visual"], local_visual=True), cues(), [{"candidate_id": "micro", "modality": "visual", "start_time": 0, "end_time": 1, "warnings": []}])
        self.assertEqual(assessment["evidence_status"], "insufficient")
        self.assertIn("required_local_visual_refinement_has_no_canonical_frames", assessment["sufficiency_reason_codes"])

    def test_d_count_without_phrase_occurrence_triggers_fallback(self):
        assessment = task5c.classify_evidence(record(interval=(1, 3)), plan("count_occurrences", ["speech"]), cues("target", (1, 3)), [speech("s", "different words", 1, 2)])
        self.assertEqual(assessment["evidence_status"], "insufficient")
        self.assertTrue(assessment["fallback_required"])

    @patch("src.retrieval.task5c.read_local_audio")
    def test_e_local_asr_recovery_adds_provenanced_candidate(self, read_audio):
        read_audio.return_value = (np.zeros(16000 * 2, dtype=np.float32), 16000)
        fallback = task5c.run_local_asr_fallback(case_id="case_x", video_id="video_x", record=record(interval=(0, 2)), cues=cues("hello", (0, 2)), source_audio=Path("missing.wav"), transcriber=lambda audio, sr: {"segments": [{"start": 0.2, "end": 0.6, "text": "Hello!"}]})
        self.assertEqual(fallback["outcome"], "recovered_evidence")
        self.assertEqual(fallback["added_candidates"][0]["source"], "local_asr_fallback")
        self.assertGreater(fallback["model_calls"], 0)
        self.assertEqual(fallback["local_audio_duration_sec"], 2.0)
        self.assertEqual(fallback["total_decoding_audio_duration_sec"], 4.0)

    @patch("src.retrieval.task5c.read_local_audio")
    def test_f_failed_recovery_does_not_fabricate_candidate(self, read_audio):
        read_audio.return_value = (np.zeros(16000, dtype=np.float32), 16000)
        fallback = task5c.run_local_asr_fallback(case_id="case_x", video_id="video_x", record=record(interval=(0, 1)), cues=cues("target", (0, 1)), source_audio=Path("missing.wav"), transcriber=lambda audio, sr: {"segments": [{"start": 0.0, "end": 0.5, "text": "unrelated"}]})
        self.assertEqual(fallback["outcome"], "still_insufficient")
        self.assertEqual(fallback["added_candidates"], [])

    def test_g_delay_with_multiple_responses_is_questionable(self):
        candidates = [speech("trigger", "Wait", 1, 1.2), speech("response_a", "yes", 1.4, 1.8), speech("response_b", "later reply", 2.0, 2.3)]
        assessment = task5c.classify_evidence(record(question="How did the passenger respond?"), plan("measure_delay", ["speech"]), cues("Wait"), candidates)
        self.assertEqual(assessment["evidence_status"], "questionable")
        self.assertIn("multiple_plausible_responses", assessment["ambiguity_flags"])
        self.assertIn("unresolved_speaker_attribution", assessment["ambiguity_flags"])

    def test_h_reference_fields_are_not_in_sufficiency_or_fallback_code(self):
        source = inspect.getsource(task5c) + inspect.getsource(run_task5c_evidence_sufficiency)
        self.assertNotIn("provided_timestamp", source)
        self.assertIn("saved_before_reference_load", source)

    def test_i_no_case_or_answer_hardcoding(self):
        source = inspect.getsource(task5c) + inspect.getsource(run_task5c_evidence_sufficiency)
        self.assertNotRegex(source, r"000\d")
        self.assertNotIn('["answer"]', source)

    def test_j_no_llm_or_vlm_calls(self):
        source = inspect.getsource(task5c) + inspect.getsource(run_task5c_evidence_sufficiency)
        self.assertNotIn("anthropic", source)
        self.assertNotIn("messages.create", source)
        self.assertIn('"task5c_llm_api_calls": 0', source)
        self.assertIn('"task5c_vlm_calls": 0', source)

    def test_k_task5c_writes_only_its_isolated_output_directory(self):
        source = inspect.getsource(run_task5c_evidence_sufficiency)
        self.assertIn('outputs/evidence_sufficiency/task5c_v1', source)
        self.assertIn("input_integrity", source)


if __name__ == "__main__":
    unittest.main()
