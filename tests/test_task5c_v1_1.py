import inspect
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import run_task5c_v1_1
from src.retrieval import task5c_v1_1 as v11


def plan(operation="count_occurrences", modalities=None):
    return {"answer_requirement": {"operation": operation}, "resolver_modalities": modalities or ["speech"], "requires_local_visual_inspection": False, "audio_role": "direct_answer"}


def cues(phrase="target", interval=(0.0, 2.0)):
    return {"quoted_phrases": [{"text": phrase}], "time_cues": [{"start_sec": interval[0], "end_sec": interval[1]}]}


def context(question="question", frames=None):
    return {"question": question, "selected_visual_evidence_frames": frames or [], "anchor_resolution": {"raw_question_intervals": [], "final_search_intervals": []}, "local_visual_refinement": {"executed": bool(frames)}}


def speech(text="target", start=0.5, end=1.0):
    return {"candidate_id": "speech", "candidate_type": "speech_segment", "modality": "speech", "transcript_text": text, "start_time": start, "end_time": end, "warnings": []}


class Task5CV11Tests(unittest.TestCase):
    def test_a_completely_outside_segment_is_excluded(self):
        valid, excluded = v11.validate_asr_segment(raw_start_time=3.0, raw_end_time=4.0, pass_decode_start_sec=0.0, pass_decode_end_sec=2.0, hard_question_start_sec=0.0, hard_question_end_sec=2.0, text="target", decoding_pass="complete", segment_id="x")
        self.assertIsNone(valid)
        self.assertEqual(excluded["warnings"], ["asr_segment_outside_decode_interval"])

    def test_b_partially_outside_segment_is_clipped_with_raw_times_preserved(self):
        valid, excluded = v11.validate_asr_segment(raw_start_time=-1.0, raw_end_time=1.0, pass_decode_start_sec=0.0, pass_decode_end_sec=2.0, hard_question_start_sec=0.0, hard_question_end_sec=2.0, text="target", decoding_pass="complete", segment_id="x")
        self.assertIsNone(excluded)
        self.assertEqual((valid["raw_start_time"], valid["raw_end_time"]), (-1.0, 1.0))
        self.assertEqual((valid["effective_start_time"], valid["effective_end_time"]), (0.0, 1.0))
        self.assertEqual(valid["warnings"], ["asr_segment_clipped_to_decode_interval"])

    def test_c_recovered_phrase_can_be_questionable_without_current_fallback_requirement(self):
        before, _ = v11.classify_v1_1(context(question="How many times does the male speak?"), plan(), cues(), [])
        after, _ = v11.classify_v1_1(context(question="How many times does the male speak?"), plan(), cues(), [speech()])
        self.assertEqual(before["evidence_status"], "insufficient")
        self.assertEqual(after["evidence_status"], "questionable")
        fallback_was_triggered, fallback_recovered = True, True
        fallback_required = after["evidence_status"] == "insufficient"
        self.assertTrue(fallback_was_triggered and fallback_recovered)
        self.assertFalse(fallback_required)

    def test_d_failed_fallback_remains_additionally_required(self):
        assessment, _ = v11.classify_v1_1(context(), plan(), cues(), [])
        self.assertEqual(assessment["evidence_status"], "insufficient")
        self.assertTrue(assessment["evidence_status"] == "insufficient")

    def test_e_pre_and_post_reference_evaluations_use_their_own_candidate_lists(self):
        pre = v11.evaluate_candidates([], (1.0, 2.0))
        post = v11.evaluate_candidates([speech(start=1.2, end=1.5)], (1.0, 2.0))
        self.assertEqual(pre["selected_candidate_count"], 0)
        self.assertFalse(pre["reference_interval_hit"])
        self.assertEqual(post["selected_candidate_count"], 1)
        self.assertTrue(post["reference_interval_hit"])

    def test_f_broad_acoustic_source_is_detected_without_inherited_warning(self):
        candidate = {"candidate_id": "a", "modality": "acoustic", "start_time": 1.0, "end_time": 2.0, "source_start_time": 0.0, "source_end_time": 3.0, "local_audio_clip_reference": None, "warnings": []}
        diagnostics = v11.broad_acoustic_diagnostics([candidate], plan("describe_sound", ["acoustic"]))
        self.assertEqual(diagnostics[0]["ambiguity_codes"], ["broad_source_acoustic_region", "local_acoustic_semantics_not_verified"])

    def test_g_visual_accounting_uses_canonical_frames_only(self):
        input_data = {"selected_visual_evidence_frames": [{"timestamp": 0}, {"timestamp": 1}], "local_visual_refinement": {"micro_windows": [{"source_frame_paths": [1, 2, 3], "selected_frame_paths": [1, 2]}], "dense_frames": [{}, {}]}}
        accounting = v11.visual_accounting(input_data, legacy_count=5)
        self.assertEqual(accounting["selected_visual_frames"], 2)
        self.assertEqual(accounting["unique_selected_visual_frame_count"], 2)

    @patch("src.retrieval.task5c_v1_1.read_local_audio")
    def test_h_complete_pass_recovery_skips_chunks(self, read_audio):
        read_audio.return_value = (np.zeros(16000 * 2, dtype=np.float32), 16000)
        fallback = v11.staged_local_asr_fallback(case_id="case", video_id="video", search_interval=(0, 2), hard_question_interval=(0, 2), phrases=["target"], source_audio=Path("x.wav"), transcriber=lambda audio, sr: {"segments": [{"start": 0.2, "end": 0.8, "text": "Target."}]})
        self.assertTrue(fallback["early_stop_after_complete_pass"])
        self.assertFalse(fallback["chunk_fallback_triggered"])
        self.assertEqual(fallback["model_calls"], 1)

    @patch("src.retrieval.task5c_v1_1.read_local_audio")
    def test_i_failed_complete_pass_runs_chunks(self, read_audio):
        read_audio.return_value = (np.zeros(16000 * 2, dtype=np.float32), 16000)
        fallback = v11.staged_local_asr_fallback(case_id="case", video_id="video", search_interval=(0, 2), hard_question_interval=(0, 2), phrases=["target"], source_audio=Path("x.wav"), transcriber=lambda audio, sr: {"segments": [{"start": 0.2, "end": 0.8, "text": "other"}]})
        self.assertFalse(fallback["early_stop_after_complete_pass"])
        self.assertTrue(fallback["chunk_fallback_triggered"])
        self.assertGreater(fallback["model_calls"], 1)

    def test_j_references_cannot_enter_core_classifier_or_fallback_module(self):
        source = inspect.getsource(v11)
        self.assertNotIn("provided_timestamp", source)
        self.assertNotIn("reference", inspect.getsource(v11.staged_local_asr_fallback))

    def test_k_no_llm_or_vlm_calls(self):
        source = inspect.getsource(v11) + inspect.getsource(run_task5c_v1_1)
        self.assertNotIn("anthropic", source)
        self.assertNotIn("messages.create", source)
        self.assertIn('"task5c_llm_api_calls": 0', source)
        self.assertIn('"task5c_vlm_calls": 0', source)

    def test_l_no_case_question_answer_or_timestamp_hardcoding_in_core_module(self):
        source = inspect.getsource(v11)
        self.assertNotRegex(source, r"000\d")
        self.assertNotIn('["answer"]', source)


if __name__ == "__main__":
    unittest.main()
