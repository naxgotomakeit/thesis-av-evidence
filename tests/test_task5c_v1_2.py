import inspect
import unittest

from scripts import run_task5c_v1_2
from src.retrieval import task5c_v1_2 as v12


def candidate():
    return {"candidate_id": "acoustic", "modality": "acoustic", "start_time": 2.0, "end_time": 3.0, "source_start_time": 0.0, "source_end_time": 5.0, "similarity_score": 0.4, "local_audio_clip_reference": None}


def plan(operation="describe_sound", resolvers=None, visual=False):
    return {"answer_requirement": {"operation": operation}, "resolver_modalities": resolvers or ["acoustic"], "requires_local_visual_inspection": visual, "audio_role": "direct_answer"}


class Task5CV12Tests(unittest.TestCase):
    def test_a_explicit_timestamp_local_clip_preserves_warning_without_penalty(self):
        semantics = v12.timestamp_semantics("What sound at 00:02?", {"time_cues": [{"start_sec": 2.0, "end_sec": 3.0}]}, {})
        item = v12.broad_score_diagnostic(candidate(), role="direct_evidence", semantics=semantics, local_audio_clip_reference="clip.wav")
        self.assertTrue(item["broad_source_warning"])
        self.assertFalse(item["broad_source_affects_sufficiency"])
        self.assertIn("broad_source_acoustic_region", item["warnings"])

    def test_b_event_anchor_and_relative_search_are_distinguished(self):
        semantics = v12.timestamp_semantics("What sound followed at 00:13?", {"time_cues": [{"start_sec": 13.0, "end_sec": 14.0}]}, {})
        self.assertEqual(semantics["anchor_timestamp_semantics"], "event_anchor_interval")
        self.assertEqual(semantics["timestamp_semantics"], "relative_search_interval")
        item = v12.broad_score_diagnostic(candidate(), role="direct_evidence", semantics=semantics, local_audio_clip_reference="clip.wav")
        self.assertTrue(item["structural_evidence_available"])

    def test_c_unanchored_broad_score_can_affect_sufficiency(self):
        semantics = {"timestamp_semantics": "retrieval_discovered_interval", "anchor_timestamp_semantics": "unknown"}
        item = v12.broad_score_diagnostic(candidate(), role="resolver", semantics=semantics, local_audio_clip_reference=None)
        self.assertTrue(item["broad_source_affects_sufficiency"])

    def test_d_broad_score_as_semantic_proof_is_questionable(self):
        semantics = {"timestamp_semantics": "direct_target_interval", "anchor_timestamp_semantics": "direct_target_interval"}
        item = v12.broad_score_diagnostic(candidate(), role="direct_evidence", semantics=semantics, local_audio_clip_reference="clip.wav", broad_score_used_for_semantic_verification=True)
        source = {"sufficiency_reason_codes": [], "critical_missing_evidence": [], "ambiguity_flags": []}
        assessment = v12.recompute_structural_status(source, [item])
        self.assertTrue(item["broad_source_affects_sufficiency"])
        self.assertEqual(assessment["evidence_status"], "questionable")
        self.assertIn("local_acoustic_semantics_not_verified", assessment["sufficiency_reason_codes"])

    def test_e_supporting_acoustic_provenance_does_not_change_status(self):
        semantics = {"timestamp_semantics": "retrieval_discovered_interval", "anchor_timestamp_semantics": "unknown"}
        item = v12.broad_score_diagnostic(candidate(), role="supporting", semantics=semantics, local_audio_clip_reference=None)
        assessment = v12.recompute_structural_status({"sufficiency_reason_codes": [], "critical_missing_evidence": [], "ambiguity_flags": []}, [item])
        self.assertFalse(item["broad_source_affects_sufficiency"])
        self.assertEqual(assessment["evidence_status"], "sufficient")

    def test_f_direct_local_audio_is_structural_not_semantic_verification(self):
        semantics = {"timestamp_semantics": "approximate_temporal_phrase", "anchor_timestamp_semantics": "approximate_temporal_phrase"}
        item = v12.broad_score_diagnostic(candidate(), role="direct_evidence", semantics=semantics, local_audio_clip_reference="clip.wav")
        self.assertTrue(item["structural_evidence_available"])
        self.assertTrue(item["semantic_interpretation_pending"])

    def test_g_speaker_ambiguity_remains_questionable_independent_of_clap(self):
        source = {"sufficiency_reason_codes": ["unresolved_speaker_attribution"], "critical_missing_evidence": [], "ambiguity_flags": ["unresolved_speaker_attribution"]}
        assessment = v12.recompute_structural_status(source, [])
        self.assertEqual(assessment["evidence_status"], "questionable")

    def test_h_references_are_not_used_by_core_role_or_status_logic(self):
        source = inspect.getsource(v12)
        self.assertNotIn("provided_timestamp", source)
        self.assertNotIn("weak_reference", inspect.getsource(v12.broad_score_diagnostic))

    def test_i_no_case_question_answer_or_timestamp_hardcoding_in_core_module(self):
        source = inspect.getsource(v12)
        self.assertNotRegex(source, r"000\d")
        self.assertNotIn('["answer"]', source)

    def test_j_no_new_model_or_api_calls(self):
        source = inspect.getsource(v12) + inspect.getsource(run_task5c_v1_2)
        for marker in ("anthropic", "messages.create", "whisper.load_model", "ClapModel"):
            self.assertNotIn(marker, source)
        self.assertIn('"new_clap_inference_calls": 0', source)


if __name__ == "__main__":
    unittest.main()
