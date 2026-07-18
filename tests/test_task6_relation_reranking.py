import inspect
import unittest

from scripts import run_task6_relation_reranking
from src.retrieval import task6_relation_reranking as task6


CFG = {"max_evidence_groups_per_case": 3, "max_candidates_per_case": 6, "max_visual_frames_per_evidence_group": 4, "max_plausible_alternatives_per_relation": 2, "max_supporting_candidates_per_case": 1}


def candidate(candidate_id, modality, start=0.0, end=1.0, roles=None, **extra):
    return {"candidate_id": candidate_id, "modality": modality, "start_time": start, "end_time": end, "roles": roles or [], **extra}


class Task6RelationRerankingTests(unittest.TestCase):
    def test_a_visual_windows_deduplicate_frames_and_preserve_dense(self):
        frames = [
            {"video_id": "v", "timestamp": 1.0, "normalized_timestamp_ms": 1000, "provenance": "micro_window", "canonical_frame_path": "micro.jpg"},
            {"video_id": "v", "timestamp": 1.0, "normalized_timestamp_ms": 1000, "provenance": "dense_frame", "canonical_frame_path": "dense.jpg"},
            {"video_id": "v", "timestamp": 2.0, "normalized_timestamp_ms": 2000, "provenance": "dense_frame", "canonical_frame_path": "next.jpg"},
        ]
        retained, dropped = task6.deduplicate_visual_frames(frames, 4, 1.0)
        self.assertEqual(len(retained), 2)
        self.assertEqual(next(x for x in retained if x["timestamp"] == 1.0)["canonical_frame_path"], "dense.jpg")
        self.assertTrue(dropped)

    def test_b_trigger_and_two_responses_emit_response_and_alternative_relations(self):
        items = [candidate("trigger", "speech", 0, 1, ["trigger"]), candidate("r1", "speech", 2, 3, ["plausible_response", "alternative"]), candidate("r2", "speech", 4, 5, ["plausible_response", "alternative"])]
        relations = task6.build_speech_relations(items)
        self.assertEqual(sum(x["relation_type"] == "response_to" for x in relations), 2)
        self.assertEqual(sum(x["relation_type"] == "alternative_to" for x in relations), 1)

    def test_c_supporting_candidate_can_be_dropped_without_speech_loss(self):
        items = [candidate("speech", "speech", roles=["trigger"]), candidate("acoustic", "acoustic", roles=["supporting"]), candidate("other", "acoustic", roles=["supporting"])]
        retained, dropped, _ = task6.apply_packet_budget(items, {**CFG, "max_supporting_candidates_per_case": 0})
        self.assertIn("speech", {x["candidate_id"] for x in retained})
        self.assertEqual(len(dropped), 2)

    def test_d_direct_acoustic_clip_is_retained_without_semantic_claim(self):
        item = candidate("audio", "acoustic", roles=["direct_evidence"], local_audio_clip_reference="clip.wav", broad_source_warning=True, semantic_interpretation_pending=True)
        retained, _, _ = task6.apply_packet_budget([item], CFG)
        self.assertEqual(retained[0]["local_audio_clip_reference"], "clip.wav")
        self.assertTrue(retained[0]["semantic_interpretation_pending"])

    def test_e_fallback_recovered_evidence_survives_budget_pressure(self):
        items = [candidate("fallback", "speech", roles=["fallback_recovered"], source="local_asr_fallback"), candidate("support", "speech", roles=["supporting"])]
        retained, _, _ = task6.apply_packet_budget(items, {**CFG, "max_candidates_per_case": 1, "max_supporting_candidates_per_case": 0})
        self.assertIn("fallback", {x["candidate_id"] for x in retained})

    def test_f_soft_budget_exceeds_to_preserve_required_evidence(self):
        items = [candidate("anchor", "speech", roles=["temporal_anchor"]), candidate("resolver", "visual", roles=["resolver"])]
        retained, _, accounting = task6.apply_packet_budget(items, {**CFG, "max_candidates_per_case": 1})
        self.assertEqual(len(retained), 2)
        self.assertEqual(accounting["violations"][0]["warning"], "budget_exceeded_to_preserve_required_evidence")

    def test_g_weak_references_cannot_enter_core_ranking(self):
        source = inspect.getsource(task6)
        self.assertNotIn("provided_timestamp", source)
        self.assertNotIn("weak_reference", inspect.getsource(task6.apply_packet_budget))

    def test_h_human_audit_notes_do_not_affect_ranking(self):
        source = inspect.getsource(task6.apply_packet_budget) + inspect.getsource(task6.role_priority)
        self.assertNotIn("human_audit", source)

    def test_i_no_case_answer_or_timestamp_hardcoding_in_core_module(self):
        source = inspect.getsource(task6)
        self.assertNotRegex(source, r"000\d")
        self.assertNotIn('["answer"]', source)

    def test_j_no_model_or_api_calls(self):
        source = inspect.getsource(task6) + inspect.getsource(run_task6_relation_reranking)
        for marker in ("anthropic", "messages.create", "whisper.load_model", "import whisper", "ClapModel"):
            self.assertNotIn(marker, source)
        self.assertIn('"llm_api_calls": 0', source)


if __name__ == "__main__":
    unittest.main()
