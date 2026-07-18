import copy
import inspect
import json
import unittest
from pathlib import Path

from scripts.run_task5b_retrieval import chinese_html
from src.retrieval.task5b import (
    apply_budget, clip_candidate, link_candidates, load_plans_immutable,
    match_quoted_phrase, modalities_to_execute, resolve_question_intervals,
)


class Task5BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_path=Path("outputs/question_planner/v2/task5a_plans.jsonl")
        cls.plans=[json.loads(x) for x in cls.plan_path.read_text(encoding="utf-8").splitlines()]

    def test_01_load_task5a_plans_without_modification(self):
        before=self.plan_path.read_bytes(); cloned=load_plans_immutable(self.plans); cloned[0]["plan"]["audio_role"]="changed"
        self.assertNotEqual(cloned[0],self.plans[0]); self.assertEqual(before,self.plan_path.read_bytes())

    def test_02_explicit_timestamp_is_hard_constraint(self):
        cues={"time_cues":[{"raw_text":"00:03-00:04","start_sec":3,"end_sec":4}],"quoted_phrases":[],"relative_temporal_expressions":[]}
        result=resolve_question_intervals("at 00:03-00:04",cues,"during",20)
        self.assertTrue(result["raw_question_intervals"][0]["hard_constraint"]); self.assertEqual((result["final_search_intervals"][0]["start_sec"],result["final_search_intervals"][0]["end_sec"]),(2,5))

    def test_03_between_range_remains_one_interval(self):
        row=next(x for x in self.plans if "between 00:17 and 00:28" in x["raw_question"])
        result=resolve_question_intervals(row["raw_question"],row["deterministic_cues"],"count_within",127)
        self.assertEqual(len(result["raw_question_intervals"]),1); self.assertEqual((result["final_search_intervals"][0]["start_sec"],result["final_search_intervals"][0]["end_sec"]),(17,28))

    def test_04_very_start_maps_to_first_five_seconds(self):
        result=resolve_question_intervals("What happened at the very start?",{"time_cues":[],"quoted_phrases":[],"relative_temporal_expressions":[]},"none",100)
        self.assertEqual(result["deterministic_interpretations"][0]["deterministic_interpretation"],"first_5_seconds"); self.assertEqual(result["raw_question_intervals"][0]["end_sec"],5); self.assertEqual(result["final_search_intervals"][0]["end_sec"],5)

    def test_05_followed_adds_post_event_context(self):
        cues={"time_cues":[{"raw_text":"00:13-00:14","start_sec":13,"end_sec":14}],"quoted_phrases":[],"relative_temporal_expressions":[]}
        result=resolve_question_intervals("What sound followed at 00:13-00:14?",cues,"during",50)
        self.assertEqual(result["expanded_intervals"][0]["expand_after_sec"],2); self.assertEqual(result["executed_relation"],"after")

    def test_06_exact_case_and_punctuation_speech_matching(self):
        rows=[{"transcript_segment_id":"a","text":"Wait","start_time":1,"end_time":2},{"transcript_segment_id":"b","text":"WAIT!","start_time":2,"end_time":3},{"transcript_segment_id":"c","text":"wait.","start_time":3,"end_time":4}]
        methods={x["match_method"] for x in match_quoted_phrase("Wait",rows)}
        self.assertIn("unicode_normalized_exact",methods); self.assertTrue(methods.intersection({"case_insensitive","punctuation_insensitive"}))

    def test_07_fuzzy_fallback_is_labelled(self):
        rows=[{"transcript_segment_id":"a","text":"oh men","start_time":1,"end_time":2}]
        self.assertEqual(match_quoted_phrase("oh man",rows)[0]["match_method"],"token_fuzzy_fallback")

    def test_08_resolver_modalities_control_execution(self):
        executed,skipped=modalities_to_execute({"resolver_modalities":["speech"]})
        self.assertEqual(executed,["speech"]); self.assertEqual({x["modality"] for x in skipped},{"visual","acoustic"})

    def test_09_required_visual_branch_is_executable(self):
        plan={"resolver_modalities":["visual"],"requires_local_visual_inspection":True,"visual_route":"anchor_guided_local_refinement"}
        executed,_=modalities_to_execute(plan); self.assertIn("visual",executed); self.assertTrue(plan["requires_local_visual_inspection"])

    def test_10_coarse_to_micro_provenance_and_linking(self):
        coarse={"start_time":0,"end_time":10}; micro={"candidate_id":"m","modality":"visual","start_time":2,"end_time":5}
        self.assertIsNotNone(clip_candidate(coarse,[{"start_sec":2,"end_sec":5}]))
        linked=link_candidates([micro,{"candidate_id":"s","modality":"speech","start_time":3,"end_time":4}],"during")
        self.assertEqual(set(linked[0]["source_candidate_ids"]),{"m","s"})

    def test_11_local_micro_selection_is_not_full_video(self):
        micro=json.loads(Path("outputs/visual_micro_index/00003/microclip_index.json").read_text(encoding="utf-8"))["microclips"]
        local=[x for x in micro if clip_candidate(x,[{"start_sec":2,"end_sec":5}])]
        self.assertLess(len(local),len(micro)); self.assertTrue(all(x["start_time"]<5 and x["end_time"]>2 for x in local))

    def test_12_count_candidates_are_not_truncated(self):
        candidates=[{"candidate_id":str(i),"modality":"speech","start_time":i,"end_time":i+.1} for i in range(10)]
        selected,budget=apply_budget(candidates,"count_occurrences"); self.assertEqual(len(selected),10); self.assertEqual(budget["removed_candidates"],[])

    def test_13_delay_trigger_and_responses_are_preserved(self):
        candidates=[{"candidate_id":"trigger","modality":"speech","start_time":1,"end_time":1.2},{"candidate_id":"response","modality":"speech","start_time":1.3,"end_time":2}]
        selected,_=apply_budget(candidates,"measure_delay"); self.assertEqual({x["candidate_id"] for x in selected},{"trigger","response"})

    def test_14_all_candidates_survive_budget_as_audit_input(self):
        candidates=[{"candidate_id":str(i),"modality":"acoustic","start_time":i*5,"end_time":i*5+5,"similarity_score":1-i/10,"hard_timestamp_consistent":True} for i in range(5)]
        original=copy.deepcopy(candidates); selected,budget=apply_budget(candidates,"describe_sound")
        self.assertEqual(candidates,original); self.assertLess(len(selected),len(candidates)); self.assertTrue(budget["removed_candidates"])

    def test_15_reference_timestamps_not_accessible_in_retrieval_builder(self):
        from scripts.run_task5b_retrieval import build_case
        source=inspect.getsource(build_case)
        self.assertNotIn("provided_timestamp",source); self.assertNotIn('["answer"]',source); self.assertNotIn("provided_context",source)

    def test_16_task5b_has_zero_llm_calls(self):
        source=Path("scripts/run_task5b_retrieval.py").read_text(encoding="utf-8")
        self.assertNotIn("import anthropic",source); self.assertNotIn("messages.create",source); self.assertIn('"task5b_llm_api_calls":0',source)

    def test_17_chinese_html_labels_render(self):
        source=inspect.getsource(chinese_html)
        for label in ("实验概览","原始问题","锚点解析","局部视觉精查","人工检查建议"): self.assertIn(label,source)

    def test_18_output_tree_is_separate_from_tasks_1_to_5a(self):
        from scripts.run_task5b_retrieval import OUT
        self.assertEqual(OUT.as_posix().split("/")[-1],"planner_guided_retrieval"); self.assertNotIn("question_planner/v2",OUT.as_posix())


if __name__=="__main__":unittest.main()
