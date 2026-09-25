from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v73_run", HERE / "run.py")
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class OfflineContractTest(unittest.TestCase):
    def test_reference_fidelity_and_backend_contract(self) -> None:
        report = RUN.contract_test(RUN._read_json(RUN.CONFIG_PATH))
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["reference_file_count"], 48)
        self.assertEqual(
            report["reference_copy_differences"],
            ["videoseal/tools/tool_map.py", "videoseal/tools/visual_tools.py"],
        )
        self.assertTrue(report["tool_schema_identical"])
        self.assertFalse(report["models_or_apis_called"])
        self.assertFalse(report["eval300_started"])

    def test_reference_step15_effective_behavior_is_not_misreported(self) -> None:
        report = RUN.contract_test(RUN._read_json(RUN.CONFIG_PATH))
        self.assertTrue(report["reference_flags"]["step15_force_code_present"])
        self.assertTrue(report["reference_flags"]["legacy_api_mode"])
        self.assertFalse(report["reference_flags"]["step15_force_effective"])
        self.assertTrue(report["effective_step15_discrepancy_confirmed"])

    def test_termination_modes_are_recordable_without_changing_controller(self) -> None:
        forced = {
            "steps": [
                {"observation": {"forced": True, "mode": "full_video"}}
            ]
        }
        self.assertEqual(RUN._termination_mode(forced, None), "full_video_64_frame_fallback")
        # A normal visual_inspect at step 15 must not be mislabeled as forced.
        step15 = {"steps": [{} for _ in range(14)] + [{"action": {"name": "visual_inspect"}}]}
        self.assertEqual(RUN._termination_mode(step15, None), "parser_or_runtime_failure")
        self.assertEqual(RUN._termination_mode({"answer": "A", "steps": []}, None), "normal_retrieval_answer")
        self.assertEqual(RUN._termination_mode({}, {"status": "timeout"}), "timeout")

    def test_launcher_pythonpath_can_import_hierarchical_dependencies(self) -> None:
        cfg = RUN._read_json(RUN.CONFIG_PATH)
        env = RUN._environment(cfg, "hierarchical")
        parts = env["PYTHONPATH"].split(":" )
        self.assertIn(str(RUN.RUNTIME_ROOT), parts)
        self.assertIn(str(RUN.RUNTIME_ROOT / "src"), parts)
        self.assertEqual(env["CONCURRENCY"], "1")
        self.assertEqual(env["TASK_TIMEOUT_SEC"], "1000")

    def test_live_report_telemetry_from_synthetic_trajectory_without_models(self) -> None:
        uid = "synthetic_1_1"
        trajectory = {
            "uid": uid,
            "video_id": "synthetic",
            "run_id": "run-1",
            "created_at": "2026-08-20T00:00:00+00:00",
            "finished_at": "2026-08-20T00:00:02+00:00",
            "system_prompt": "same downstream prompt",
            "tools_schema": "same schema",
            "answer": "A",
            "elapsed_sec": 2.0,
            "steps": [
                {
                    "action": {"name": "visual_retrieve", "arguments": {"query": "person opens door"}},
                    "timing": {"model_elapsed_sec": 0.2, "tool_elapsed_sec": 0.5},
                    "observation": {
                        "name": "visual_retrieve",
                        "ok": True,
                        "output": [{"start_time": "00:00:10", "end_time": "00:00:20", "caption": "door"}],
                        "metadata": {
                            "retrieval_backend": "hierarchical",
                            "tool_elapsed_sec": 0.5,
                            "hierarchy_used": True,
                            "hierarchy_complete": True,
                            "hierarchy_stage_latency": {
                                "coarse_elapsed_sec": 0.1,
                                "medium_elapsed_sec": 0.2,
                                "fine_elapsed_sec": 0.15,
                            },
                            "hierarchy_traversal_elapsed_sec": 0.45,
                            "selected_coarse": [{"coarse_id": "c1", "start_sec": 0, "end_sec": 60}],
                            "selected_medium": [{"medium_id": "m1", "start_sec": 0, "end_sec": 30}],
                            "selected_fine": [{"fine_id": "f1", "medium_id": "m1", "start_sec": 10, "end_sec": 20}],
                            "coarse_medium_fine_paths": [
                                {"coarse": {"coarse_id": "c1"}, "medium": {"medium_id": "m1"}, "fine": {"fine_id": "f1"}}
                            ],
                        },
                    },
                },
                {
                    "action": {"name": "visual_inspect", "arguments": {"spans": []}},
                    "timing": {"model_elapsed_sec": 0.3, "tool_elapsed_sec": 1.0},
                    "observation": {
                        "name": "visual_inspect", "ok": True,
                        "output": {"answer": "A"},
                        "metadata": {"tool_elapsed_sec": 1.0, "model_elapsed_sec": 0.8, "sent_image_count": 4},
                    },
                },
            ],
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            run_dir = root / "synthetic" / "run-1"
            run_dir.mkdir(parents=True)
            (run_dir / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
            metrics = root / "synthetic" / "metrics"
            metrics.mkdir()
            (metrics / f"{uid}.json").write_text(json.dumps({"status": "success"}), encoding="utf-8")
            preds = root / "synthetic" / "preds"
            preds.mkdir()
            (preds / f"{uid}.json").write_text(
                json.dumps({"uid": uid, "video_id": "synthetic", "pred": "A", "gt": "A"}),
                encoding="utf-8",
            )
            row = RUN._summarize_run(
                output=root, backend="hierarchical", uid=uid, runner_elapsed_sec=2.2
            )
        self.assertTrue(row["trajectory_valid"])
        self.assertTrue(row["backend_execution_verified"])
        self.assertTrue(row["hierarchy_used"])
        self.assertEqual(row["completion_mode"], "normal_retrieval_answer")
        self.assertEqual(row["total_frames"], 4)
        self.assertEqual(row["latency_sec"]["retrieval_total"], 0.5)
        self.assertEqual(row["latency_sec"]["hierarchical_coarse_total"], 0.1)
        self.assertEqual(row["final_candidate_spans"][0]["caption"], "door")

    def test_flat_telemetry_metadata_does_not_change_tool_response_or_order(self) -> None:
        sys.path.insert(0, str(RUN.REFERENCE_RUNTIME))
        try:
            from videoseal.tools import visual_tools

            with tempfile.TemporaryDirectory() as raw:
                index = Path(raw)
                (index / "semantic_captions.json").write_text(
                    json.dumps(
                        {
                            "0_16": {"caption": "first caption"},
                            "32_48": {"caption": "second caption"},
                        }
                    ),
                    encoding="utf-8",
                )
                client = mock.Mock()
                summary = "Span relevance: [00:00:00–00:00:16] RELATED"
                client.generate_text.return_value = summary
                client.get_last_usage.return_value = {}
                env = {
                    "VISUAL_RETRIEVE_SUMMARY_ENABLED": "1",
                    "VISUAL_RETRIEVE_SUMMARY_MAX_SPANS": "100",
                    "VISUAL_RETRIEVE_RETURN_SPANS": "0",
                    "RETRIEVE_MIN_TIME_GAP_SEC": "15",
                    "SEMANTIC_RETRIEVE_MIX": "embed",
                }
                with mock.patch.dict(os.environ, env, clear=False), mock.patch.object(
                    visual_tools, "query_embed", return_value=[("0_16", 0.9), ("32_48", 0.8)]
                ) as query_mock, mock.patch.object(
                    visual_tools, "build_mllm_client_from_env_prefix", return_value=client
                ):
                    result = visual_tools.VisualRetrieveAliasTool().forward(
                        query="first", top_k=30, video_id="synthetic", index_path=str(index)
                    )
        finally:
            sys.path.remove(str(RUN.REFERENCE_RUNTIME))
        self.assertEqual(result.output, {"summary": summary})
        self.assertEqual(result.metadata["raw_candidate_count"], 2)
        self.assertEqual(
            [row["caption"] for row in result.metadata["raw_candidates"]],
            ["first caption", "second caption"],
        )
        self.assertEqual(result.metadata["summarizer_useful_spans"][0]["caption"], "first caption")
        query_mock.assert_called_once_with(index, "first", topk=30)
        client.generate_text.assert_called_once()

    def test_historical_missing_raw_telemetry_is_explicit_not_silent_empty(self) -> None:
        extracted = RUN._retrieval_telemetry(
            {
                "output": {"summary": "historical summary"},
                "metadata": {"retrieval_backend": "videoseal_flat"},
            }
        )
        self.assertIsNone(extracted["raw_candidate_count"])
        self.assertEqual(extracted["raw_candidates"], [])
        self.assertIn("historical trajectory", extracted["extraction_error"])
        self.assertEqual(extracted["planner_visible_summary"], "historical summary")

    def test_current_structured_telemetry_extracts_without_error(self) -> None:
        extracted = RUN._retrieval_telemetry(
            {
                "output": {"summary": "same Planner-visible response"},
                "metadata": {
                    "raw_candidate_count": 1,
                    "raw_candidates": [
                        {"start_time": "00:00:00", "end_time": "00:00:16", "caption": "cap"}
                    ],
                    "summarizer_useful_spans": [
                        {"start_time": "00:00:00", "end_time": "00:00:16", "caption": "cap"}
                    ],
                },
            }
        )
        self.assertIsNone(extracted["extraction_error"])
        self.assertEqual(extracted["raw_candidate_count"], 1)
        self.assertEqual(extracted["raw_candidates"], extracted["summarizer_useful_spans"])


if __name__ == "__main__":
    unittest.main()
