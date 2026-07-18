"""One canonical per-question execution controller."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Callable

import yaml

from src.instrumentation.timing import StageTimer, existing_offline_artifact_timings, timing_consistency

from .final_qa import run_final_qa
from .payload import build_final_payload
from .planner import run_planner
from .reranking import build_evidence_packet
from .retrieval import run_planner_guided_retrieval
from .state import CaseState, ExecutionMode
from .sufficiency import FallbackExecutor, run_evidence_sufficiency
from .versions import CanonicalConfig


def _jsonl_by_case(path: Path) -> dict[str, dict[str, Any]]:
    return {row["case_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


class CanonicalOnlineRunner:
    """Drive a single case through final validated canonical behavior."""

    def __init__(
        self,
        config: CanonicalConfig,
        *,
        manifest_path: Path | None = None,
        data_root: Path | None = None,
        query_scorer: Any | None = None,
        materialization_root: Path | None = None,
    ):
        self.config = config
        self.manifest_path = manifest_path or config.path("case_manifest")
        configured_root = os.getenv("EGOSOUND_DATA_ROOT")
        self.data_root = data_root or (Path(configured_root) if configured_root else None)
        self.query_scorer = query_scorer
        self.materialization_root = materialization_root or config.project_root / "outputs/canonical_pipeline/runtime_media"

    def _safe_case(self, case_id: str) -> dict[str, Any]:
        rows = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        source = next((row for row in rows if row["case_id"] == case_id), None)
        if source is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        video_id = str(source.get("video_id", "")).strip()
        question = str(source.get("question", "")).strip()
        if not video_id or not question:
            raise ValueError(f"Manifest case lacks video_id or question: {case_id}")
        index_files = [
            self.config.project_root / "outputs/visual_index" / video_id / "visual_state_regions.json",
            self.config.project_root / "outputs/visual_index" / video_id / "region_embeddings.npy",
            self.config.project_root / "outputs/audio_index" / video_id / "transcript_embedding_index.json",
            self.config.project_root / "outputs/audio_index" / video_id / "transcript_embeddings.npy",
            self.config.project_root / "outputs/audio_index" / video_id / "acoustic_embedding_index.json",
            self.config.project_root / "outputs/audio_index" / video_id / "acoustic_embeddings.npy",
        ]
        missing = [str(path) for path in index_files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Reusable offline indexes missing for {case_id}: {missing}")
        audio_metadata = json.loads((self.config.project_root / "outputs/audio_index" / video_id / "audio_metadata.json").read_text(encoding="utf-8"))
        source_audio = Path(audio_metadata["audio_path"])
        source_video: Path | None = None
        if self.data_root is not None and source.get("video_path"):
            source_video = self.data_root / str(source["video_path"])
        if self.manifest_path.resolve() != self.config.path("case_manifest").resolve():
            if source_video is None or not source_video.is_file():
                raise FileNotFoundError(f"Source video unavailable for manifest case {case_id}; configure EGOSOUND_DATA_ROOT")
            if not source_audio.is_file():
                raise FileNotFoundError(f"Source audio unavailable for manifest case {case_id}: {source_audio}")
        # Strict allowlist: gold/reference/answer fields never enter CaseState.
        return {"case_id": source["case_id"], "video_id": video_id, "question": question, "video_duration": float(source["video_duration"]), "source_video_path": str(source_video) if source_video else None, "source_audio_path": str(source_audio)}

    def validate_case(self, case_id: str) -> dict[str, Any]:
        """Validate one manifest-selected case and return only online-safe fields."""
        return self._safe_case(case_id)

    def run_case(
        self,
        case_id: str,
        *,
        mode: ExecutionMode = ExecutionMode.REGRESSION_REPLAY,
        planner_request: Callable | None = None,
        fallback_executor: FallbackExecutor | None = None,
        final_client: Any | None = None,
    ) -> CaseState:
        """Run one case without serialized stage handoffs or historical patches."""
        case = self._safe_case(case_id)
        state = CaseState(case_id=case["case_id"], video_id=case["video_id"], question=case["question"], video_duration_sec=case["video_duration"], mode=mode, source_video_path=case["source_video_path"], source_audio_path=case["source_audio_path"])
        plans = _jsonl_by_case(self.config.path("planner_plans"))
        frozen_task5c = _jsonl_by_case(self.config.path("task5c_frozen"))
        budget = yaml.safe_load(self.config.path("task6_budget").read_text(encoding="utf-8"))
        timer = StageTimer()
        with timer.stage("online_end_to_end_total"):
            with timer.stage("question_planner", parent_stage="online_end_to_end_total"):
                run_planner(state, frozen_record=plans.get(case_id), request=planner_request)
            with timer.stage("modality_routing", parent_stage="online_end_to_end_total"):
                state.record("modality_routing", "executed_by_task5b_v1_1")
            with timer.stage("retrieval_total", parent_stage="online_end_to_end_total"):
                run_planner_guided_retrieval(state, self.config.project_root, self.query_scorer if mode is ExecutionMode.EXECUTE_LIVE else None, self.materialization_root if mode is ExecutionMode.EXECUTE_LIVE else None)
            with timer.stage("evidence_sufficiency", parent_stage="online_end_to_end_total"):
                run_evidence_sufficiency(state, frozen_v1_2=frozen_task5c.get(case_id) if mode is ExecutionMode.REGRESSION_REPLAY else None, fallback_executor=fallback_executor, materialization_root=self.materialization_root if mode is ExecutionMode.EXECUTE_LIVE else None, project_root=self.config.project_root if mode is ExecutionMode.EXECUTE_LIVE else None)
            with timer.stage("relation_reranking", parent_stage="online_end_to_end_total"):
                build_evidence_packet(state, budget)
            with timer.stage("final_payload_build", parent_stage="online_end_to_end_total"):
                build_final_payload(state, self.config.project_root)
            if mode is ExecutionMode.REGRESSION_REPLAY:
                timer.skip("final_model_api", "regression_replay_zero_external_calls", parent_stage="online_end_to_end_total")
                timer.skip("structured_output_parse", "regression_replay_zero_external_calls", parent_stage="final_model_api")
                timer.skip("local_validation", "regression_replay_zero_external_calls", parent_stage="online_end_to_end_total")
                run_final_qa(state, self.config.project_root)
            else:
                with timer.stage("final_model_api", parent_stage="online_end_to_end_total"):
                    run_final_qa(state, self.config.project_root, final_client)
        state.timings = existing_offline_artifact_timings() + timer.as_dicts()
        return state

    def run_live_case(
        self,
        case_id: str,
        *,
        planner_request: Callable,
        fallback_executor: FallbackExecutor,
        final_client: Any,
        return_preflight_blocked: bool = False,
    ) -> CaseState:
        """Execute the real canonical online path with complete stage records."""
        case = self._safe_case(case_id)
        state = CaseState(
            case_id=case["case_id"], video_id=case["video_id"], question=case["question"],
            video_duration_sec=case["video_duration"], mode=ExecutionMode.EXECUTE_LIVE,
            source_video_path=case["source_video_path"], source_audio_path=case["source_audio_path"],
        )
        budget = yaml.safe_load(self.config.path("task6_budget").read_text(encoding="utf-8"))
        timer = StageTimer()
        with timer.stage("online_end_to_end_total"):
            with timer.stage("question_planner", parent_stage="online_end_to_end_total") as stage:
                run_planner(state, request=planner_request)
                stage.api_calls = state.external_calls["planner"]
                stage.model_calls = state.external_calls["planner"]

            with timer.stage("retrieval_total", parent_stage="online_end_to_end_total"):
                with timer.stage("modality_routing", parent_stage="retrieval_total"):
                    state.record("modality_routing", "task5a_plan_fields_executed_by_task5b_v1_1")
                if self.query_scorer is None:
                    raise RuntimeError("Generalized live retrieval requires a fresh question-conditioned query scorer")
                run_planner_guided_retrieval(state, self.config.project_root, self.query_scorer, self.materialization_root)
            runtime = (state.retrieval_result or {}).get("runtime", {})
            scoring_runtime = runtime.get("query_scoring", {})
            executed = {item["modality"] for item in (state.retrieval_result or {}).get("executed_modalities", [])}
            for modality in ("visual", "speech", "acoustic"):
                name = f"{modality}_retrieval"
                score_time = scoring_runtime.get(modality, {})
                query_scored = modality in scoring_runtime
                branch_executed = modality in executed
                if branch_executed or query_scored:
                    query_total = sum(float(score_time.get(key, 0.0)) for key in ("encoder_model_load_sec", "query_encode_sec", "similarity_search_sec"))
                    timer.measured(
                        name, float(runtime.get("branch_latency_sec", {}).get(modality, 0.0) if branch_executed else 0.0) + query_total,
                        parent_stage="retrieval_total",
                        model_calls=int(runtime.get("branch_local_model_calls", {}).get(modality, 0)),
                        notes=[] if branch_executed else ["query_scored_for_primary_anchor_without_task5b_resolver_branch"],
                    )
                    load_sec = float(score_time.get("encoder_model_load_sec", 0.0))
                    if load_sec > 0:
                        timer.measured(f"{modality}_encoder_model_load", load_sec, parent_stage=name)
                    else:
                        timer.skip(
                            f"{modality}_encoder_model_load",
                            "persistent_encoder_already_loaded",
                            parent_stage=name,
                        )
                    timer.measured(f"{modality}_query_encode", float(score_time.get("query_encode_sec", 0.0)), parent_stage=name, model_calls=1)
                    timer.measured(f"{modality}_similarity_search", float(score_time.get("similarity_search_sec", 0.0)), parent_stage=name)
                else:
                    timer.skip(name, "modality_not_required", parent_stage="retrieval_total")
                    timer.skip(f"{modality}_encoder_model_load", "modality_not_required", parent_stage=name)
                    timer.skip(f"{modality}_query_encode", "modality_not_required", parent_stage=name)
                    timer.skip(f"{modality}_similarity_search", "modality_not_required", parent_stage=name)
            visual_refinement = (state.retrieval_result or {}).get("local_visual_refinement", {})
            if visual_refinement.get("executed"):
                timer.measured("local_visual_refinement", float(runtime.get("local_visual_refinement_sec") or 0.0), parent_stage="refinement_total")
            else:
                timer.skip("local_visual_refinement", "planner_did_not_require_local_visual_refinement", parent_stage="refinement_total")
            timer.measured("temporal_linking", float(runtime.get("temporal_linking_sec", 0.0)), parent_stage="retrieval_total")
            timer.measured("evidence_budget_selection", float(runtime.get("budget_selection_sec", 0.0)), parent_stage="retrieval_total")
            if visual_refinement.get("executed"):
                timer.measured("refinement_total", float(runtime.get("local_visual_refinement_sec") or 0.0), parent_stage="retrieval_total")
            else:
                timer.skip("refinement_total", "no_question_dependent_visual_refinement_executed", parent_stage="retrieval_total")

            with timer.stage("sufficiency_and_fallback", parent_stage="online_end_to_end_total"):
                run_evidence_sufficiency(state, fallback_executor=fallback_executor, materialization_root=self.materialization_root, project_root=self.config.project_root)
            c5 = state.usage["task5c_v1_2_runtime"]
            timer.measured("evidence_sufficiency", c5["pre_classification_sec"] + c5["post_fallback_processing_sec"], parent_stage="sufficiency_and_fallback")
            timer.measured("fallback_decision", c5["fallback_decision_sec"], parent_stage="sufficiency_and_fallback")
            if c5.get("local_audio_materialization_sec") is not None:
                timer.measured("local_audio_refinement", c5["local_audio_materialization_sec"], parent_stage="sufficiency_and_fallback")
                timer.measured("local_wav_materialization", c5["local_audio_materialization_sec"], parent_stage="local_audio_refinement")
            else:
                timer.skip("local_audio_refinement", "selected_acoustic_evidence_did_not_require_materialization", parent_stage="sufficiency_and_fallback")
                timer.skip("local_wav_materialization", "selected_acoustic_evidence_did_not_require_materialization", parent_stage="local_audio_refinement")
            if state.fallback_execution_count:
                timer.measured("fallback_execution", c5["fallback_execution_sec"], parent_stage="sufficiency_and_fallback", model_calls=c5["fallback_model_calls"])
                timer.measured("fallback_local_asr", (state.fallback_result or {}).get("canonical_local_asr_total_sec", c5["fallback_local_asr_sec"]), parent_stage="fallback_execution", model_calls=c5["fallback_model_calls"])
            else:
                timer.skip("fallback_execution", "fallback_not_triggered", parent_stage="sufficiency_and_fallback")
                timer.skip("fallback_local_asr", "fallback_not_triggered", parent_stage="fallback_execution")

            with timer.stage("reranking_and_packet", parent_stage="online_end_to_end_total"):
                build_evidence_packet(state, budget)
            c6 = state.usage["task6_v1_2_runtime"]
            timer.measured("relation_construction", c6["relation_construction_sec"], parent_stage="reranking_and_packet")
            timer.measured("relation_reranking", c6["relation_reranking_sec"], parent_stage="reranking_and_packet")
            timer.measured("evidence_packet_build", c6["evidence_packet_build_sec"], parent_stage="reranking_and_packet")

            with timer.stage("final_payload_build", parent_stage="online_end_to_end_total"):
                build_final_payload(state, self.config.project_root)
            if state.preflight_result and state.preflight_result["status"] == "blocked":
                if not return_preflight_blocked:
                    raise RuntimeError(f"Task 7A preflight blocked live final QA: {state.case_id}")
                timer.skip("final_model_pipeline", "task7a_preflight_blocked", parent_stage="online_end_to_end_total")
                timer.skip("final_model_api", "task7a_preflight_blocked", parent_stage="final_model_pipeline")
                timer.skip("structured_output_parse", "task7a_preflight_blocked", parent_stage="final_model_pipeline")
                timer.skip("local_validation", "task7a_preflight_blocked", parent_stage="final_model_pipeline")
                state.validated_answer = {
                    "case_id": state.case_id,
                    "answer_status": "insufficient_evidence",
                    "answer": None,
                    "reasoning_summary": "Task 7A preflight blocked final QA because an answer-required asset was unavailable or invalid.",
                    "evidence_used": [],
                    "final_uncertainties": copy.deepcopy((state.final_payload or {}).get("pipeline_uncertainties", [])),
                    "missing_information": copy.deepcopy((state.preflight_result or {}).get("missing_assets", [])),
                    "confidence": {"level": "low", "basis": "Final model was not called because deterministic preflight blocked the request."},
                    "abstain": True,
                    "result_source": "deterministic_task7a_preflight",
                }
                state.record("final_model_api", "skipped_task7a_preflight_blocked", external_calls=0)
            else:
                with timer.stage("final_model_pipeline", parent_stage="online_end_to_end_total"):
                    run_final_qa(state, self.config.project_root, final_client)
                c7 = state.usage["task7b_v3_runtime"]
                timer.measured("final_model_api", c7["final_model_api_stage_sec"], parent_stage="final_model_pipeline", api_calls=c7["total_api_calls"], model_calls=c7["total_api_calls"])
                timer.measured("structured_output_parse", c7["structured_output_parse_sec"], parent_stage="final_model_pipeline")
                timer.measured("local_validation", c7["local_validation_sec"], parent_stage="final_model_pipeline")

        online_records = timer.as_dicts()
        top = ["question_planner", "retrieval_total", "sufficiency_and_fallback", "reranking_and_packet", "final_payload_build", "final_model_pipeline"]
        consistency = timing_consistency(online_records, top)
        online = consistency["online_end_to_end_duration_sec"]
        top_sum = consistency["non_overlapping_top_level_duration_sum_sec"]
        consistency["timing_coverage"] = None if not online else top_sum / online
        consistency["timing_coverage_warning"] = bool(consistency["timing_coverage"] is not None and consistency["timing_coverage"] < 0.90)
        state.usage["timing_consistency"] = consistency
        state.timings = existing_offline_artifact_timings() + online_records
        return state
