"""Zero-call Task7A replay and gated five-case EgoSchema temporal-fix smoke."""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.final_qa import run_final_qa  # noqa: E402
from src.canonical_pipeline.fingerprint import (  # noqa: E402
    attach_run_fingerprint,
    build_run_fingerprint,
    case_run_fingerprint,
)
from src.canonical_pipeline.payload import build_final_payload  # noqa: E402
from src.canonical_pipeline.provider_journal import ProviderAttemptJournal  # noqa: E402
from src.canonical_pipeline.query_scoring import QueryScoreResult  # noqa: E402
from src.canonical_pipeline.reranking import build_evidence_packet  # noqa: E402
from src.canonical_pipeline.retrieval import run_planner_guided_retrieval  # noqa: E402
from src.canonical_pipeline.smoke_trace import case_runtime_trace  # noqa: E402
from src.canonical_pipeline.state import CaseState, ExecutionMode  # noqa: E402
from src.canonical_pipeline.sufficiency import run_evidence_sufficiency  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402
from src.evaluation.egoschema_adapter import (  # noqa: E402
    load_manifest,
    load_posthoc_label,
    runtime_case,
)
from src.evaluation.egoschema_runtime import (  # noqa: E402
    VisualOnlyFallback,
    apply_dataset_modality_availability,
    reveal_options_for_final_qa,
    validate_multiple_choice_answer,
)


MANIFEST = ROOT / "data/manifests/egoschema_comparison_pilot.json"
OLD = ROOT / "outputs/egoschema/ours_v0/protocol_b_pilot_25"
OUT = ROOT / "outputs/egoschema/ours_v0/protocol_b_pilot_25_fixed_v0_1"
DATA_ROOT = Path(os.environ.get("EGOSCHEMA_DATA_ROOT", "D:/ThesisData/egoschema_subset"))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class SavedVisualScoreScorer:
    """Expose saved question-conditioned visual scores without loading a model."""

    def __init__(self, traces: dict[str, dict[str, Any]]):
        self.traces = traces
        self.calls = 0

    def score_case(
        self, case: dict[str, Any], modalities: list[str]
    ) -> dict[str, QueryScoreResult]:
        if modalities != ["visual"]:
            raise RuntimeError(f"EgoSchema replay expected visual-only scoring: {modalities}")
        source = self.traces[str(case["case_id"])]
        if source["question"] != case["question"]:
            raise RuntimeError("Saved visual scores do not match the replay question")
        rows = copy.deepcopy(
            source["initial_retrieval"]["modalities"]["visual"][
                "initial_retrieval_candidates"
            ]
        )
        self.calls += 1
        score_map = {str(row["region_id"]): copy.deepcopy(row) for row in rows}
        return {
            "visual": QueryScoreResult(
                modality="visual",
                query_text=case["question"],
                query_vector=np.empty(0, dtype=np.float32),
                score_map=score_map,
                ranked_results=rows,
                top_k_results=rows[:3],
                index_files=copy.deepcopy(
                    source["initial_retrieval"]["modalities"]["visual"].get(
                        "index_files", []
                    )
                ),
                model="openai_clip_vit_b_32_saved_score_replay",
                model_load_sec=0.0,
                query_encode_sec=0.0,
                similarity_search_sec=0.0,
                historical_score_dependency=False,
            )
        }


def _old_traces(case_ids: list[str]) -> dict[str, dict[str, Any]]:
    return {
        case_id: _load(OLD / "per_case" / case_id / "runtime_result.json")
        for case_id in case_ids
    }


def _zero_replay() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    safe_manifest = load_manifest(MANIFEST)
    case_ids = [str(row["case_id"]) for row in safe_manifest["cases"]]
    old = _old_traces(case_ids)
    scorer = SavedVisualScoreScorer(old)
    config = load_canonical_config(ROOT)
    budget = yaml.safe_load(config.path("task6_budget").read_text(encoding="utf-8"))
    fallback = VisualOnlyFallback()
    audits = []
    traces = []
    for case_id in case_ids:
        replay_started = time.perf_counter()
        case = runtime_case(safe_manifest, case_id)
        previous = old[case_id]
        state = CaseState(
            case_id=case_id,
            video_id=case.video_id,
            question=case.question,
            video_duration_sec=case.duration_sec,
            mode=ExecutionMode.EXECUTE_LIVE,
            source_video_path=str((DATA_ROOT / case.video_path).resolve()),
            source_audio_path=None,
            available_modalities=("visual",),
        )
        state.planner_output = copy.deepcopy(previous["planner"]["structured_output"])
        state.deterministic_cues = copy.deepcopy(
            previous["planner"]["deterministic_cues"]
        )
        state.usage["task5a_v2"] = {
            "reused_saved_structured_output": True,
            "api_calls": 0,
            "source_case_id": case_id,
        }
        apply_dataset_modality_availability(state)
        run_planner_guided_retrieval(
            state,
            ROOT,
            scorer,
            OUT / "runtime_media",
        )
        run_evidence_sufficiency(
            state,
            fallback_executor=fallback.execute,
            materialization_root=OUT / "runtime_media",
            project_root=ROOT,
        )
        build_evidence_packet(state, budget)
        build_final_payload(state, ROOT)
        if (state.preflight_result or {}).get("status") != "ready":
            raise RuntimeError(f"Task7A replay failed preflight: {case_id}")
        trace = case_runtime_trace(state)
        trace["video_duration_sec"] = case.duration_sec
        trace["zero_call_replay"] = {
            "planner_output_reused": True,
            "visual_scores_reused": True,
            "external_api_calls": 0,
            "gold_loaded": False,
            "deterministic_task7a_replay_latency_sec": (
                time.perf_counter() - replay_started
            ),
        }
        trace["task6_visual_frame_selection_anchor"] = copy.deepcopy(
            state.evidence_packet.get("visual_frame_selection_anchor")
        )
        path = OUT / "per_case" / case_id / "task7a_replay.json"
        _write(path, trace)
        traces.append(trace)
        raw = trace["initial_retrieval"]["modalities"]["visual"][
            "initial_retrieval_candidates"
        ][0]
        micro = trace["refinement"]["local_visual"]["micro_windows"]
        anchor = trace["task6_visual_frame_selection_anchor"]
        final = [
            frame["timestamp_sec"]
            for evidence in trace["final_selected_evidence"]
            for frame in evidence.get("frames", [])
        ]
        micro_intervals = [
            [float(item["start_time"]), float(item["end_time"])] for item in micro
        ]
        aligned = bool(
            anchor.get("source")
            in {
                "temporal_anchor_or_trigger",
                "visual_semantic",
                "visual_semantic_coarse",
                "true_fallback",
            }
            and (
                float(raw["start_time"]) <= 6.0
                or (
                    anchor.get("source") == "visual_semantic"
                    and anchor.get("timestamp_sec") is not None
                    and max(
                        abs(float(value) - float(anchor["timestamp_sec"]))
                        for value in final
                    )
                    <= 1.0
                )
            )
        )
        audits.append(
            {
                "case_id": case_id,
                "raw_semantic_top1": {
                    "region_id": raw["region_id"],
                    "start_sec": raw["start_time"],
                    "end_sec": raw["end_time"],
                    "score": raw["similarity_score"],
                },
                "selected_micro_windows": micro_intervals,
                "selected_micro_scores": [item.get("similarity_score") for item in micro],
                "dense_temporal_range": [
                    trace["refinement"]["local_visual"]["dense_frames"][0]["timestamp"],
                    trace["refinement"]["local_visual"]["dense_frames"][-1]["timestamp"],
                ],
                "task6_anchor_source": anchor.get("source"),
                "task6_anchor_timestamp_sec": anchor.get("timestamp_sec"),
                "final_task7a_timestamps": final,
                "alignment_pass": aligned,
            }
        )
    patterns = {tuple(item["final_task7a_timestamps"]) for item in audits}
    early = sum(
        item["final_task7a_timestamps"] == [0.0, 0.5, 1.0, 1.5]
        for item in audits
    )
    universal_micro = all(
        item["selected_micro_windows"] == [[0.0, 4.0], [2.0, 6.0]]
        for item in audits
    )
    acceptance = {
        "case_count": len(audits),
        "external_api_or_model_calls": 0,
        "gold_loaded": False,
        "saved_visual_score_replay_calls": scorer.calls,
        "early_payload_count": early,
        "distinct_final_timestamp_patterns": len(patterns),
        "universal_early_micro_windows": universal_micro,
        "late_top1_alignment_pass": all(item["alignment_pass"] for item in audits),
        "anchor_provenance_complete": all(
            item["task6_anchor_source"]
            in {
                "temporal_anchor_or_trigger",
                "visual_semantic",
                "visual_semantic_coarse",
                "true_fallback",
            }
            for item in audits
        ),
    }
    acceptance["passed"] = bool(
        len(audits) == 25
        and early < 25
        and len(patterns) > 1
        and not universal_micro
        and acceptance["late_top1_alignment_pass"]
        and acceptance["anchor_provenance_complete"]
    )
    _write(OUT / "zero_call_task7a_audit.json", audits)
    _write(OUT / "zero_call_acceptance.json", acceptance)
    return traces, acceptance


def _select_smoke(case_ids: list[str]) -> list[str]:
    old_eval = {
        case_id: _load(OLD / "per_case" / case_id / "posthoc_evaluation.json")
        for case_id in case_ids
    }
    old_trace = _old_traces(case_ids)
    selected: list[str] = []
    predicates = (
        lambda cid: old_trace[cid]["validated_answer"]["answer_status"]
        != "answered",
        lambda cid: (
            old_trace[cid]["validated_answer"]["answer_status"] == "answered"
            and not old_eval[cid]["correct"]
        ),
        lambda cid: old_eval[cid]["correct"],
    )
    for predicate in predicates:
        candidate = next(cid for cid in case_ids if predicate(cid) and cid not in selected)
        selected.append(candidate)
    seen_types = {old_eval[cid]["question_type"] for cid in selected}
    for case_id in case_ids:
        question_type = old_eval[case_id]["question_type"]
        if case_id not in selected and question_type not in seen_types:
            selected.append(case_id)
            seen_types.add(question_type)
        if len(selected) == 5:
            break
    for case_id in case_ids:
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) == 5:
            break
    return selected


def _execute_paid(
    traces: list[dict[str, Any]], *, all_cases: bool = False
) -> dict[str, Any]:
    """Execute or resume the fixed paid run without repeating durable cases."""
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is absent; paid smoke was not started")
    from google import genai

    safe_manifest = load_manifest(MANIFEST)
    case_ids = [str(row["case_id"]) for row in safe_manifest["cases"]]
    selected = case_ids if all_cases else _select_smoke(case_ids)
    by_id = {trace["case_id"]: trace for trace in traces}
    journal = ProviderAttemptJournal(OUT / "provider_attempt_journal")
    client = genai.Client()
    results = []
    for case_id in selected:
        result_path = OUT / "smoke" / "per_case" / case_id / "runtime_result.json"
        evaluation_path = result_path.parent / "posthoc_evaluation.json"
        if result_path.is_file() and evaluation_path.is_file():
            results.append(_load(evaluation_path))
            continue
        journal.assert_case_resumable(case_id)
        case = runtime_case(safe_manifest, case_id)
        replay = by_id[case_id]
        state = CaseState(
            case_id=case_id,
            video_id=case.video_id,
            question=case.question,
            video_duration_sec=case.duration_sec,
            mode=ExecutionMode.EXECUTE_LIVE,
            source_video_path=str((DATA_ROOT / case.video_path).resolve()),
            available_modalities=("visual",),
        )
        state.planner_output = copy.deepcopy(replay["planner"]["structured_output"])
        state.deterministic_cues = copy.deepcopy(replay["planner"]["deterministic_cues"])
        state.final_payload = copy.deepcopy(replay["final_payload"])
        state.preflight_result = copy.deepcopy(replay["preflight"])
        reveal_options_for_final_qa(state, case)
        run_final_qa(state, ROOT, journal.gemini_client(case_id, client))
        for attempt in state.final_model_output["attempts"]:
            if attempt.get("success"):
                journal.validated(
                    case_id,
                    "final_model_api",
                    int(attempt["attempt_number"]),
                    {"structured_output_valid": True, "retry_reason": None},
                )
        state.validated_answer = validate_multiple_choice_answer(
            state.validated_answer, case.options
        )
        runtime = {
            "case_id": case_id,
            "question": case.question,
            "final_payload": state.final_payload,
            "raw_model_output": state.final_model_output,
            "validated_answer": state.validated_answer,
            "model_usage": state.usage.get("task7b_v3_runtime"),
            "planner_calls": 0,
            "gemini_calls": state.external_calls["final_qa"],
            "gold_loaded": False,
            "task7a_replay_latency_sec": replay["zero_call_replay"][
                "deterministic_task7a_replay_latency_sec"
            ],
            "source_run": "fixed_v0_1_paid_smoke"
            if case_id in _select_smoke(case_ids)
            else "fixed_v0_1_resumed_full",
        }
        _write(result_path, runtime)
        journal.mark_case_checkpointed(case_id, result_path)
        label = load_posthoc_label(
            MANIFEST,
            case_id,
            raw_prediction_saved=True,
            validated_prediction_saved=True,
        )
        old_result = _load(OLD / "per_case" / case_id / "posthoc_evaluation.json")
        predicted = state.validated_answer.get("predicted_option_index")
        evaluation = {
            "case_id": case_id,
            "predicted_option_index": predicted,
            "predicted_option_text": state.validated_answer.get(
                "predicted_option_text"
            ),
            "answer_status": state.validated_answer["answer_status"],
            "gold_option_index": int(label["label_index"]),
            "gold_option_text": label["correct_option"],
            "correct": predicted is not None
            and int(predicted) == int(label["label_index"]),
            "old_result": {
                "predicted_option_index": old_result["predicted_option_index"],
                "answer_status": _load(
                    OLD / "per_case" / case_id / "runtime_result.json"
                )["validated_answer"]["answer_status"],
                "correct": old_result["correct"],
            },
            "final_frame_timestamps": [
                frame["timestamp_sec"]
                for evidence in replay["final_selected_evidence"]
                for frame in evidence.get("frames", [])
            ],
            "gold_loaded_posthoc_only": True,
            "question_type": next(
                row["question_type"]
                for row in safe_manifest["cases"]
                if str(row["case_id"]) == case_id
            ),
            "confidence": copy.deepcopy(state.validated_answer.get("confidence")),
        }
        _write(evaluation_path, evaluation)
        results.append(evaluation)
    summary = {
        "case_ids": selected,
        "completed": len(results),
        "correct": sum(bool(item["correct"]) for item in results),
        "wrong": sum(
            not item["correct"] and item["predicted_option_index"] is not None
            for item in results
        ),
        "abstain": sum(item["predicted_option_index"] is None for item in results),
        "planner_calls": 0,
        "gemini_calls": sum(
            len(journal.records(case_id)) for case_id in selected
        ),
        "technical_failures": 0,
    }
    _write(OUT / ("paid_run_summary.json" if all_cases else "smoke_summary.json"), summary)
    return summary


def _html_report(traces: list[dict[str, Any]], acceptance: dict[str, Any]) -> None:
    rows = []
    sections = []
    for trace in traces:
        case_id = trace["case_id"]
        initial = trace["initial_retrieval"]["modalities"]["visual"][
            "initial_retrieval_candidates"
        ]
        micro = trace["refinement"]["local_visual"]["micro_windows"]
        dense = trace["refinement"]["local_visual"]["dense_frames"]
        anchor = trace["task6_visual_frame_selection_anchor"]
        final = [
            frame
            for evidence in trace["final_selected_evidence"]
            for frame in evidence.get("frames", [])
        ]
        rows.append(
            [
                case_id,
                f"{initial[0]['start_time']}-{initial[0]['end_time']}",
                [[item["start_time"], item["end_time"]] for item in micro],
                anchor["source"],
                anchor["timestamp_sec"],
                [item["timestamp_sec"] for item in final],
            ]
        )
        bars = []
        for item in initial:
            left = float(item["start_time"]) / 1.8
            width = max(0.4, (float(item["end_time"]) - float(item["start_time"])) / 1.8)
            bars.append(
                f"<i class='coarse' style='left:{left}%;width:{width}%' title='rank={item['rank']} score={item['similarity_score']}'></i>"
            )
        micro_bars = "".join(
            f"<i class='micro' style='left:{float(item['start_time'])/1.8}%;width:{max(.4,(float(item['end_time'])-float(item['start_time']))/1.8)}%'></i>"
            for item in micro
        )
        final_bars = "".join(
            f"<i class='final' style='left:{float(item['timestamp_sec'])/1.8}%;width:.45%'></i>"
            for item in final
        )
        thumbnails = []
        for label, path in [
            *[(f"coarse R{x['rank']} {x['start_time']}s", ROOT / "outputs/visual_index" / case_id / x["representative_keyframe_path"]) for x in initial[:3]],
            *[(f"micro {x['start_time']}-{x['end_time']}s", ROOT / x["representative_frame_path"]) for x in micro],
            *[(f"final {x['timestamp_sec']}s", ROOT / x["frame_path"]) for x in final],
        ]:
            if path.is_file():
                rel = os.path.relpath(path, OUT).replace("\\", "/")
                thumbnails.append(
                    f"<figure><img src='{html.escape(rel)}'><figcaption>{html.escape(label)}</figcaption></figure>"
                )
        sections.append(
            f"<section><h2>{case_id}</h2><p>{html.escape(trace['question'])}</p>"
            f"<p><b>Anchor:</b> {anchor['source']} @ {anchor['timestamp_sec']}s</p>"
            f"<div class='axis'>{''.join(bars)}</div><div class='axis'>{micro_bars}</div>"
            f"<div class='axis'><i class='dense' style='left:{float(dense[0]['timestamp'])/1.8}%;width:{(float(dense[-1]['timestamp'])-float(dense[0]['timestamp']))/1.8}%'></i>{final_bars}</div>"
            f"<div class='gallery'>{''.join(thumbnails)}</div>"
            f"<pre>{html.escape(json.dumps({'task5c':trace['sufficiency_fallback'],'task6':trace['reranking'],'final':final},ensure_ascii=False,indent=2))}</pre></section>"
        )
    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in rows
    )
    document = f"""<!doctype html><meta charset='utf-8'><title>EgoSchema fixed temporal review</title><style>
body{{font:14px system-ui;margin:2rem;background:#f5f7fb;color:#172033}}section{{background:white;padding:1rem;margin:1.5rem 0}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:.35rem}}.axis{{height:24px;background:#e5e7eb;position:relative;margin:.5rem 0}}.axis i{{position:absolute;height:16px;top:4px}}.coarse{{background:#2563eb}}.micro{{background:#d97706}}.dense{{background:#a78bfa}}.final{{background:#111827;z-index:2}}.gallery{{display:flex;gap:.5rem;overflow:auto}}figure{{margin:0}}img{{width:150px;height:100px;object-fit:cover}}pre{{max-height:420px;overflow:auto;white-space:pre-wrap}}
</style><h1>Ours-v0 EgoSchema fixed_v0_1 zero-call Task7A review</h1><p>Acceptance: {acceptance['passed']} · external calls: 0 · blue=coarse, amber=micro, purple=dense, black=final.</p><table><tr><th>Case</th><th>Raw Top1</th><th>Micro</th><th>Anchor source</th><th>Anchor</th><th>Final frames</th></tr>{table_rows}</table>{''.join(sections)}"""
    (OUT / "ours_v0_egoschema_review.html").write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-smoke", action="store_true")
    parser.add_argument(
        "--execute-full",
        action="store_true",
        help="Resume all manifest cases, skipping durable paid checkpoints.",
    )
    args = parser.parse_args()
    if args.execute_smoke and args.execute_full:
        raise SystemExit("Choose either --execute-smoke or --execute-full")
    OUT.mkdir(parents=True, exist_ok=True)
    traces, acceptance = _zero_replay()
    _html_report(traces, acceptance)
    if not acceptance["passed"]:
        raise SystemExit("Zero-call acceptance failed; paid smoke blocked")
    paid = (
        _execute_paid(traces, all_cases=args.execute_full)
        if (args.execute_smoke or args.execute_full)
        else None
    )
    _write(
        OUT / "run_summary.json",
        {
            "zero_call_acceptance": acceptance,
            "paid_run": paid,
            "old_run_validity": "invalid_temporal_evidence_propagation",
        },
    )
    print(json.dumps({"zero_call": acceptance, "paid_run": paid}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
