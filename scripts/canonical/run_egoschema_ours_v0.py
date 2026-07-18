"""Run Ours-v0 Protocol B on the frozen 25-case EgoSchema pilot.

The command is resume-safe. Offline indexing, dry-run preflight, five-case
smoke gating, live execution, and post-hoc evaluation are explicit phases.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.canonical_pipeline.fingerprint import (  # noqa: E402
    FingerprintMismatchError,
    attach_run_fingerprint,
    build_run_fingerprint,
    case_run_fingerprint,
    validate_reusable_result_fingerprint,
)
from src.canonical_pipeline.live_boundaries import (  # noqa: E402
    anthropic_requester,
    environment_presence,
)
from src.canonical_pipeline.provider_journal import (  # noqa: E402
    ProviderAttemptJournal,
    request_fingerprint,
)
from src.canonical_pipeline.query_scoring import FreshQueryScorer  # noqa: E402
from src.canonical_pipeline.runner import CanonicalOnlineRunner  # noqa: E402
from src.canonical_pipeline.smoke_trace import case_runtime_trace  # noqa: E402
from src.canonical_pipeline.versions import load_canonical_config  # noqa: E402
from src.evaluation.egoschema_adapter import (  # noqa: E402
    RetrievalProtocol,
    canonical_manifest_row,
    load_manifest,
    load_posthoc_label,
    runtime_case,
)
from src.evaluation.egoschema_runtime import (  # noqa: E402
    VisualOnlyFallback,
    apply_dataset_modality_availability,
    mc_accuracy_summary,
    reveal_options_for_final_qa,
    validate_multiple_choice_answer,
)
from src.evaluation.egoschema_visual_index import build_visual_indexes  # noqa: E402
from src.question_planner.task5a import ModelReply  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data/manifests/egoschema_comparison_pilot.json"
DEFAULT_DATA_ROOT = Path("D:/ThesisData/egoschema_subset")
OUT = ROOT / "outputs/egoschema/ours_v0/protocol_b_pilot_25"
PER_CASE = OUT / "per_case"
RUNTIME_MANIFEST = OUT / "runtime_manifest_protocol_b.json"
PROTOCOL = RetrievalProtocol.QUESTION_ONLY


class SavedAttemptReplayJournal:
    """Replay already validated provider responses without crossing a provider boundary."""

    def __init__(self, source: ProviderAttemptJournal, case_id: str):
        self.source = source
        self.case_id = case_id
        records = source.records(case_id)
        self.by_stage = {str(item["stage"]): item for item in records}
        required = {"question_planner", "final_model_api"}
        if set(self.by_stage) != required or any(
            item.get("state") != "validated" for item in records
        ):
            raise RuntimeError("Saved-attempt replay requires exactly two validated responses")

    def assert_case_resumable(self, case_id: str) -> None:
        if case_id != self.case_id:
            raise RuntimeError("Saved-attempt replay case mismatch")

    def planner_request(self, case_id: str, request: Any) -> Any:
        del request
        if case_id != self.case_id:
            raise RuntimeError("Saved planner response case mismatch")
        record = self.by_stage["question_planner"]

        def replay(system: str, user: str, number: int) -> ModelReply:
            if number != int(record["attempt_number"]):
                raise RuntimeError("Saved planner attempt number mismatch")
            actual_request = request_fingerprint(
                {"system": system, "user": user, "attempt": number}
            )
            if actual_request != record["request_sha256"]:
                raise RuntimeError("Saved planner response request fingerprint mismatch")
            response = record["response"]
            return ModelReply(
                text=str(response["text"]),
                latency_sec=float(response["latency_sec"]),
                input_tokens=int(response["input_tokens"]),
                output_tokens=int(response["output_tokens"]),
            )

        return replay

    def gemini_client(self, case_id: str, client: Any) -> Any:
        del client
        if case_id != self.case_id:
            raise RuntimeError("Saved Gemini response case mismatch")
        record = self.by_stage["final_model_api"]

        class Interactions:
            def create(self, **request: Any) -> Any:
                if request_fingerprint(request) != record["request_sha256"]:
                    raise RuntimeError("Saved Gemini response request fingerprint mismatch")
                response = record["response"]
                return SimpleNamespace(
                    output_text=response["output_text"],
                    usage=copy.deepcopy(response.get("usage") or {}),
                )

        return SimpleNamespace(interactions=Interactions())

    def validated(self, case_id: str, stage: str, attempt: int, value: dict[str, Any]) -> None:
        del case_id, stage, attempt, value

    def records(self, case_id: str) -> list[dict[str, Any]]:
        return self.source.records(case_id)


def _validate_authorized_mc_adapter_migration(
    record: dict[str, Any], expected: dict[str, Any]
) -> bool:
    """Allow only the explicitly authorized post-answer adapter source change."""
    actual = record.get("run_fingerprint")
    if not isinstance(actual, dict):
        return False
    ignored = {"fingerprint_sha256", "working_source"}
    actual_core = {key: value for key, value in actual.items() if key not in ignored}
    expected_core = {key: value for key, value in expected.items() if key not in ignored}
    return actual_core == expected_core


def _runner_with_journal(
    *,
    config: Any,
    data_root: Path,
    scorer: FreshQueryScorer,
    final_adapter: Any,
    journal: Any,
) -> CanonicalOnlineRunner:
    """Build the same frozen runner with an explicitly supplied attempt journal."""
    return CanonicalOnlineRunner(
        config,
        manifest_path=RUNTIME_MANIFEST,
        data_root=data_root,
        query_scorer=scorer,
        materialization_root=OUT / "runtime_media",
        attempt_journal=journal,
        post_planner_adapter=apply_dataset_modality_availability,
        pre_final_qa_adapter=final_adapter,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _runtime_path(case_id: str) -> Path:
    return PER_CASE / case_id / "runtime_result.json"


def _evaluation_path(case_id: str) -> Path:
    return PER_CASE / case_id / "posthoc_evaluation.json"


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _smoke_ids(cases: list[dict[str, Any]]) -> list[str]:
    """Choose five type-diverse cases without labels or predictions."""
    selected: list[str] = []
    seen_types: set[str] = set()
    for row in cases:
        question_type = str(row.get("question_type", "unknown"))
        if question_type not in seen_types:
            selected.append(str(row["case_id"]))
            seen_types.add(question_type)
        if len(selected) == 5:
            return selected
    for row in cases:
        case_id = str(row["case_id"])
        if case_id not in selected:
            selected.append(case_id)
        if len(selected) == 5:
            break
    return selected


def _safe_runtime_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases: dict[str, Any] = {}
    for source in manifest["cases"]:
        case = runtime_case(manifest, str(source["case_id"]))
        if case.audio_available:
            raise RuntimeError("This frozen EgoSchema subset was audited as visual-only")
        rows.append(canonical_manifest_row(case, PROTOCOL))
        cases[case.case_id] = case
    if len(rows) != 25 or len({row["video_id"] for row in rows}) != 25:
        raise RuntimeError("Frozen EgoSchema pilot must contain 25 distinct videos")
    _write_json(RUNTIME_MANIFEST, rows)
    return rows, cases


def _timing(trace: dict[str, Any], name: str) -> float | None:
    item = next((row for row in trace["timings"] if row["stage_name"] == name), None)
    return None if item is None else item.get("duration_sec")


def _percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _visual_intervals(trace: dict[str, Any]) -> list[tuple[float, float]]:
    return sorted(
        (float(item["start_sec"]), float(item["end_sec"]))
        for item in trace.get("final_selected_evidence", [])
        if item.get("modality") == "visual"
    )


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = float("-inf")
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def _frame_count(trace: dict[str, Any]) -> int:
    return sum(
        len(item.get("frames", []))
        for item in trace.get("final_selected_evidence", [])
        if item.get("modality") == "visual"
    )


def _failure_diagnostic(trace: dict[str, Any], correct: bool) -> dict[str, Any]:
    if correct:
        return {
            "first_likely_failure_stage": "not_applicable_correct_prediction",
            "retrieval_found_correct_region": "not_determined_no_temporal_gold",
            "basis": "The MC prediction matches gold; temporal evidence correctness was not independently annotated.",
        }
    planner_audit = next(
        (item for item in trace.get("audit", []) if item.get("stage") == "dataset_modality_availability"),
        {},
    )
    initial = trace.get("initial_retrieval", {}).get("modalities", {}).get("visual", {})
    refinement = trace.get("refinement", {}).get("local_visual") or {}
    final_frames = _frame_count(trace)
    status = (trace.get("validated_answer") or {}).get("answer_status")
    if planner_audit.get("unavailable_requested_modalities"):
        stage = "planner/routing"
        basis = "The generic AV planner requested unavailable dataset modalities and required availability projection."
    elif not initial.get("initial_retrieval_candidates"):
        stage = "coarse visual retrieval"
        basis = "No fresh visual retrieval candidates were recorded."
    elif not refinement.get("executed"):
        stage = "local refinement"
        basis = "The frozen planner did not execute local visual refinement."
    elif final_frames == 0:
        stage = "Task6 evidence compaction"
        basis = "No visual frame survived into the final model-facing context."
    elif status in {"insufficient_evidence", "query_or_premise_inconsistent"}:
        sufficiency = trace.get("sufficiency_fallback", {})
        if sufficiency.get("post_status") == "insufficient" or not final_frames:
            stage = "legitimate insufficient evidence"
            basis = (
                "The final non-answer status agrees with structural insufficiency or an empty "
                "model-facing visual packet in the saved trace."
            )
        else:
            stage = "unclear"
            basis = (
                "The model abstained, but Task5C did not independently establish structural "
                "insufficiency; human evidence review is required."
            )
    else:
        stage = "unclear"
        basis = (
            "Visual evidence reached the final selector, but EgoSchema supplies no temporal gold; "
            "the saved frames require human review to distinguish selection from MC reasoning failure."
        )
    return {
        "first_likely_failure_stage": stage,
        "retrieval_found_correct_region": "not_determined_no_temporal_gold",
        "basis": basis,
        "automatic_diagnostic_only": True,
    }


def _case_efficiency(trace: dict[str, Any], offline: dict[str, Any]) -> dict[str, Any]:
    frames = _frame_count(trace)
    intervals = _visual_intervals(trace)
    duration = float(trace.get("video_duration_sec", 180.0) or 180.0)
    model_usage = trace.get("model_usage", {})
    planner = model_usage.get("planner") or {}
    gemini = model_usage.get("gemini") or {}
    gemini_usage = gemini.get("usage") or {}
    return {
        "offline_indexing": offline,
        "planner_latency_sec": _timing(trace, "question_planner"),
        "visual_query_encode_sec": _timing(trace, "visual_query_encode"),
        "visual_similarity_search_sec": _timing(trace, "visual_similarity_search"),
        "visual_retrieval_sec": _timing(trace, "visual_retrieval"),
        "local_visual_refinement_sec": _timing(trace, "local_visual_refinement"),
        "task5c_sec": _timing(trace, "evidence_sufficiency"),
        "task6_relation_reranking_sec": _timing(trace, "relation_reranking"),
        "final_answer_api_sec": _timing(trace, "final_gemini_api"),
        "online_total_sec": _timing(trace, "online_end_to_end_total"),
        "model_facing_frame_count": frames,
        "frame_fraction_of_1fps_index": frames / 180.0,
        "selected_temporal_union_sec": _union_duration(intervals),
        "selected_temporal_fraction": _union_duration(intervals) / duration if duration else None,
        "planner_calls": int(planner.get("api_calls") or 0),
        "planner_input_tokens": planner.get("input_tokens"),
        "planner_output_tokens": planner.get("output_tokens"),
        "final_calls": int(gemini.get("total_api_calls") or 0),
        "final_input_tokens": gemini_usage.get("input_tokens"),
        "final_output_tokens": gemini_usage.get("output_tokens"),
        "final_total_tokens": gemini_usage.get("total_tokens"),
        "final_cached_tokens": gemini_usage.get("total_cached_tokens"),
        "retries": int(planner.get("retries") or 0) + int(gemini.get("retry_api_calls") or 0),
    }


def _posthoc(
    *, manifest_path: Path, trace: dict[str, Any], source: dict[str, Any], offline: dict[str, Any]
) -> dict[str, Any]:
    case_id = str(trace["case_id"])
    label = load_posthoc_label(
        manifest_path,
        case_id,
        raw_prediction_saved=True,
        validated_prediction_saved=True,
    )
    prediction = trace["validated_answer"]
    selected = prediction.get(
        "predicted_option_index", prediction.get("selected_option_index")
    )
    index = int(selected) if selected is not None else None
    correct = index is not None and index == int(label["label_index"])
    return {
        "case_id": case_id,
        "question_type": source.get("question_type"),
        "question": source["question"],
        "options": source["options"],
        "predicted_option_index": index,
        "predicted_option_text": prediction.get(
            "predicted_option_text", prediction.get("selected_option_text")
        ),
        "gold_option_index": int(label["label_index"]),
        "gold_option_text": label["correct_option"],
        "correct": correct,
        "gold_loaded_posthoc_only": True,
        "failure_analysis": _failure_diagnostic(trace, correct),
        "efficiency": _case_efficiency(trace, offline),
    }


def _aggregate(runtime: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    online = [float(item["efficiency"]["online_total_sec"]) for item in evaluations if item["efficiency"].get("online_total_sec") is not None]
    frames = [int(item["efficiency"]["model_facing_frame_count"]) for item in evaluations]
    coverage = [float(item["efficiency"]["selected_temporal_fraction"]) for item in evaluations]
    status_counts: dict[str, int] = {}
    for trace in runtime:
        status = str((trace.get("validated_answer") or {}).get("answer_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    by_type: dict[str, Any] = {}
    for question_type in sorted({str(item["question_type"]) for item in evaluations}):
        rows = [item for item in evaluations if str(item["question_type"]) == question_type]
        by_type[question_type] = {
            "cases": len(rows),
            "correct": sum(bool(item["correct"]) for item in rows),
            "accuracy": sum(bool(item["correct"]) for item in rows) / len(rows),
        }
    mc_summary = mc_accuracy_summary(evaluations)
    return {
        **mc_summary,
        "answer_status_distribution": status_counts,
        "online_latency_sec": {
            "mean": statistics.fmean(online),
            "median": statistics.median(online),
            "p95": _percentile(online, 0.95),
            "min": min(online),
            "max": max(online),
            "measured_case_count": len(online),
        },
        "mean_model_facing_frames": statistics.fmean(frames),
        "mean_temporal_coverage_fraction": statistics.fmean(coverage),
        "planner_calls": sum(item["efficiency"]["planner_calls"] for item in evaluations),
        "final_answer_calls": sum(item["efficiency"]["final_calls"] for item in evaluations),
        "retries": sum(item["efficiency"]["retries"] for item in evaluations),
        "whisper_calls": 0,
        "accuracy_by_question_type": by_type,
        "temporal_cue_strata": {
            "question_contains_explicit_time": {
                "cases": sum(bool(item["planner"]["deterministic_cues"].get("time_cues")) for item in runtime),
            },
            "global_vs_local": "not_determined_without_adding_a_new_classifier; planner operation and search intervals are retained per case",
        },
    }


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _json(value: Any) -> str:
    return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"


def _html_report(
    runtime: list[dict[str, Any]], evaluations: list[dict[str, Any]], aggregate: dict[str, Any]
) -> str:
    evaluations_by_id = {item["case_id"]: item for item in evaluations}
    overview = []
    details = []
    for trace in runtime:
        case_id = trace["case_id"]
        evaluation = evaluations_by_id[case_id]
        answer = trace["validated_answer"]
        overview.append([
            case_id,
            evaluation["question_type"],
            evaluation["question"],
            "ABSTAIN" if evaluation["predicted_option_index"] is None else evaluation["predicted_option_index"],
            evaluation["gold_option_index"],
            evaluation["correct"],
            answer.get("answer_status"),
            evaluation["failure_analysis"]["first_likely_failure_stage"],
        ])
        initial = trace["initial_retrieval"]["modalities"]["visual"]
        initial_rows = [[
            item.get("rank"), item.get("region_id"), round(float(item.get("similarity_score", 0)), 6),
            f"{item.get('start_time')}-{item.get('end_time')}", item.get("representative_keyframe_path"),
        ] for item in initial.get("initial_retrieval_candidates", [])]
        temporal_rows = [[
            item.get("rank"), item.get("region_id"), item.get("decision"), item.get("decision_reason"),
            item.get("temporal_overlap_sec"), item.get("temporal_distance_sec"),
        ] for item in trace["temporal_linking"]["candidate_temporal_decisions"]["visual"]]
        before = trace["reranking"]["before"]
        retained = trace["reranking"]["retained"]
        dropped = trace["reranking"]["actually_dropped"]
        frame_cards = []
        for evidence in trace["final_selected_evidence"]:
            for frame in evidence.get("frames", []):
                source = ROOT / frame["frame_path"]
                relative = os.path.relpath(source, OUT).replace("\\", "/")
                frame_cards.append(
                    f"<figure><img src='{html.escape(relative)}'><figcaption>"
                    f"{html.escape(str(frame['timestamp_sec']))}s · {html.escape(str(evidence['evidence_id']))}</figcaption></figure>"
                )
        timing_rows = [[
            row["stage_name"], row.get("executed"),
            "N/A" if row.get("duration_sec") is None else f"{float(row['duration_sec']):.6f}",
            row.get("skip_reason"),
        ] for row in trace["timings"]]
        details.append(f"""
<section id='{case_id}'><h2>{case_id}</h2>
<h3>Question / Options / Prediction</h3><p><strong>{html.escape(evaluation['question'])}</strong></p>
{_table(['Index','Option'], [[i, option] for i, option in enumerate(evaluation['options'])])}
{_table(['Field','Value'], [['Prediction',evaluation['predicted_option_index']],['Prediction text',evaluation['predicted_option_text']],['Gold (post-hoc)',evaluation['gold_option_index']],['Correct',evaluation['correct']],['Status',answer.get('answer_status')],['Confidence',answer.get('confidence')]])}
<h3>Planner（retrieval 只见 question）</h3>{_json(trace['planner'])}
<h3>Initial visual retrieval candidates</h3>{_table(['Rank','Region ID','Score','Interval','Keyframe'],initial_rows)}
<h3>Temporal localization</h3>{_table(['Rank','Region','Decision','Reason','Overlap','Distance'],temporal_rows)}{_json(trace['temporal_linking']['anchor_resolution'])}
<h3>Local visual refinement</h3>{_json(trace['refinement']['local_visual'])}
<h3>Task5C sufficiency / fallback</h3>{_json(trace['sufficiency_fallback'])}
<h3>Task6 before / retained / dropped</h3>{_table(['Entering','Retained','Dropped'],[[len(before),len(retained),len(dropped)]])}<details><summary>Full Task6 trace</summary>{_json(trace['reranking'])}</details>
<h3>Exact final model-facing frames</h3><div class='gallery'>{''.join(frame_cards) or '<p>No visual frames.</p>'}</div>{_json(trace['final_payload'])}
<h3>Failure analysis</h3>{_json(evaluation['failure_analysis'])}
<h3>Latency / efficiency</h3>{_table(['Stage','Executed','Latency sec','Skip reason'],timing_rows)}{_json(evaluation['efficiency'])}
</section>""")
    return f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>Ours-v0 EgoSchema Protocol B</title><style>
body{{font:14px system-ui;margin:2rem;color:#172033;background:#f4f7fb}}section{{background:white;border:1px solid #d5dfeb;border-radius:10px;padding:1.2rem;margin:1.5rem 0}}table{{width:100%;border-collapse:collapse;margin:.7rem 0}}th,td{{border:1px solid #cbd5e1;padding:.4rem;vertical-align:top}}th{{background:#eaf1f8}}pre{{white-space:pre-wrap;max-height:520px;overflow:auto;background:#f2f5f8;padding:.7rem}}.gallery{{display:flex;gap:.6rem;overflow:auto}}figure{{margin:0}}img{{height:180px}}nav{{position:sticky;top:0;background:#fff;padding:.6rem;border:1px solid #ccd6e2;z-index:2}}
</style></head><body><h1>Ours-v0 · EgoSchema Protocol B · 25 cases</h1><p>Retrieval sees question only. Options are revealed after Task7A. Gold is post-hoc only.</p>
<h2>Aggregate</h2>{_json(aggregate)}<h2>Case overview</h2>{_table(['Case','Type','Question','Prediction','Gold','Correct','Status','Suspected first failure'],overview)}
<nav>{' · '.join(f"<a href='#{item['case_id']}'>{item['case_id'][:8]}</a>" for item in runtime)}</nav>{''.join(details)}</body></html>"""


def _write_outputs(
    runtime: list[dict[str, Any]], evaluations: list[dict[str, Any]], aggregate: dict[str, Any]
) -> None:
    _write_json(OUT / "aggregate_metrics.json", aggregate)
    failures = [
        {"case_id": item["case_id"], "correct": item["correct"], **item["failure_analysis"]}
        for item in evaluations
        if not item["correct"]
    ]
    _write_json(OUT / "failure_analysis.json", {"wrong_case_count": len(failures), "cases": failures})
    (OUT / "ours_v0_egoschema_review.html").write_text(
        _html_report(runtime, evaluations, aggregate), encoding="utf-8"
    )
    lines = [
        "# Ours-v0 EgoSchema Protocol B pilot", "",
        f"- Completed: {aggregate['case_count']}/25",
        f"- Correct: {aggregate['correct']}/25",
        f"- MC accuracy: {aggregate['mc_accuracy']:.4f}",
        f"- Abstentions: {aggregate['abstention_count']}/25 ({aggregate['abstention_rate']:.2%})",
        f"- Answered-only accuracy: {aggregate['answered_only_accuracy']:.4f}" if aggregate['answered_only_accuracy'] is not None else "- Answered-only accuracy: unavailable",
        f"- Mean model-facing frames: {aggregate['mean_model_facing_frames']:.3f}",
        f"- Mean online latency: {aggregate['online_latency_sec']['mean']:.3f}s",
        f"- Planner/final calls: {aggregate['planner_calls']}/{aggregate['final_answer_calls']}",
        "- Whisper calls: 0", "",
        "Gold labels were loaded only after each runtime result was durably saved.",
    ]
    (OUT / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--build-indexes", action="store_true")
    parser.add_argument("--execute-live", action="store_true")
    parser.add_argument("--force-reindex", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    data_root = args.data_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env", override=False)
    safe_manifest = load_manifest(manifest_path)
    runtime_rows, cases_by_id = _safe_runtime_manifest(safe_manifest)
    case_ids = [row["case_id"] for row in runtime_rows]
    smoke_ids = _smoke_ids(safe_manifest["cases"])

    if args.build_indexes:
        summary = build_visual_indexes(
            project_root=ROOT,
            cases=safe_manifest["cases"],
            data_root=data_root,
            summary_path=OUT / "offline_indexing.json",
            force=args.force_reindex,
        )
        print(json.dumps({"offline_indexing": "complete", "videos": len(summary["videos"]), "summary": str(OUT / "offline_indexing.json")}, indent=2))
        return 0

    offline = _load(OUT / "offline_indexing.json")
    if offline is None or offline.get("status") != "complete":
        raise SystemExit("Offline EgoSchema visual indexes are not complete; run --build-indexes first")
    offline_by_video = {item["video_id"]: item for item in offline["videos"]}
    config = load_canonical_config(ROOT)
    scorer = FreshQueryScorer(ROOT)

    def final_adapter(state: Any) -> None:
        reveal_options_for_final_qa(state, cases_by_id[state.case_id])

    runner = _runner_with_journal(
        config=config,
        data_root=data_root,
        scorer=scorer,
        final_adapter=final_adapter,
        journal=ProviderAttemptJournal(
            OUT / "runtime_media/provider_attempt_journal"
        ),
    )
    readiness = []
    for case_id in case_ids:
        safe = runner.validate_case(case_id)
        readiness.append({
            "case_id": case_id,
            "video_id": safe["video_id"],
            "available_modalities": list(safe["available_modalities"]),
            "visual_index_ready": True,
            "audio_available": False,
            "historical_task4_dependency": False,
            "source_video_ready": Path(str(safe["source_video_path"])).is_file(),
        })
    presence = environment_presence()
    run_fingerprint = build_run_fingerprint(
        ROOT,
        config,
        manifest_path,
        (row["video_id"] for row in runtime_rows),
        planner_model=os.environ.get("ANTHROPIC_MODEL"),
        available_modalities_by_video={row["video_id"]: ("visual",) for row in runtime_rows},
    )
    case_fingerprints = {
        row["case_id"]: case_run_fingerprint(run_fingerprint, row["video_id"])
        for row in runtime_rows
    }
    manifest_record = {
        "run": "ours_v0_egoschema_protocol_b_pilot_25",
        "protocol": PROTOCOL.value,
        "case_ids": case_ids,
        "smoke_case_ids": smoke_ids,
        "mode": "execute_live" if args.execute_live else "dry_run",
        "environment_presence": presence,
        "readiness": readiness,
        "research_behavior": "frozen_ours_v0",
        "retrieval_visible_fields": ["question"],
        "final_visible_fields": ["question", "options", "task7a_selected_visual_evidence"],
        "gold_access": "posthoc_after_durable_runtime_result",
        "audio_available": False,
        "whisper_calls_expected": 0,
        "models": {
            "planner": os.environ.get("ANTHROPIC_MODEL"),
            "final": "gemini-3.5-flash",
            "thinking_level": "low",
            "store": False,
        },
        "run_fingerprint": run_fingerprint,
    }
    _write_json(OUT / "run_manifest.json", manifest_record)
    if not args.execute_live:
        print(json.dumps({"dry_run": True, "ready": len(readiness), "smoke_cases": smoke_ids, "external_calls": 0, "environment_presence": presence}, indent=2))
        return 0
    if not all(presence.values()):
        raise SystemExit("Required live environment variables are absent; no API call was made")

    existing = [_load(_runtime_path(case_id)) for case_id in case_ids]
    authorized_adapter_migrations: list[str] = []
    for case_id, result in zip(case_ids, existing):
        if result is not None:
            try:
                validate_reusable_result_fingerprint(
                    result, case_fingerprints[case_id]
                )
            except FingerprintMismatchError:
                if not _validate_authorized_mc_adapter_migration(
                    result, case_fingerprints[case_id]
                ):
                    raise
                authorized_adapter_migrations.append(case_id)
    cold = scorer.preload(case_ids[0], ("visual",))
    fallback = VisualOnlyFallback()
    from google import genai

    planner_request, planner_model = anthropic_requester()
    final_client = genai.Client()
    manifest_record["models"]["planner"] = planner_model
    manifest_record["encoder_cold_start_sec"] = cold
    manifest_record["encoder_lifecycle_before_cases"] = scorer.lifecycle_audit()
    manifest_record["authorized_mc_adapter_checkpoint_migrations"] = (
        authorized_adapter_migrations
    )
    manifest_record["provider_responses_recovered_without_new_calls"] = []
    manifest_record["prevented_duplicate_paid_calls"] = 0
    _write_json(OUT / "run_manifest.json", manifest_record)

    phases = [("smoke", smoke_ids), ("full", [item for item in case_ids if item not in smoke_ids])]
    try:
        for phase, phase_ids in phases:
            for case_id in phase_ids:
                if _runtime_path(case_id).is_file():
                    continue
                saved_records = runner.attempt_journal.records(case_id)
                recovery = bool(saved_records)
                active_runner = runner
                if recovery:
                    replay_journal = SavedAttemptReplayJournal(
                        runner.attempt_journal, case_id
                    )
                    active_runner = _runner_with_journal(
                        config=config,
                        data_root=data_root,
                        scorer=scorer,
                        final_adapter=final_adapter,
                        journal=replay_journal,
                    )
                state = active_runner.run_live_case(
                    case_id,
                    planner_request=planner_request,
                    fallback_executor=fallback.execute,
                    final_client=final_client,
                )
                state.validated_answer = validate_multiple_choice_answer(
                    state.validated_answer or {}, cases_by_id[case_id].options
                )
                trace = case_runtime_trace(state)
                trace["video_duration_sec"] = state.video_duration_sec
                trace["protocol"] = PROTOCOL.value
                trace["options_visible_during_retrieval"] = False
                trace["options_revealed_after_task7a"] = True
                trace["dataset_modality_availability"] = copy.deepcopy(
                    state.usage.get("dataset_modality_availability")
                )
                trace["resume_reconciliation"] = {
                    "provider_response_recovered_without_new_call": recovery,
                    "recovered_stages": (
                        ["question_planner", "final_model_api"] if recovery else []
                    ),
                    "prevented_duplicate_paid_calls": 2 if recovery else 0,
                    "timing_status": (
                        "reconstructed_local_path_provider_latency_not_remeasured"
                        if recovery
                        else "measured_live"
                    ),
                }
                if recovery:
                    for item in trace.get("timings", []):
                        if item.get("parent_stage") != "offline_indexing_total":
                            item["duration_sec"] = None
                            item.setdefault("notes", []).append(
                                "excluded_from_latency_aggregate_saved_provider_response_replay"
                            )
                attach_run_fingerprint(trace, case_fingerprints[case_id])
                runtime_path = _runtime_path(case_id)
                _write_json(runtime_path, trace)
                runner.mark_case_checkpointed(case_id, runtime_path)
                if recovery:
                    manifest_record[
                        "provider_responses_recovered_without_new_calls"
                    ].append(case_id)
                    manifest_record["prevented_duplicate_paid_calls"] += 2

                source = next(row for row in safe_manifest["cases"] if row["case_id"] == case_id)
                evaluation = _posthoc(
                    manifest_path=manifest_path,
                    trace=trace,
                    source=source,
                    offline=offline_by_video[state.video_id],
                )
                _write_json(_evaluation_path(case_id), evaluation)
                manifest_record["completed_case_ids"] = [
                    item for item in case_ids if _runtime_path(item).is_file() and _evaluation_path(item).is_file()
                ]
                manifest_record["current_phase"] = phase
                manifest_record["encoder_lifecycle"] = scorer.lifecycle_audit()
                manifest_record["fallback_lifecycle"] = fallback.lifecycle_audit()
                _write_json(OUT / "run_manifest.json", manifest_record)
            if phase == "smoke" and not all(_runtime_path(item).is_file() and _evaluation_path(item).is_file() for item in smoke_ids):
                raise RuntimeError("Five-case smoke gate did not complete; full run blocked")
    finally:
        manifest_record["encoder_lifecycle"] = scorer.lifecycle_audit()
        manifest_record["fallback_lifecycle"] = fallback.lifecycle_audit()
        _write_json(OUT / "run_manifest.json", manifest_record)
        scorer.close()

    runtime = [_load(_runtime_path(case_id)) for case_id in case_ids]
    evaluations = [_load(_evaluation_path(case_id)) for case_id in case_ids]
    if any(item is None for item in runtime) or any(item is None for item in evaluations):
        raise RuntimeError("Pilot ended without 25 durable runtime/evaluation pairs")
    runtime_rows_final = [item for item in runtime if item is not None]
    evaluation_rows_final = [item for item in evaluations if item is not None]
    aggregate = _aggregate(runtime_rows_final, evaluation_rows_final)
    aggregate["offline_indexing"] = {
        key: offline[key]
        for key in (
            "model_load_sec_once", "frame_sampling_sec_total", "clip_image_encoding_sec_total",
            "temporal_region_construction_sec_total", "micro_index_construction_sec_total",
            "total_offline_indexing_sec_sum", "sampled_frame_count_total", "index_size_bytes_total",
        )
    }
    aggregate["smoke"] = {"case_ids": smoke_ids, "passed": True}
    aggregate["encoder_lifecycle"] = manifest_record["encoder_lifecycle"]
    _write_outputs(runtime_rows_final, evaluation_rows_final, aggregate)
    manifest_record["status"] = "complete"
    manifest_record["completed_case_ids"] = case_ids
    manifest_record["smoke_passed"] = True
    manifest_record["aggregate_metrics_path"] = "aggregate_metrics.json"
    manifest_record["whisper_calls"] = 0
    _write_json(OUT / "run_manifest.json", manifest_record)
    print(json.dumps({
        "completed": 25,
        "correct": aggregate["correct"],
        "mc_accuracy": aggregate["mc_accuracy"],
        "smoke_passed": True,
        "html": str(OUT / "ours_v0_egoschema_review.html"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
