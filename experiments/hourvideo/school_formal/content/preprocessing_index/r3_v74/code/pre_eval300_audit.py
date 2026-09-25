from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("v73_run_for_audit", HERE / "run.py")
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)
CONFIG = RUN._read_json(HERE / "config.json")
OUTPUT = Path(CONFIG["output_root"])
AUDIT = OUTPUT / "pre_eval300_audit"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tool_output_text(step: dict[str, Any]) -> str:
    observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
    output = observation.get("output")
    if isinstance(output, dict):
        return str(output.get("answer") or output.get("summary") or "")
    return str(output or "")


def _inspect_spans(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, step in enumerate(trajectory.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        if action.get("name") != "visual_inspect":
            continue
        arguments = action.get("arguments") if isinstance(action.get("arguments"), dict) else {}
        rows.append({"step": index, "spans": arguments.get("spans") or [], "context": arguments.get("context")})
    return rows


def _authorization_row(row: dict[str, Any]) -> dict[str, Any]:
    trajectory = _read(Path(row["trajectory_path"]))
    steps = trajectory.get("steps") if isinstance(trajectory.get("steps"), list) else []
    final_indices = [
        index
        for index, step in enumerate(steps)
        if isinstance(step, dict) and "<final>" in str(step.get("model_response") or "")
    ]
    final_index = max(final_indices) if final_indices else len(steps)
    valid_tools = []
    inspectors = []
    for index, step in enumerate(steps[:final_index], start=1):
        if not isinstance(step, dict):
            continue
        action = step.get("action") if isinstance(step.get("action"), dict) else {}
        observation = step.get("observation") if isinstance(step.get("observation"), dict) else {}
        if action.get("name") and observation and observation.get("ok") is not False and not observation.get("error"):
            valid_tools.append((index, str(action.get("name")), step))
            if action.get("name") == "visual_inspect":
                inspectors.append((index, step))
    last_valid = valid_tools[-1] if valid_tools else None
    last_inspector = inspectors[-1] if inspectors else None
    inspector_text = _tool_output_text(last_inspector[1]) if last_inspector else ""
    search_more = "SEARCH_MORE" in inspector_text.upper() if inspector_text else None
    answer_match = re.search(r"(?im)^\s*Answer\s*:\s*([^\n]+)", inspector_text)
    answer_line = answer_match.group(1).strip() if answer_match else None
    explicit_options = []
    if answer_line and "SEARCH_MORE" not in answer_line.upper():
        explicit_options = list(dict.fromkeys(re.findall(r"(?<![A-Z])([A-E])(?![A-Z])", answer_line.upper())))
    prediction = str(row.get("prediction") or "").upper()
    option_consistent = len(explicit_options) == 1 and explicit_options[0] == prediction
    if last_inspector is None or not inspector_text:
        authorized: bool | None = None
        verdict = None
    elif not last_valid or last_valid[1] != "visual_inspect":
        authorized = False
        verdict = "SEARCH_MORE" if search_more else "STALE_INSPECTOR_EVIDENCE"
    elif search_more:
        authorized = False
        verdict = "SEARCH_MORE"
    elif len(explicit_options) == 1:
        authorized = option_consistent
        verdict = "SUFFICIENT_OPTION"
    else:
        authorized = False
        verdict = "AMBIGUOUS_OR_MISSING_OPTION"
    return {
        "uid": row["uid"],
        "backend": row["backend"],
        "completion_mode": row["completion_mode"],
        "normal_retrieval_answer_semantics": (
            "ended before max-step full-video fallback; this label alone does not prove sufficient evidence"
        ),
        "prediction": prediction,
        "final_before_step": final_index + 1 if final_indices else None,
        "last_valid_tool": (
            {"step": last_valid[0], "name": last_valid[1]} if last_valid else None
        ),
        "last_inspector_step": last_inspector[0] if last_inspector else None,
        "last_inspector_verdict": verdict,
        "last_inspector_output": inspector_text or None,
        "last_inspector_search_more": search_more,
        "last_inspector_answer_line": answer_line,
        "last_inspector_explicit_options": explicit_options,
        "planner_prediction_matches_inspector_option": option_consistent,
        "evidence_authorized_by_last_inspector": authorized,
    }


def _updated_smoke_report() -> dict[str, Any]:
    original = _read(OUTPUT / "paired_smoke_report.json")
    updated = copy.deepcopy(original)
    for backend_report in updated["reports"]:
        backend = backend_report["backend"]
        refreshed = []
        for old_row in backend_report["rows"]:
            outer = (old_row.get("latency_sec") or {}).get("runner_outer_monotonic")
            refreshed.append(
                RUN._summarize_run(
                    output=OUTPUT / "smoke" / backend,
                    backend=backend,
                    uid=old_row["uid"],
                    runner_elapsed_sec=outer,
                    runner_returncode=int(old_row.get("runner_returncode") or 0),
                )
            )
        backend_report["rows"] = refreshed
    updated.update(
        {
            "formal_names": {
                "videoseal_flat": "videoseal_flat_top30",
                "hierarchical": "hierarchical_fixed_width_3_3_6_lexical_coarse",
            },
            "method_description": "lexical-guided hierarchical visual-semantic retrieval",
            "experiment_type": "accuracy-efficiency comparison",
            "fairness_contract": {
                "paired_runtime_correctness": True,
                "backend_isolation": True,
                "hierarchy_execution": True,
                "planner_visible_tool_contract": True,
                "shared_downstream": True,
                "equal_raw_candidate_budget": False,
                "raw_candidate_budget_difference_intended": True,
                "flat_raw_candidate_max": 30,
                "hierarchical_raw_candidate_max": 6,
            },
            "telemetry_note": (
                "New runs record raw candidates and summarizer useful spans in metadata without changing tool output. "
                "Historical flat smoke trajectories predate this telemetry and explicitly carry extraction_error."
            ),
            "eval300_started": False,
        }
    )
    return updated


def _disagreement(updated: dict[str, Any], authorization: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for report in updated["reports"] for row in report["rows"]]
    by_key = {(row["backend"], row["uid"]): row for row in rows}
    a6_uid = next(row["uid"] for row in rows if row["uid"].startswith("a6d45"))
    ab_uid = next(row["uid"] for row in rows if row["uid"].startswith("ab93"))
    b9_uid = next(row["uid"] for row in rows if row["uid"].startswith("b9eed"))
    flat_a6 = by_key[("videoseal_flat", a6_uid)]
    hier_a6 = by_key[("hierarchical", a6_uid)]
    flat_traj = _read(Path(flat_a6["trajectory_path"]))
    hier_traj = _read(Path(hier_a6["trajectory_path"]))
    return {
        "definitions": {
            "baseline": "videoseal_flat_top30",
            "ours": "hierarchical_fixed_width_3_3_6_lexical_coarse",
            "raw_candidate_budgets_equal": False,
        },
        "primary_case": {
            "uid": a6_uid,
            "gold": "A",
            "flat_prediction": "A",
            "hierarchical_prediction": "B",
            "flat_raw_top30": [
                {
                    "query": trace["query"],
                    "raw_candidate_count": trace["raw_candidate_count"],
                    "raw_candidates": trace["raw_candidates"],
                    "summarizer_useful_spans": trace["summarizer_useful_spans"],
                    "planner_visible_summary": trace["planner_visible_summary"],
                    "extraction_error": trace["telemetry_extraction_error"],
                }
                for trace in flat_a6["retrieval_trace"]
            ],
            "flat_inspect_spans": _inspect_spans(flat_traj),
            "hierarchical_retrievals": hier_a6["retrieval_trace"],
            "hierarchical_inspect_spans": _inspect_spans(hier_traj),
            "stage_findings": {
                "coarse_candidates": (
                    "first query routed to vacuum-related C03/C04 regions; second selected C06/C03/C04. "
                    "No gold temporal annotation exists to prove the first exact loss stage."
                ),
                "medium_candidates": (
                    "selected M008/M009/M010 then M020/M019/M016; retained captions/actions were "
                    "vacuum, opening drawers or opening cabinets rather than cooking preparation."
                ),
                "fine_candidates": (
                    "six Fine spans per retrieval inherited those Medium captions; the second retrieval "
                    "contained open-drawer/open-cabinet windows, not direct cooking evidence."
                ),
                "summarizer": (
                    "free-form summary relabeled 11:15-11:30 as initial organization and 14:00-14:15 "
                    "as cooking preparations, then inferred equal duration B from weak raw evidence."
                ),
                "planner_span_selection": (
                    "only the first hierarchical retrieval was inspected. After Inspector SEARCH_MORE, "
                    "the second retrieval was not sent to Inspector before final B."
                ),
                "inspector": "the only hierarchical Inspector returned SEARCH_MORE with confidence 0.00",
                "planner_final": "Planner finalized B after a retrieval, not after an authorizing Inspector",
                "flat_caveat": (
                    "flat Inspector authorized A, but its input context stated organization=48s and "
                    "cooking=64s; the verbal A rationale is arithmetically inconsistent and only coincides with gold."
                ),
            },
            "primary_attribution": "insufficient_evidence_to_determine",
            "contributing_attributions": [
                "fine_ranking_or_quota_miss",
                "summarizer_filtering",
                "planner_span_selection",
                "planner_final_mismatch",
            ],
        },
        "other_cases": [
            {
                "uid": ab_uid,
                "gold": "C",
                "flat_prediction": "E",
                "hierarchical_prediction": "E",
                "finding": (
                    "Both inspected broad puzzle-piece sorting but missed the brief grab-and-throw tabs action; "
                    "both Inspectors inferred color sorting E from incomplete evidence."
                ),
                "primary_attribution": "insufficient_evidence_to_determine",
                "contributing_attributions": ["inspector_reasoning"],
            },
            {
                "uid": b9_uid,
                "gold": "D",
                "flat_prediction": "A",
                "hierarchical_prediction": "A",
                "finding": (
                    "Both paths anchored on plank-like frames and did not retrieve Sprinting Drills. Flat Inspector "
                    "returned ambiguous A,E and Planner selected A; hierarchical Inspector said SEARCH_MORE and "
                    "later retrievals were not inspected before final A."
                ),
                "primary_attribution": "insufficient_evidence_to_determine",
                "contributing_attributions": [
                    "inspector_reasoning",
                    "planner_span_selection",
                    "planner_final_mismatch",
                ],
            },
        ],
        "answer_authorization": authorization,
    }


def _index_snapshot() -> dict[str, Any]:
    videos = [line.strip().split("_")[0] for line in (HERE / "three_question_uids.txt").read_text().splitlines() if line.strip()]
    records = []
    hierarchy_root = Path(CONFIG["hierarchical_root"])
    case_configs = hierarchy_root.parent / "case_configs"
    for video_id in videos:
        case = hierarchy_root / video_id
        config_path = case_configs / f"{video_id}.json"
        case_cfg = _read(config_path)
        paths = [
            case / "shared_hierarchy.json",
            case / "r3_2_navigation_map.json",
            case / "r3_medium_captions.json",
            case / "medium_siglip.float32.npy",
            config_path,
            Path(case_cfg["siglip_npz"]),
            Path(case_cfg["siglip_metadata"]),
        ]
        records.append(
            {
                "video_id": video_id,
                "files": [
                    {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
                    for path in paths
                    if path.is_file()
                ],
            }
        )
    return {
        "current_index_files": records,
        "historical_verification": {
            "fine_siglip_npz": "verified against per-shard historical metadata for all three smoke videos",
            "navigation_caption_medium_embedding": (
                "unverified against build-time hash: no historical artifact SHA manifest was found; current hashes frozen here"
            ),
        },
    }


def _code_snapshot() -> dict[str, Any]:
    code_paths = [
        HERE / "README.md",
        HERE / "DATA_MACHINE_RUNBOOK.md",
        HERE / "DIFF_FROM_REFERENCE_48.md",
        HERE / "config.json",
        HERE / "run.py",
        HERE / "paired_eval_statistics.py",
        HERE / "test_contract.py",
        HERE / "pre_eval300_audit.py",
        HERE / "build_transfer_package.py",
        HERE / "reference_runtime/videoseal/tools/tool_map.py",
        HERE / "reference_runtime/videoseal/tools/retrieval_adapter.py",
        HERE / "reference_runtime/videoseal/tools/visual_tools.py",
    ]
    result_paths = sorted((OUTPUT / "smoke").rglob("trajectory.json")) + sorted((OUTPUT / "smoke").rglob("preds/*.json"))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": CONFIG,
        "code_files": [
            {"path": str(path.relative_to(HERE)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in code_paths
        ],
        "existing_smoke_artifacts": [
            {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in result_paths
        ],
        "index_snapshot": _index_snapshot(),
        "reference_bundle_sha256": _sha256(Path(CONFIG["reference_bundle"])),
        "models_or_apis_called": False,
        "smoke_rerun": False,
        "eval300_started": False,
    }


def _timing_protocol() -> str:
    return """# V7.3 formal timing protocol

## Frozen comparison

Baseline is `videoseal_flat_top30`; ours is
`hierarchical_fixed_width_3_3_6_lexical_coarse`. Both backends share the same
Planner, VideoSEAL summarizer, Inspector, sampling rules and budget ceilings,
but raw retrieval candidate budgets differ: flat returns at most 30 and the
hierarchical method at most 6. Actual steps, Inspector calls and inspected
frames are trajectory outcomes and are reported as accuracy-efficiency results.

## Required clocks

Use monotonic wall-clock measurements and record separately:

1. service/model initialization outside question E2E;
2. hierarchical SigLIP encoder cold initialization/query encoding;
3. cold first retrieval and warm subsequent retrieval calls;
4. Planner calls and total;
5. retrieval calls and total;
6. Inspector calls, model calls and total;
7. full-video fallback latency;
8. per-question E2E;
9. offline indexing time separately, never inside online E2E.

## Current process semantics

The reference timeout path uses a fresh spawned worker for every question.
The SigLIP encoder cache is process-local and lazily loads on first `encode`, so
the observed ~27 seconds recurs on every hierarchical question, while later
retrievals in that same question are warm. Formal runtime-faithful E2E must retain
this cost and additionally expose it as `siglip_encoder_cold_load_sec`.

Running all 300 UIDs from one shard runner does not currently reuse the encoder
because each timed question is still spawned. Cross-question reuse would require
a persistent worker/sidecar and would change runtime isolation and timeout
semantics; it is not implemented in V7.3.

## Warm-up policy

Do not use an Eval300 question for warm-up. No warm-up is required for the
runtime-faithful primary result. A future separately versioned deployment may
encode a fixed neutral non-evaluation string before agent timing using the same
encoder instance, but that is not part of frozen V7.3 and must never enter the
trajectory or affect prediction. Calling `_encoder()` alone is insufficient
because model loading is lazy and happens on first `encode`.
"""


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    updated = _updated_smoke_report()
    all_rows = [row for report in updated["reports"] for row in report["rows"]]
    authorization = [_authorization_row(row) for row in all_rows]
    disagreement = _disagreement(updated, authorization)
    snapshot = _code_snapshot()
    offline_cost = {
        "flat": {
            "build_report": str(
                OUTPUT.parent
                / "hourvideo_v7_2_videoseal_visual_only_embedding_fair_smoke_v1/indexes/visual_only_flat_index_build_report.json"
            ),
            "storage_bytes": 25397638,
            "documents": 324,
            "embedding_model": "text-embedding-3-large",
            "artifact_hash_verification": "21/21 artifacts and 3/3 caption sources verified",
            "embedding_calls_in_builder_code_path": 3,
            "actual_historical_request_count_cost_and_exact_wall_time": "unverified",
        },
        "hierarchical_three_smoke_videos": {
            "retrieval_file_storage_bytes": 9901644,
            "fine_siglip_frames": [1800, 1822, 1549],
            "fine_siglip_elapsed_sec_total": 147.737,
            "medium_caption_local_qwen_calls": 116,
            "medium_caption_generation_sec_total": 159.764,
            "medium_caption_model_load_sec_total": 14.466,
            "external_api_calls": 0,
            "coarse_map_local_qwen3_calls": 3,
            "coarse_map_wall_sec_total": 58.503,
            "coarse_map_tokens": {"input": 40807, "output": 3495},
            "complete_offline_index_wall_time": "unverified",
            "provenance": snapshot["index_snapshot"]["historical_verification"],
        },
    }
    audit = {
        "experiment": CONFIG["experiment"],
        "formal_names": updated["formal_names"],
        "method_description": "lexical-guided hierarchical visual-semantic retrieval",
        "experiment_type": "accuracy-efficiency comparison",
        "smoke_conclusions": updated["fairness_contract"],
        "medium_scoring": {
            "formula": "0.6 * min_max_normalized_siglip + 0.3 * caption_lexical",
            "third_term": None,
            "renormalize_by_0_9": False,
            "coarse_score_role": "hard gate only",
            "effective_weight_ratio": "2:1",
            "constant_rescaling_changes_ranking_without_threshold": False,
            "tie_break": ["combined_score descending", "start_sec ascending", "medium_id ascending"],
        },
        "fine_caption": {
            "source": "parent Medium r3_medium_captions.json qwen_caption",
            "fine_has_independent_caption": False,
            "same_medium_two_fines_same_caption": True,
            "different_time_and_embedding": True,
            "summary_ambiguity_risk": (
                "yes: repeated parent caption makes two Fine windows from one Medium textually indistinguishable "
                "except for timestamps; observed summaries may relabel/infer actions beyond the raw caption"
            ),
        },
        "summarizer": {
            "input_candidate_max": {"videoseal_flat_top30": 30, "hierarchical_fixed_width_3_3_6_lexical_coarse": 6},
            "configured_summary_max_spans": 100,
            "effective_useful_span_max": {"videoseal_flat_top30": 30, "hierarchical_fixed_width_3_3_6_lexical_coarse": 6},
            "max_tokens": 800,
            "temperature": 0.0,
            "planner_visible_default": {"summary": "free-form text"},
            "planner_visible_useful_spans": False,
            "summary_text_may_mention": "up to every input candidate; code does not cap mentions below the 800-token limit",
            "tool_schema_identical": True,
        },
        "telemetry": {
            "future_metadata_complete": True,
            "tool_response_changed": False,
            "candidate_ranking_changed": False,
            "historical_flat_raw_candidates": "unavailable with explicit extraction_error",
            "historical_hierarchical_raw_candidates": "recovered from recorded provenance paths",
        },
        "answer_authorization_counts": {
            "true": sum(item["evidence_authorized_by_last_inspector"] is True for item in authorization),
            "false": sum(item["evidence_authorized_by_last_inspector"] is False for item in authorization),
            "unknown": sum(item["evidence_authorized_by_last_inspector"] is None for item in authorization),
        },
        "siglip_cold_load": {
            "cause": "fresh spawned per-question worker plus process-local lazy encoder cache",
            "observed_first_query_encode_sec": [27.443, 26.882, 26.969],
            "same_question_later_retrieval_sec": "approximately 0.065-0.142",
            "formal_eval_repeats_per_question": True,
            "persistent_reuse_requires_runtime_change": True,
            "runtime_changed_this_audit": False,
        },
        "offline_cost": offline_cost,
        "migration_readiness": {
            "code_package_can_be_prepared": True,
            "runtime_faithful_hierarchical_eval_can_run_after_gpu_release": True,
            "conditions": [
                "use frozen flat-top30 vs hierarchical-3-3-6 definitions",
                "retain per-question cold-load cost in primary E2E and report it separately",
                "verify transferred package and current index manifests before launch",
                "do not claim equal raw candidate budget",
            ],
            "remaining_provenance_risk": (
                "historical build-time hashes are missing for navigation maps, Medium captions and Medium embeddings; "
                "current hashes are frozen by this audit but historical identity remains unverified"
            ),
        },
        "models_or_apis_called": False,
        "smoke_rerun": False,
        "eval300_started": False,
    }
    _write_json(AUDIT / "updated_paired_smoke_report.json", updated)
    _write_json(AUDIT / "answer_authorization_audit.json", {"runs": authorization})
    _write_json(AUDIT / "paired_disagreement_analysis.json", disagreement)
    _write_json(AUDIT / "pre_eval300_audit_report.json", audit)
    _write_json(AUDIT / "actual_config_and_code_hash_snapshot.json", snapshot)
    (AUDIT / "timing_protocol.md").write_text(_timing_protocol(), encoding="utf-8")
    lines = [
        "# V7.3 pre-Eval300 audit",
        "",
        "- Baseline: `videoseal_flat_top30`",
        "- Ours: `hierarchical_fixed_width_3_3_6_lexical_coarse`",
        "- Type: accuracy-efficiency comparison; raw candidate budgets are intentionally unequal.",
        "- Paired runtime correctness/backend isolation/hierarchy/tool contract/shared downstream: passed.",
        "- Equal raw candidate budget: false; difference intended: true.",
        "- Answer authorization: 3 true, 3 false, 0 unknown.",
        "- Historical flat raw candidates cannot be reliably backfilled; summaries are retained with explicit errors.",
        "- SigLIP cold initialization repeats per question under the frozen spawn runtime and remains in primary E2E.",
        "- No model/API call, smoke rerun or Eval300 occurred during this audit.",
        "",
        "See the JSON artifacts in this directory for field-level evidence.",
    ]
    (AUDIT / "PRE_EVAL300_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
