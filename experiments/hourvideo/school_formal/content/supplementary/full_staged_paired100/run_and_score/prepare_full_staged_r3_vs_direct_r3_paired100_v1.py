#!/usr/bin/env python3
"""Freeze the no-gold paired100 population and formal preflight artifacts.

This program is deterministic and never imports or invokes a provider.  It
uses canonical identities only for sampling, then links the selected routes to
frozen R3 Planner artifacts and already-terminal R3 Direct route statuses.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from staged_api_v1.config import StagedConfig, canonical_sha, sha256_file
from staged_api_v1.preflight import (
    legacy_dependency_files,
    legacy_downstream_config,
    live_dependency_versions,
    runtime_fingerprint_payload,
    validate,
)
from staged_api_v1.store import StagedStore, atomic_json, atomic_text


SEED = "full_staged_r3_vs_direct_r3_paired100_v1_seed_20260908"
SAMPLE_SIZE = 100
CONFIG = ROOT / "config/full_staged_api_r3_paired100_final_v2.json"
BASE_CANDIDATE = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_eval300_v1_aligned_v2/formal_candidate_manifest.json"
FROZEN_POPULATION = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_v1/paired100_manifest.json"
PREFLIGHT = ROOT / "outputs/full_staged_api_preflight/full_staged_api_r3_vs_direct_r3_paired100_final_v2"
DIRECT_ROOT = ROOT / "outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1"
PLANNER_API_ROOT = Path(
    "${SCHOOL_MAIN_SYSTEM_ROOT}/"
    "thesis-av-evidence-hourvideo/hourvideo_v6_1_runtime/outputs/experiments/"
    "hourvideo_v6_6_2_planner_api_ablation_formal_eval300_v1"
)
PLANNER_ATTEMPTS = PLANNER_API_ROOT / "telemetry/api_attempts.jsonl"
PLANNER_SUMMARY = PLANNER_API_ROOT / "canonical_summary_v2/planner_cost.json"
DIRECT_STRUCTURAL = DIRECT_ROOT / "canonical_summary_v1/structural_validation.json"
FINGERPRINT_FILES = [
    "src/staged_api_v1/contracts.py",
    "src/staged_api_v1/provider.py",
    "src/staged_api_v1/store.py",
    "src/staged_api_v1/config.py",
    "src/staged_api_v1/legacy_bridge.py",
    "src/staged_api_v1/preflight.py",
    "src/staged_api_v1/runtime.py",
    "scripts/run_full_staged_api_r3_eval300_v1.py",
    "scripts/prepare_full_staged_r3_vs_direct_r3_paired100_v1.py",
    "scripts/run_full_staged_api_r3_paired100_v1.py",
]


def _hash_rank(question_id: str) -> str:
    return hashlib.sha256(f"{SEED}\0{question_id}".encode("utf-8")).hexdigest()


def _allocate(ids: list[str]) -> tuple[dict[str, int], dict[str, dict]]:
    by_video: dict[str, list[str]] = defaultdict(list)
    video_order: list[str] = []
    for qid in ids:
        video = qid.rsplit("_", 2)[0]
        if video not in by_video:
            video_order.append(video)
        by_video[video].append(qid)
    exact = {video: SAMPLE_SIZE * len(by_video[video]) / len(ids) for video in video_order}
    quota = {video: math.floor(exact[video]) for video in video_order}
    if any(value < 1 for value in quota.values()):
        raise RuntimeError("largest-remainder allocation failed to cover every video")
    remaining = SAMPLE_SIZE - sum(quota.values())
    ranked_remainders = sorted(
        video_order,
        key=lambda video: (-(exact[video] - quota[video]), video_order.index(video)),
    )
    for video in ranked_remainders[:remaining]:
        quota[video] += 1
    audit = {
        video: {
            "canonical_count": len(by_video[video]),
            "exact_quota": exact[video],
            "floor_quota": math.floor(exact[video]),
            "remainder": exact[video] - math.floor(exact[video]),
            "final_quota": quota[video],
            "remainder_tie_break_index": video_order.index(video),
        }
        for video in video_order
    }
    return quota, audit


def _select(ids: list[str], quota: dict[str, int]) -> tuple[list[str], dict[str, list[dict]]]:
    by_video: dict[str, list[str]] = defaultdict(list)
    for qid in ids:
        by_video[qid.rsplit("_", 2)[0]].append(qid)
    selected = set()
    ranks: dict[str, list[dict]] = {}
    for video, rows in by_video.items():
        ordered = sorted(rows, key=lambda qid: (_hash_rank(qid), qid))
        ranks[video] = [
            {"question_id": qid, "selection_hash_sha256": _hash_rank(qid), "selected": index < quota[video]}
            for index, qid in enumerate(ordered)
        ]
        selected.update(ordered[: quota[video]])
    canonical_selected = [qid for qid in ids if qid in selected]
    if len(canonical_selected) != SAMPLE_SIZE or len(set(canonical_selected)) != SAMPLE_SIZE:
        raise RuntimeError("paired100 selection cardinality mismatch")
    return canonical_selected, ranks


def _planner_attempt_costs(selected: set[str]) -> dict:
    rows = []
    for line in PLANNER_ATTEMPTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        parts = str(row.get("logical_call_id", "")).split("::")
        if len(parts) != 3 or parts[0] not in selected or parts[1] != "r3_2" or parts[2] != "planner":
            continue
        rows.append({
            "question_id": parts[0],
            "attempt_index": row.get("attempt_index"),
            "status": row.get("status"),
            "input_tokens": row.get("input_tokens"),
            "output_tokens": row.get("output_tokens"),
            "latency_sec": row.get("latency_sec"),
            "recorded_estimated_cost_usd": row.get("estimated_cost_usd"),
            "estimated_cost_basis": row.get("estimated_cost_basis"),
            "is_retry": row.get("is_retry"),
            "request_id": row.get("request_id"),
        })
    known = [row for row in rows if row["recorded_estimated_cost_usd"] is not None]
    represented = {row["question_id"] for row in rows}
    return {
        "source_path": str(PLANNER_ATTEMPTS),
        "source_sha256": sha256_file(PLANNER_ATTEMPTS),
        "canonical_planner_cost_summary_path": str(PLANNER_SUMMARY),
        "canonical_planner_cost_summary_sha256": sha256_file(PLANNER_SUMMARY),
        "selected_questions_with_attempt_telemetry": len(represented),
        "selected_questions_missing_attempt_telemetry": sorted(selected - represented),
        "attempt_count": len(rows),
        "retry_attempt_count": sum(bool(row["is_retry"]) for row in rows),
        "known_cost_attempt_count": len(known),
        "unknown_cost_attempt_count": len(rows) - len(known),
        "known_recorded_estimated_cost_usd": sum(float(row["recorded_estimated_cost_usd"]) for row in known),
        "known_input_tokens": sum(int(row["input_tokens"] or 0) for row in rows),
        "known_output_tokens": sum(int(row["output_tokens"] or 0) for row in rows),
        "known_latency_sec": sum(float(row["latency_sec"] or 0) for row in rows),
        "scope_note": "historical R3 Planner generation only; reused and not charged as new paired100 spend",
        "attempts": rows,
    }


def materialise() -> dict:
    cfg = StagedConfig.load(CONFIG)
    base = validate(cfg, workspace_root=ROOT)
    ids = list(base["ordered_question_ids"])
    quota, allocation = _allocate(ids)
    recomputed, hash_ranks = _select(ids, quota)
    frozen_population_document = json.loads(FROZEN_POPULATION.read_text(encoding="utf-8"))
    selected = list(frozen_population_document["ordered_question_ids"])
    if selected != recomputed:
        raise RuntimeError("existing frozen paired100 population no longer matches its deterministic sampling rule")
    selected_set = set(selected)
    by_q = {row["question_id"]: row for row in base["planner_reuse"]["rows"]}

    direct_links = []
    for qid in selected:
        path = DIRECT_ROOT / "route_status" / f"R3:{qid}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen R3 Direct status: {qid}")
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("state") not in {"terminal_success", "terminal_failed"}:
            raise RuntimeError(f"R3 Direct route is not terminal: {qid}")
        direct_links.append({
            "question_id": qid,
            "route_id": f"R3:{qid}",
            "status_path": str(path),
            "status_sha256": sha256_file(path),
            "terminal_state": status["state"],
            "prediction_present": status.get("prediction") in list("ABCDE"),
            "use_policy": "paired analysis reference only; never supplied to Full Staged runtime",
        })

    legacy_files = legacy_dependency_files(cfg)
    dependencies = live_dependency_versions(cfg)
    runtime_fingerprint = canonical_sha(runtime_fingerprint_payload(
        config_path=CONFIG,
        workspace_root=ROOT,
        workspace_files=FINGERPRINT_FILES,
        legacy_files=legacy_files,
        dependency_versions=dependencies,
    ))
    planner_cost = _planner_attempt_costs(selected_set)
    manifest = {
        **base,
        "experiment_id": cfg.experiment_id,
        "question_count": SAMPLE_SIZE,
        "route_count": SAMPLE_SIZE,
        "ordered_question_ids": selected,
        "planner_reuse": {"count": SAMPLE_SIZE, "all_provider_anthropic_haiku45": True, "rows": [by_q[qid] for qid in selected]},
        "paired100_selection": {
            "population_size": len(ids),
            "sample_size": SAMPLE_SIZE,
            "seed": SEED,
            "algorithm": "Hamilton/largest remainder by canonical per-video count; equal remainders tie by canonical video order; within video sort SHA256(seed + NUL + question_id); select quota; output in canonical Eval300 order",
            "uses_predictions_correctness_gold_or_pilot_performance": False,
            "pilot_membership_policy": "neither intentionally included nor excluded",
            "frozen_population_source_path": str(FROZEN_POPULATION),
            "frozen_population_source_sha256": sha256_file(FROZEN_POPULATION),
            "population_reused_without_reselection": True,
            "video_allocation": allocation,
            "within_video_hash_ranks": hash_ranks,
        },
        "direct_r3_pairing": {
            "source_experiment": "direct_v1_2_3x16_r1_r3_eval300_formal_v1",
            "source_structural_validation_path": str(DIRECT_STRUCTURAL),
            "source_structural_validation_sha256": sha256_file(DIRECT_STRUCTURAL),
            "count": len(direct_links),
            "all_terminal": all(row["terminal_state"].startswith("terminal_") for row in direct_links),
            "links": direct_links,
        },
        "historical_r3_planner_cost": planner_cost,
        "provider_response_retention": {
            "contract": "complete_sanitised_provider_response_before_parse_v1",
            "journal": "journals/provider_responses.jsonl",
            "durable_order": [
                "request_start", "provider_transport", "provider_response",
                "attempt_end_and_charge", "provider_tool_envelope_validation",
                "v6_6_2_semantic_validation", "retry_or_downgrade",
            ],
            "tool_envelope_journal": "journals/controller_results.jsonl",
            "semantic_validation_and_retry_journals": [
                "legacy_live/cases/<question_id>/model_attempts.jsonl",
                "legacy_live/cases/<question_id>/shared_attempt_audit.jsonl",
            ],
            "downgrade_and_terminal_journal": "legacy_live/cases/<question_id>/r3_2/route_status.json",
            "historical_pilot_policy": "read-only; missing individual Fine payloads remain explicitly missing and are never reconstructed",
        },
        "cost_plan": {
            "pilot_observed_downstream_mean_usd_all_five": 0.15377913,
            "paired100_downstream_point_estimate_usd": 15.377913,
            "proposed_hard_budget_usd": cfg.hard_api_budget_usd,
            "proposal_status": "REQUIRES_EXPLICIT_APPROVAL_BEFORE_LIVE_START",
            "new_spend_scope": "Shared/Fine/Final Haiku calls only",
            "complete_system_cost_reporting": "historical selected Planner cost plus newly observed downstream cost, reported as separate components",
        },
        "config_path": str(CONFIG),
        "config_sha256": sha256_file(CONFIG),
        "fingerprinted_workspace_files": FINGERPRINT_FILES,
        "fingerprinted_legacy_files": [str(path) for path in legacy_files],
        "live_dependency_versions": dependencies,
        "runtime_fingerprint": runtime_fingerprint,
        "manifest_state": "paired100_frozen_prelaunch_no_api",
        "api_calls": 0,
        "model_calls": 0,
        "gold_loaded": False,
    }
    # Remove full-population-only proposal so the formal manifest has one clear population.
    manifest.pop("pilot_proposal", None)
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    atomic_json(PREFLIGHT / "paired100_manifest.json", manifest)
    atomic_json(PREFLIGHT / "selection_audit.json", manifest["paired100_selection"])
    atomic_json(PREFLIGHT / "planner_cost_summary.json", planner_cost)
    atomic_json(PREFLIGHT / "direct_r3_pairing.json", manifest["direct_r3_pairing"])
    legacy = legacy_downstream_config(cfg, preflight_root=PREFLIGHT)
    legacy["eval300_ordered_uid_path"] = str(PREFLIGHT / "paired100_question_ids.txt")
    atomic_json(PREFLIGHT / "legacy_downstream_config.json", legacy)
    atomic_text(PREFLIGHT / "paired100_question_ids.txt", "\n".join(selected) + "\n")

    output = Path(cfg.output_root)
    output.mkdir(parents=True, exist_ok=True)
    StagedStore(output, experiment_id=cfg.experiment_id, hard_budget_usd=cfg.hard_api_budget_usd,
                request_reserve_usd=cfg.per_request_budget_reserve_usd)
    status_count = len(list((output / "route_status").glob("*.json"))) if (output / "route_status").exists() else 0
    summary = {
        "status": "PASS_PRELAUNCH",
        "questions": len(selected),
        "videos": len(quota),
        "planner_artifacts": len(manifest["planner_reuse"]["rows"]),
        "direct_r3_links": len(direct_links),
        "runtime_fingerprint": runtime_fingerprint,
        "manifest_path": str(PREFLIGHT / "paired100_manifest.json"),
        "manifest_sha256": sha256_file(PREFLIGHT / "paired100_manifest.json"),
        "formal_output_namespace": str(output),
        "formal_ledger_spend_usd": "0",
        "formal_route_status_count": status_count,
        "api_calls": 0,
        "model_calls": 0,
        "gold_loaded": False,
    }
    atomic_json(PREFLIGHT / "prelaunch_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(materialise(), indent=2))
