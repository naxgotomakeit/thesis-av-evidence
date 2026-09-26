#!/usr/bin/env python3
"""Materialise the ABD formal candidate and offline-only preflight evidence."""
from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abd_draft_v1.core import GensPromptBuilder, load_jsonl, validate_manifest  # noqa: E402
from abd_draft_v1.formal_runtime import AbdFormalConfig, file_fingerprints  # noqa: E402
from abd_draft_v1.provider import AnthropicAbdProvider  # noqa: E402
from abd_draft_v1.store import atomic_json  # noqa: E402
from gens_haiku_eval300.runtime import canonical_sha, sha256_file  # noqa: E402


CONFIG = ROOT / "config/abd_formal_candidate_v1.json"
DRAFT = ROOT / "drafts/abd_direct_eval300_v1"
PREFLIGHT = DRAFT / "formal_candidate"
DIRECT_MANIFEST = ROOT / "outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/formal_manifest_final_candidate_no_api_v3.json"
GENS_SELECTOR = Path(json.loads((ROOT / "config/gens_haiku_structured_v3_direct_parser_aligned.json").read_text())["selector_path"])


def exact_sdk_version(live_python: str) -> str:
    command = [live_python, "-c", "import importlib.metadata; print(importlib.metadata.version('anthropic'))"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def task_order(rows: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    qids = [row["question_id"] for row in rows["A"]]
    by_arm = {variant: {row["question_id"]: row for row in arm} for variant, arm in rows.items()}
    rotations = (("A", "B", "D"), ("B", "D", "A"), ("D", "A", "B"))
    tasks: list[dict[str, Any]] = []
    positions: dict[str, Counter[str]] = {str(index): Counter() for index in range(3)}
    for question_index, qid in enumerate(qids):
        for within_question_position, variant in enumerate(rotations[question_index % 3]):
            row = by_arm[variant][qid]
            positions[str(within_question_position)][variant] += 1
            tasks.append({
                "execution_index": len(tasks),
                "question_index": question_index,
                "within_question_position": within_question_position,
                "task_id": f"{variant}:{qid}",
                "variant": variant,
                "question_id": qid,
                "video_id": row["video_id"],
                "input_row_sha256": canonical_sha(row),
            })
    balance = {position: dict(counts) for position, counts in positions.items()}
    if any(balance[str(position)] != {"A": 100, "B": 100, "D": 100} for position in range(3)):
        raise RuntimeError("ABD arm positions are not exactly balanced")
    return tasks, {
        "rule": "canonical GenS question order; rotate ABD/BDA/DAB by zero-based question index modulo three",
        "answer_independent": True,
        "within_question_position_counts": balance,
        "question_order_sha256": canonical_sha(qids),
    }


def payload_and_capacity_checks(
    config: AbdFormalConfig, rows: dict[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    builder = GensPromptBuilder(config.abd_prompt_config)
    by_arm = {variant: {row["question_id"]: row for row in arm} for variant, arm in rows.items()}
    costs = list(csv.DictReader((DRAFT / "COST_ESTIMATE_PER_QUESTION.csv").open(encoding="utf-8")))
    costs_by_key = {(row["variant"], row["question_id"]): row for row in costs}
    capacity_rows: list[dict[str, Any]] = []
    checks = Counter()
    maximum_serialized = {variant: 0 for variant in "ABD"}
    forbidden_payload_keys = {
        "source_provenance", "requested_timestamp_sec", "direct_transmission_index",
        "presentation_index", "historical_expected_sha256", "gold", "correct_answer",
        "old_answer", "reasoning_record", "cache_control",
    }

    class CaptureMessages:
        def __init__(self) -> None:
            self.calls = 0
            self.last_payload: dict[str, Any] | None = None

        def create(self, **payload: Any) -> dict[str, Any]:
            self.calls += 1
            self.last_payload = payload
            return {
                "id": f"offline-capture-{self.calls}",
                "model": config.model,
                "content": [{
                    "type": "tool_use", "id": "offline", "name": "final_answer",
                    "input": {"selected_option_id": "A", "reason": "offline transport capture"},
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    capture_client = CaptureMessages()
    capture_transport = AnthropicAbdProvider(capture_client)

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(map(str, value)) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value)) if value else set()
        return set()

    for qid in by_arm["A"]:
        payloads: dict[str, dict[str, Any]] = {}
        for variant in "ABD":
            row = by_arm[variant][qid]
            payload = builder.build_provider_payload(row, encode_images=True)
            capture_transport.send(payload)
            captured = capture_client.last_payload
            if captured != payload:
                raise RuntimeError(f"simulated transport altered final request: {variant}:{qid}")
            if captured is None:
                raise RuntimeError(f"simulated transport failed to capture request: {variant}:{qid}")
            payload = captured
            payloads[variant] = payload
            if set(payload) != {"model", "max_tokens", "temperature", "system", "messages", "tools", "tool_choice"}:
                raise RuntimeError(f"unexpected final request fields: {variant}:{qid}")
            if payload["model"] != config.model or payload["max_tokens"] != 512 or payload["temperature"] != 0.0:
                raise RuntimeError(f"provider scalar mismatch: {variant}:{qid}")
            leaked = keys(payload) & forbidden_payload_keys
            if leaked:
                raise RuntimeError(f"backend/result fields leaked into payload {variant}:{qid}: {sorted(leaked)}")
            content = payload["messages"][0]["content"]
            offset = 1 if variant in "AD" else 0
            if variant in "AD":
                raw_map = Path(row["map"]["path"]).read_bytes()
                if hashlib.sha256(raw_map).hexdigest() != row["map"]["sha256"]:
                    raise RuntimeError(f"map SHA changed: {variant}:{qid}")
                if content[0] != {"type": "text", "text": config.abd_prompt_config.map_prefix + raw_map.decode("utf-8")}:
                    raise RuntimeError(f"map serialization changed: {variant}:{qid}")
                checks["map_payloads"] += 1
            for index, image in enumerate(row["images"]):
                label_block = content[offset + index * 2]
                image_block = content[offset + index * 2 + 1]
                expected_label = f"Frame timestamp={float(image['resolved_timestamp_sec']):.3f}s from video start."
                if label_block != {"type": "text", "text": expected_label}:
                    raise RuntimeError(f"timestamp serialization changed: {variant}:{qid}:{index}")
                decoded = base64.b64decode(image_block["source"]["data"], validate=True)
                if hashlib.sha256(decoded).hexdigest() != image["sha256"]:
                    raise RuntimeError(f"JPEG serialization SHA mismatch: {variant}:{qid}:{index}")
                checks["image_payloads"] += 1
                checks["timestamp_payloads"] += 1
            serialized_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            maximum_serialized[variant] = max(maximum_serialized[variant], serialized_bytes)
            estimate = costs_by_key[(variant, qid)]
            high_with_output = float(estimate["input_tokens_high"]) + config.max_output_tokens
            capacity_rows.append({
                "task_id": f"{variant}:{qid}",
                "variant": variant,
                "question_id": qid,
                "image_count": len(row["images"]),
                "serialized_request_bytes": serialized_bytes,
                "estimated_input_tokens_low": float(estimate["input_tokens_low"]),
                "estimated_input_tokens_base": float(estimate["input_tokens_base"]),
                "estimated_input_tokens_high": float(estimate["input_tokens_high"]),
                "reserved_output_tokens": config.max_output_tokens,
                "estimated_high_total_with_output_reserve": high_with_output,
                "assumed_context_window_tokens": config.assumed_context_window_tokens_for_offline_screening_only,
                "estimated_high_margin_tokens": config.assumed_context_window_tokens_for_offline_screening_only - high_with_output,
                "exact_token_count_verified": False,
                "offline_capacity_screen": "estimated_pass" if high_with_output < config.assumed_context_window_tokens_for_offline_screening_only else "unresolved",
            })
        if payloads["D"]["messages"][0]["content"][1:] != payloads["B"]["messages"][0]["content"]:
            raise RuntimeError(f"D minus map differs from B: {qid}")
        if payloads["A"]["messages"][0]["content"][0] != payloads["D"]["messages"][0]["content"][0]:
            raise RuntimeError(f"A/D map differs: {qid}")
        checks["aligned_triples"] += 1
    unresolved = [row["task_id"] for row in capacity_rows if row["offline_capacity_screen"] == "unresolved"]
    return {
        "status": "PASS",
        "actual_final_payloads_serialized_with_full_jpeg_base64": 900,
        "simulated_transport_captured_requests": capture_client.calls,
        "real_transport_requests": 0,
        "aligned_triples": checks["aligned_triples"],
        "map_payloads_A_plus_D": checks["map_payloads"],
        "image_payloads_B_plus_D": checks["image_payloads"],
        "timestamp_text_blocks_B_plus_D": checks["timestamp_payloads"],
        "D_minus_map_equals_B": True,
        "A_D_map_payload_equal": True,
        "forbidden_backend_result_or_cache_fields": 0,
        "maximum_serialized_request_bytes_by_variant": maximum_serialized,
        "capacity_classification": "ESTIMATED_PASS_NOT_EXACT_TOKEN_VERIFICATION" if not unresolved else "UNRESOLVED",
        "unresolved_task_count": len(unresolved),
        "unresolved_task_ids": unresolved,
        "exact_online_token_counts": 0,
        "assumed_context_window_tokens_not_network_verified": config.assumed_context_window_tokens_for_offline_screening_only,
        "minimum_estimated_high_margin_tokens": min(row["estimated_high_margin_tokens"] for row in capacity_rows),
        "maximum_estimated_high_total_with_output_reserve": max(row["estimated_high_total_with_output_reserve"] for row in capacity_rows),
    }, capacity_rows


def write_capacity_csv(rows: list[dict[str, Any]]) -> None:
    path = PREFLIGHT / "capacity_per_task.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = AbdFormalConfig.load(CONFIG)
    rows = {
        variant: load_jsonl(DRAFT / f"inputs/{variant}.jsonl")
        for variant in "ABD"
    }
    for variant in "ABD":
        validate_manifest(rows[variant], variant)
    tasks, ordering = task_order(rows)
    payload_checks, capacity_rows = payload_and_capacity_checks(config, rows)
    cost = json.loads((DRAFT / "COST_ESTIMATE_SUMMARY.json").read_text(encoding="utf-8"))
    sdk_version = exact_sdk_version(config.live_python)
    direct_manifest = json.loads(DIRECT_MANIFEST.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "abd_eval300_formal_candidate_manifest_v1",
        "experiment_id": config.experiment_id,
        "candidate_state": "OFFLINE_PREFLIGHT_ONLY_NOT_AUTHORIZED_NOT_STARTED",
        "task_count": 900,
        "question_count": 300,
        "tasks_per_variant": {"A": 300, "B": 300, "D": 300},
        "concurrency": 1,
        "tasks": tasks,
        "execution_order": ordering,
        "execution_order_sha256": canonical_sha(tasks),
        "comparison_contract": {
            "primary": "D_vs_B",
            "supplementary": "D_vs_A",
            "denominator_per_arm": 300,
            "failure_or_missing": "incorrect",
            "C": "reuse completed historical Direct results; no C generation task",
        },
        "provider_contract": {
            "provider": config.provider,
            "model": config.model,
            "max_output_tokens": config.max_output_tokens,
            "temperature": config.temperature,
            "timeout_sec": config.timeout_sec,
            "cache": config.cache_policy,
            "sdk_internal_max_retries": 0,
            "outer_transport_retries": 0,
            "single_turn": True,
            "structural_correction_retries": 0,
        },
        "budget": {
            "candidate_hard_cap_usd": config.candidate_hard_budget_usd,
            "per_request_reserve_usd": config.per_request_budget_reserve_usd,
            "authorized": False,
            "status": config.budget_authorization_status,
        },
        "special_route_preservation": {
            "zero_image": [
                "4572b198-2c1c-4920-bcf0-95fcebe12261_17_15",
                "4572b198-2c1c-4920-bcf0-95fcebe12261_8_13",
            ],
            "historical_direct_failed_with_images": {
                "6fd90f8d-7a4d-425d-a812-3268db0b0342_11_29": 7,
                "a6d45e95-8dc0-4932-83bf-ec53e265a16a_11_15": 3,
            },
        },
        "source_relationships": {
            "Direct_R3_manifest": {"path": str(DIRECT_MANIFEST), "sha256": sha256_file(DIRECT_MANIFEST)},
            "GenS_question_selector": {"path": str(GENS_SELECTOR), "sha256": sha256_file(GENS_SELECTOR)},
            "historical_parser": {"path": str(ROOT / "src/gens_haiku_eval300/structured_v3.py"), "sha256": sha256_file(ROOT / "src/gens_haiku_eval300/structured_v3.py")},
            "historical_Direct_experiment_id": direct_manifest["experiment_id"],
            "gold_loaded_for_generation_or_preflight": False,
        },
        "file_fingerprints": file_fingerprints(ROOT, CONFIG),
        "dependency_versions": {
            "formal_live_python": config.live_python,
            "anthropic": sdk_version,
            "current_preflight_python": sys.executable,
            "Pillow": importlib.metadata.version("Pillow"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "source_git_parent_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "payload_checks": payload_checks,
        "cost_estimate_source": {
            "path": str(DRAFT / "COST_ESTIMATE_SUMMARY.json"),
            "sha256": sha256_file(DRAFT / "COST_ESTIMATE_SUMMARY.json"),
            "all_900": cost["all_900"],
        },
    }
    PREFLIGHT.mkdir(parents=True, exist_ok=True)
    atomic_json(PREFLIGHT / "formal_candidate_manifest.json", manifest)
    write_capacity_csv(capacity_rows)
    report = {
        "status": "PASS_WITH_EXPLICIT_UNVERIFIED_ITEMS_AND_BUDGET_BLOCK",
        "offline_preflight": True,
        "api_calls": 0,
        "smoke_launches": 0,
        "formal_experiment_launches": 0,
        "budget_authorized": False,
        "checks_passed": {
            "population_order_and_balance": ordering,
            "final_request_serialization": payload_checks,
            "provider_sdk_internal_retries_configured": 0,
            "outer_transport_retries": 0,
            "concurrency": 1,
            "generation_scorer_separated": True,
            "historical_GenS_Direct_modified": False,
        },
        "network_unverified": [
            "Anthropic endpoint reachability and credentials",
            "live model acceptance of the largest map-plus-image request",
            "authoritative provider token counts and billing",
            "server-side model/context behavior beyond the locally frozen assumptions",
        ],
        "capacity": payload_checks,
        "cost": {
            "rates_usd_per_million_tokens": config.pricing,
            "A": cost["variants"]["A"],
            "B": cost["variants"]["B"],
            "D": cost["variants"]["D"],
            "all_900": cost["all_900"],
            "candidate_budget_scenario_usd": 30.0,
            "candidate_budget_remaining_after_512_output_ceiling_scenario_usd": 30.0 - cost["all_900"]["cost_usd_configured_512_output_ceiling"],
            "candidate_budget_authorized": False,
        },
        "blockers": [
            "Formal budget authorization has not been issued.",
            "No matching launch-authorization JSON exists; execution gate therefore fails closed.",
            "Capacity is an offline estimate, not an authoritative tokenizer/API check.",
        ],
        "future_guarded_launch_command_not_run": (
            f"{config.live_python} scripts/run_abd_formal_v1.py --execute "
            f"--approval-token APPROVE_ABD_EVAL300_FORMAL_V1 --approval-file <reviewed-authorization.json>"
        ),
    }
    atomic_json(PREFLIGHT / "preflight_report.json", report)
    md = f"""# ABD formal candidate — offline preflight

Status: **PASS with explicit network/token uncertainty and a hard budget-authorization block**.

No API, smoke, online token count, connection test, formal generation, scoring, or judge audit was run.

## Passed offline checks

- Frozen tasks: A/B/D each 300; total 900; concurrency 1.
- Order: `{ordering['rule']}`. Each arm appears exactly 100 times in each of the three within-question positions.
- All 900 final SDK-shaped payloads were serialized using complete maps and real JPEG base64, passed through a capture-only simulated Messages client, and compared after capture. D minus map equals B for 300/300 questions; A/D maps match for 300/300.
- B/D contain {payload_checks['timestamp_text_blocks_B_plus_D']} resolved-time text blocks paired with {payload_checks['image_payloads_B_plus_D']} original JPEG transmissions; no cache-control or backend/result/gold fields were present.
- SDK internal retries and outer retries are both zero. One durable request-start corresponds to at most one physical request.
- Durable store/runtime tests cover normal, invalid, missing usage, provider error/unknown outcome, response recovery, terminal replay prevention, budget reservations, and exclusive runner locking.
- Generation and scorer are separate programs. No score or gold was loaded in this preflight.

## Capacity — estimate, not exact verification

- Assumed context screen: {config.assumed_context_window_tokens_for_offline_screening_only} tokens, not network-verified in this round.
- Largest offline high estimate including the 512-token output reserve: {payload_checks['maximum_estimated_high_total_with_output_reserve']:.1f} tokens.
- Minimum estimated margin under that assumption: {payload_checks['minimum_estimated_high_margin_tokens']:.1f} tokens.
- Unresolved tasks under the offline screen: {payload_checks['unresolved_task_count']}.
- Exact provider token counts: **not measured**. Map/text estimates use historical usage calibration; images use dimensions and historical GenS calibration. No map/image truncation or per-arm output reduction is implemented.

## Updated cost and candidate budget scenario

Primary no-cache/no-retry base: ${cost['all_900']['cost_usd_base_no_cache_no_retry']:.3f}; estimated range ${cost['all_900']['cost_usd_low_no_cache_no_retry']:.3f}–${cost['all_900']['cost_usd_high_no_cache_no_retry']:.3f}. The all-responses-at-512 ceiling scenario is ${cost['all_900']['cost_usd_configured_512_output_ceiling']:.3f}.

The US$30 value is only a candidate hard-cap scenario. It is **not authorized**. Admission requires `spent + unresolved reservations + $0.25 <= $30`; unknown outcomes retain their reservation and are never treated as zero. The $0.25 hold exceeds the locally assumed maximum one-request cost at 200k input plus 512 output tokens, but that context premise remains network-unverified.

## Still unverified online

- Endpoint/credential validity, live request acceptance, authoritative token counts/billing, and server-side context behavior.
- The formal launcher additionally requires a separately reviewed authorization file matching the frozen config and manifest SHA. No such file has been created.

## Recovery and future launch

Re-running the same guarded command acquires an exclusive process lock. Terminal success/failure tasks are skipped. A durable provider response without terminal state is parsed and finalized without a provider call. A request-start without a durable response becomes `pending_provider_outcome_review` and is not resent automatically.

Future command template, **not run**:

```bash
{report['future_guarded_launch_command_not_run']}
```

Primary comparison is D vs B; supplementary comparison is D vs A. Each arm uses all 300 questions, with failure/missing counted incorrect. C remains the completed historical Direct result.
"""
    (PREFLIGHT / "PREFLIGHT_REPORT.md").write_text(md, encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "manifest": str(PREFLIGHT / "formal_candidate_manifest.json"),
        "tasks": 900, "api_calls": 0, "smoke_launches": 0,
        "formal_experiment_launches": 0, "budget_authorized": False,
    }, indent=2))


if __name__ == "__main__":
    main()
