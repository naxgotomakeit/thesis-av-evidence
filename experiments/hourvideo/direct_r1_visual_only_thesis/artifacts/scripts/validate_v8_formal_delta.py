#!/usr/bin/env python3
"""Offline delta-only gate for the v8 175-route formal candidate."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_direct_r1_visual_only_correction_v8_formal as runner
from direct_api_v1.formal_runtime import NamespaceLockError, NamespaceWriterLock
from direct_api_v1.maps import FORBIDDEN_QUESTION_KEYS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


class CaptureAgent:
    kwargs = None
    def __init__(self, **kwargs):
        CaptureAgent.kwargs = kwargs


def lock_test() -> dict:
    with tempfile.TemporaryDirectory(prefix="r1vo8-lock-test-") as temporary:
        root = Path(temporary)
        rejected = False
        with NamespaceWriterLock(root):
            try:
                with NamespaceWriterLock(root):
                    pass
            except NamespaceLockError:
                rejected = True
        return {"second_writer_rejected": rejected, "lock_removed_after_exit": not (root / ".writer.lock").exists()}


def main() -> None:
    manifest, reuse, ledger = runner.prepare_and_validate(allow_create=True)
    source_manifest = load(runner.SOURCE_FORMAL_MANIFEST)
    source_reuse = load(runner.SOURCE_REUSE_MANIFEST)
    source_v7_manifest = load(runner.SOURCE_V7_MANIFEST)

    routes = manifest["routes"]
    qids = {row["question_id"] for row in routes}
    source_qids = {row["question_id"] for row in source_manifest["routes"]}
    inputs = []
    gold_free = True
    for route in routes:
        path = runner.NAMESPACE / "inputs" / f"{route['question_id']}.json"
        value = load(path)
        gold_free &= not bool(set(value) & FORBIDDEN_QUESTION_KEYS)
        inputs.append(sha(path) == route["question_sha256"])

    maps = []
    frames = []
    for video in manifest["videos"]:
        map_path = Path(video["r1_visual_only_map"])
        map_value = load(map_path)
        v7_map = runner.SOURCE_V7 / map_path.relative_to(ROOT)
        maps.append({
            "video_id": video["video_id"],
            "sha_matches_manifest": sha(map_path) == video["r1_visual_only_sha256"],
            "byte_identical_to_v7": sha(map_path) == sha(v7_map),
            "all_audio_channels_empty": all(region.get("audio_channel") == []
                                              for region in map_value.get("coarse_regions", [])),
        })
        frame_path = Path(video["frame_sha_manifest"])
        frame_value = load(frame_path)
        v7_frame = runner.SOURCE_V7 / frame_path.relative_to(ROOT)
        frames.append({
            "video_id": video["video_id"],
            "manifest_sha_matches": sha(frame_path) == video["frame_sha_manifest_sha256"],
            "byte_identical_to_v7": sha(frame_path) == sha(v7_frame),
            "frame_count": len(frame_value["frames"]),
            "send_time_sha_check_required": True,
            "evidence_scope": frame_value["source_evidence"],
        })

    rerun_qids = {row["question_id"] for row in reuse["rerun"]["routes"]}
    reused_r1_qids = {row["question_id"] for row in reuse["reuse_r1"]["routes"]}
    reused_r3_qids = {row["question_id"] for row in reuse["reuse_r3"]["routes"]}
    reuse_groups_unchanged = (
        canonical(reuse["reuse_r1"]) == canonical(source_reuse["reuse_r1"])
        and canonical(reuse["reuse_r3"]) == canonical(source_reuse["reuse_r3"])
    )

    current_run_files = {
        "statuses": len(list((runner.NAMESPACE / "route_status").glob("*.json"))),
        "artifacts": len(list((runner.NAMESPACE / "route_artifacts").glob("*.json"))),
        "responses": len(list((runner.NAMESPACE / "provider_responses").glob("*.json"))),
        "requests": len(list((runner.NAMESPACE / "request_payloads").glob("*.json"))),
        "journal_rows": sum(len(path.read_text().splitlines())
                            for path in (runner.NAMESPACE / "journals").glob("*.jsonl")),
    }
    smoke_route_ids = {"R1VO7SMOKE:" + source_v7_manifest["routes"][0]["question_id"]}
    formal_route_ids = {row["route_id"] for row in routes}

    orchestrator = runner.build_orchestrator(manifest, agent_class=CaptureAgent)
    orchestrator.agent_factory(routes[0], object(), object())
    kwargs = CaptureAgent.kwargs
    v7_config = load(runner.SOURCE_V7 / "config/direct_v1_anthropic_smoke.json")
    qa_params = {
        "model": kwargs["model"], "max_output_tokens": kwargs["max_output_tokens"],
        "timeout_sec": kwargs["timeout_sec"], "max_retries": kwargs["max_retries"],
        "max_billable_input_tokens": kwargs["max_billable_input_tokens"],
        "transport_mode": kwargs["transport_mode"],
        "require_file_credential": kwargs["require_file_credential"],
        "max_total_usd_from_latest_ledger": kwargs["max_total_usd"],
    }
    direct_hashes = {str(path.relative_to(ROOT)): sha(path)
                     for path in sorted((ROOT / "src/direct_api_v1").glob("*.py"))}
    source_direct_hashes = {str(path.relative_to(runner.SOURCE_V7)): sha(path)
                            for path in sorted((runner.SOURCE_V7 / "src/direct_api_v1").glob("*.py"))}
    lock_result = lock_test()
    committed = float(ledger["spent_usd"]) + float(ledger["reserved_usd"])
    available = float(ledger["cap_usd"]) - committed
    inherited_states = {}
    for row in ledger["attempts"].values():
        inherited_states[row["state"]] = inherited_states.get(row["state"], 0) + 1

    checks = {
        "v7_QA_code_byte_identical": direct_hashes == source_direct_hashes,
        "v7_provider_config_byte_identical": sha(ROOT / "config/direct_v1_anthropic_smoke.json")
            == sha(runner.SOURCE_V7 / "config/direct_v1_anthropic_smoke.json"),
        "formal_scope_exact_175": len(routes) == 175 and len(qids) == 175 and qids == source_qids,
        "formal_inputs_frozen_and_gold_free": all(inputs) and gold_free,
        "visual_only_maps_frozen": all(row["sha_matches_manifest"] and row["byte_identical_to_v7"]
                                        and row["all_audio_channels_empty"] for row in maps),
        "frame_manifests_frozen": all(row["manifest_sha_matches"] and row["byte_identical_to_v7"]
                                       for row in frames),
        "merge_partition_175_plus_125_and_paired300": len(rerun_qids) == 175
            and len(reused_r1_qids) == 125 and len(reused_r3_qids) == 300
            and not (rerun_qids & reused_r1_qids)
            and rerun_qids | reused_r1_qids == reused_r3_qids,
        "reuse_groups_unchanged_from_validated_v3": reuse_groups_unchanged,
        "formal_namespace_clean": all(value == 0 for value in current_run_files.values()),
        "smoke_not_formal_result": not (smoke_route_ids & formal_route_ids)
            and not any("SMOKE" in route_id for route_id in formal_route_ids),
        "writer_lock_effective": all(lock_result.values()),
        "recovery_and_accounting_code_unchanged_from_v7": direct_hashes == source_direct_hashes,
        "credential_file_identity_locked": kwargs["require_file_credential"] is True
            and kwargs["credential_file_sha256"] == manifest["credential_file_sha256"]
            and sha(Path(manifest["credential_file_path"])) == manifest["credential_file_sha256"],
        "urllib_forced": kwargs["transport_mode"] == "urllib_fallback",
        "QA_scalar_protocol_unchanged": kwargs["model"] == v7_config["model"]
            and kwargs["max_output_tokens"] == v7_config["max_output_tokens"]
            and kwargs["timeout_sec"] == v7_config["timeout_sec"]
            and kwargs["max_retries"] == v7_config["max_retries"]
            and manifest["protocol_identity"]["max_new_images_per_turn"] == 3
            and manifest["protocol_identity"]["max_unique_images_per_question"] == 16
            and manifest["protocol_identity"]["max_route_turns"] == 32,
        "aggregate_budget_carried": abs(float(ledger["cap_usd"]) - 15.0) < 1e-12
            and abs(float(ledger["spent_usd"]) - 0.082101) < 1e-12
            and abs(float(ledger["reserved_usd"]) - 1.2628) < 1e-12
            and abs(available - 13.655099) < 1e-12
            and inherited_states == {"inherited_settled_smoke": 7,
                                     "inherited_uncertain_liability": 5}
            and abs(kwargs["max_total_usd"] - available) < 1e-12,
        "manifest_and_runner_locked": load(runner.LOCK_PATH)["candidate_manifest_sha256"] == sha(runner.MANIFEST_PATH)
            and load(runner.LOCK_PATH)["runner_sha256"] == sha(Path(runner.__file__)),
    }
    result = {
        "schema_version": "direct_r1_visual_only_correction_v8_formal_delta_validation_v1",
        "real_api_calls": 0, "gold_read": False,
        "baseline": "v7 real-smoke PASS; unaffected evidence reused by byte-identical hashes",
        "checks": checks,
        "formal_namespace": str(runner.NAMESPACE),
        "formal_namespace_counts": current_run_files,
        "route_counts": {"new_r1": len(rerun_qids), "reused_r1": len(reused_r1_qids),
                         "reused_r3": len(reused_r3_qids)},
        "map_checks": maps, "frame_manifest_checks": frames,
        "lock_test": lock_result, "actual_agent_factory": qa_params,
        "budget": {"cap_usd": ledger["cap_usd"], "settled_smoke_usd": ledger["spent_usd"],
                   "historical_unknown_reserved_usd": ledger["reserved_usd"],
                   "committed_usd": committed, "available_usd": available,
                   "inherited_attempt_states": inherited_states},
        "allowed_differences_from_v7": [
            "scope: fixed 1-question smoke -> frozen 175-question correction partition",
            "namespace and route prefix: isolated formal R1VO8 namespace",
            "budget: cumulative smoke cap $2 -> correction cap $15 with liabilities carried",
            "agent max_total_usd: recalculated from latest aggregate ledger before each route",
            "formal manifest/launch lock/reuse manifest and input copies",
        ],
        "QA_differences_from_v7": [],
        "evidence_limitations": [
            "frame manifests are current frozen snapshots, not complete historical frame SHA freezes",
            "server-side model identity cannot be proven byte-identical across time",
        ],
        "pass": all(checks.values()),
        "acceptance": "PASS" if all(checks.values()) else "FAIL",
    }
    output = ROOT / "validation/v8_formal_delta_validation.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
