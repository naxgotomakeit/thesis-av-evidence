#!/usr/bin/env python3
"""Prepare, validate, and explicitly launch the v8 175-route formal candidate.

The default mode is offline-only. A real launch requires all three explicit
confirmations in ``main``. The QA implementation is the byte-identical v7
implementation; only scope, namespace, and aggregate budget differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_V7 = Path("${PROJECT_MSC_ROOT}/direct_r1_visual_only_correction_v7_bounded_smoke")
SOURCE_V7_RUN = SOURCE_V7 / "runs/direct_r1_visual_only_correction_v7_smoke_1q_cumulative_budget_2_00"
SOURCE_V7_LEDGER = SOURCE_V7_RUN / "budget_ledger.json"
SOURCE_V7_LEDGER_SHA = "e1f5bb25e26e1eb93dad0a84f8815fbfb15ce80aca7f3791321aa707ecdfdf1c"
SOURCE_V7_MANIFEST = SOURCE_V7_RUN / "smoke_manifest_1q_cumulative_budget_2_00.json"
SOURCE_V7_MANIFEST_SHA = "a5c18e42647f1951109bb27dd7c8e7704b7af1a2e7d70110d339dc40fdaca031"
SOURCE_FORMAL_ROOT = Path("${PROJECT_MSC_ROOT}/direct_r1_visual_only_correction_v3_urllib_frozen")
SOURCE_FORMAL_MANIFEST = SOURCE_FORMAL_ROOT / "runs/direct_r1_visual_only_correction_v3_urllib_frozen/formal_manifest_visual_only_correction_v3_urllib_frozen.json"
SOURCE_FORMAL_MANIFEST_SHA = "d1a997294fc481360aa2790b50a67efdea53b4d5ec3a09ecf8b0c77c609eca92"
SOURCE_REUSE_MANIFEST = SOURCE_FORMAL_ROOT / "manifests/route_reuse_manifest_v3.json"

EXPERIMENT = "direct_r1_visual_only_correction_v8_formal_175"
NAMESPACE = ROOT / "runs" / EXPERIMENT
MANIFEST_PATH = NAMESPACE / "formal_manifest_visual_only_correction_v8.json"
LOCK_PATH = NAMESPACE / "formal_launch_lock_visual_only_correction_v8.json"
REUSE_PATH = ROOT / "manifests/route_reuse_manifest_v8.json"
TOTAL_CAP_USD = 15.0
INHERITED_SETTLED_USD = 0.082101
INHERITED_UNKNOWN_USD = 1.2628
INITIAL_AVAILABLE_USD = 13.655099
ROUTE_COUNT = 175

sys.path.insert(0, str(ROOT / "src"))
from direct_api_v1.anthropic_provider import AnthropicDirectAgent
from direct_api_v1.formal_runtime import FormalOrchestrator
from direct_api_v1.frame_resolver import FrozenFrameResolver
from direct_api_v1.maps import FORBIDDEN_QUESTION_KEYS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stable_write(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"refusing to overwrite non-identical frozen asset: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _assert_source_identity() -> None:
    expected = (
        (SOURCE_V7_LEDGER, SOURCE_V7_LEDGER_SHA),
        (SOURCE_V7_MANIFEST, SOURCE_V7_MANIFEST_SHA),
        (SOURCE_FORMAL_MANIFEST, SOURCE_FORMAL_MANIFEST_SHA),
    )
    for path, digest in expected:
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"source identity drift: {path}")
    for path in sorted((ROOT / "src/direct_api_v1").glob("*.py")):
        source = SOURCE_V7 / path.relative_to(ROOT)
        if not source.is_file() or sha(path) != sha(source):
            raise RuntimeError(f"QA implementation differs from v7: {path.relative_to(ROOT)}")
    if sha(ROOT / "config/direct_v1_anthropic_smoke.json") != sha(SOURCE_V7 / "config/direct_v1_anthropic_smoke.json"):
        raise RuntimeError("provider/QA config differs from v7")


def _materialize_candidate() -> tuple[dict, dict]:
    source = load(SOURCE_FORMAL_MANIFEST)
    source_v7 = load(SOURCE_V7_MANIFEST)
    videos = []
    for old in source["videos"]:
        video_id = old["video_id"]
        new_map = ROOT / f"maps/r1_visual_only/{video_id}/r1_visual_only_navigation_map.json"
        new_frame_manifest = ROOT / f"manifests/frame_sha256_current_snapshot/{video_id}.json"
        if sha(new_map) != old["r1_visual_only_sha256"]:
            raise RuntimeError(f"visual-only map drift: {video_id}")
        if sha(new_frame_manifest) != old["frame_sha_manifest_sha256"]:
            raise RuntimeError(f"frame manifest drift: {video_id}")
        video = dict(old)
        video["r1_visual_only_map"] = str(new_map)
        video["frame_sha_manifest"] = str(new_frame_manifest)
        videos.append(video)
    video_by_id = {row["video_id"]: row for row in videos}

    routes = []
    for old in source["routes"]:
        route = dict(old)
        route["route_id"] = "R1VO8:" + old["question_id"]
        route["map_path"] = video_by_id[old["video_id"]]["r1_visual_only_map"]
        route["map_sha256"] = video_by_id[old["video_id"]]["r1_visual_only_sha256"]
        routes.append(route)
    if len(routes) != ROUTE_COUNT or len({row["question_id"] for row in routes}) != ROUTE_COUNT:
        raise RuntimeError("formal route set is not 175 unique question IDs")

    reuse = load(SOURCE_REUSE_MANIFEST)
    reuse["schema_version"] = "direct_r1_visual_only_route_reuse_manifest_v8"
    reuse["rerun"]["routes"] = routes
    reuse["rerun"]["count"] = len(routes)
    reuse["formal_candidate_experiment_id"] = EXPERIMENT
    reuse["source_reuse_manifest"] = str(SOURCE_REUSE_MANIFEST)
    reuse["source_reuse_manifest_sha256"] = sha(SOURCE_REUSE_MANIFEST)

    config = load(ROOT / "config/direct_v1_anthropic_smoke.json")
    credential_path = Path(config["credential_env_path"])
    if not credential_path.is_file():
        raise RuntimeError("credential file missing")
    credential_sha = sha(credential_path)
    code_snapshot = {str(path.relative_to(ROOT)): sha(path)
                     for path in sorted((ROOT / "src/direct_api_v1").glob("*.py"))}
    protocol = dict(source_v7["protocol_identity"])
    protocol.update({
        "required_transport": "urllib_fallback",
        "credential_resolution": "file_only_sha_locked_no_environment_override",
        "provider_hash": code_snapshot["src/direct_api_v1/anthropic_provider.py"],
        "controller_hash": code_snapshot["src/direct_api_v1/controller.py"],
        "formal_runtime_hash": code_snapshot["src/direct_api_v1/formal_runtime.py"],
    })
    manifest = {
        "schema_version": "direct_r1_visual_only_correction_v8_formal_candidate_v1",
        "experiment_id": EXPERIMENT,
        "state": "prepared_not_started_no_api",
        "api_calls": 0,
        "model_calls": 0,
        "gold_loaded": False,
        "question_count": ROUTE_COUNT,
        "video_count": len(videos),
        "route_count": len(routes),
        "method": "R1",
        "concurrency": 1,
        "hard_budget_usd": TOTAL_CAP_USD,
        "inherited_settled_smoke_usd": INHERITED_SETTLED_USD,
        "inherited_unknown_liability_usd": INHERITED_UNKNOWN_USD,
        "initial_available_usd": INITIAL_AVAILABLE_USD,
        "credential_file_path": str(credential_path),
        "credential_file_sha256": credential_sha,
        "credential_resolution": "file_only_sha_locked_no_environment_override",
        "required_transport": "urllib_fallback",
        "source_v7_manifest_path": str(SOURCE_V7_MANIFEST),
        "source_v7_manifest_sha256": SOURCE_V7_MANIFEST_SHA,
        "source_v7_ledger_path": str(SOURCE_V7_LEDGER),
        "source_v7_ledger_sha256": SOURCE_V7_LEDGER_SHA,
        "source_formal_manifest_path": str(SOURCE_FORMAL_MANIFEST),
        "source_formal_manifest_sha256": SOURCE_FORMAL_MANIFEST_SHA,
        "reuse_manifest_path": str(REUSE_PATH),
        "protocol_identity": protocol,
        "qa_code_snapshot_sha256": code_snapshot,
        "routes": routes,
        "videos": videos,
    }
    return manifest, reuse


def _create_namespace(manifest: dict, reuse: dict) -> None:
    if NAMESPACE.exists():
        return
    NAMESPACE.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "journals", "route_status", "route_artifacts", "route_checkpoints",
                 "provider_responses", "request_payloads", "failure_diagnostics"):
        (NAMESPACE / name).mkdir()
    for name in ("request_start.jsonl", "attempt_end.jsonl", "controller_result.jsonl", "lifecycle.jsonl"):
        (NAMESPACE / "journals" / name).write_bytes(b"")
    source_inputs = ROOT / "runs/direct_r1_visual_only_correction_v1/inputs"
    for route in manifest["routes"]:
        source_input = source_inputs / f"{route['question_id']}.json"
        if not source_input.is_file() or sha(source_input) != route["question_sha256"]:
            raise RuntimeError(f"question input drift: {route['question_id']}")
        if set(load(source_input)) & FORBIDDEN_QUESTION_KEYS:
            raise RuntimeError(f"gold field found: {route['question_id']}")
        shutil.copy2(source_input, NAMESPACE / "inputs" / source_input.name)
    stable_write(REUSE_PATH, reuse)
    stable_write(MANIFEST_PATH, manifest)

    source_ledger = load(SOURCE_V7_LEDGER)
    attempts = {}
    for attempt_id, record in source_ledger["attempts"].items():
        inherited = dict(record)
        inherited["inherited_from_smoke_ledger"] = str(SOURCE_V7_LEDGER)
        inherited["source_ledger_sha256"] = SOURCE_V7_LEDGER_SHA
        if record.get("state") == "settled":
            inherited["state"] = "inherited_settled_smoke"
        else:
            inherited["state"] = "inherited_uncertain_liability"
        attempts[attempt_id] = inherited
    charges = [{**row, "inherited_from_smoke_ledger": str(SOURCE_V7_LEDGER)}
               for row in source_ledger.get("charges", [])]
    stable_write(NAMESPACE / "budget_ledger.json", {
        "schema_version": "reserved_budget_ledger_v2",
        "cap_usd": TOTAL_CAP_USD,
        "spent_usd": INHERITED_SETTLED_USD,
        "reserved_usd": INHERITED_UNKNOWN_USD,
        "attempt_ids": list(source_ledger.get("attempt_ids", [])),
        "attempts": attempts,
        "charges": charges,
        "inherited_accounting": {
            "source_ledger": str(SOURCE_V7_LEDGER),
            "source_ledger_sha256": SOURCE_V7_LEDGER_SHA,
            "settled_smoke_usd": INHERITED_SETTLED_USD,
            "unknown_liability_usd": INHERITED_UNKNOWN_USD,
        },
    })
    stable_write(LOCK_PATH, {
        "schema_version": "direct_r1_visual_only_correction_v8_formal_launch_lock_v1",
        "experiment_id": EXPERIMENT,
        "candidate_manifest_path": str(MANIFEST_PATH),
        "candidate_manifest_sha256": sha(MANIFEST_PATH),
        "runner_sha256": sha(Path(__file__).resolve()),
        "route_count": ROUTE_COUNT,
        "hard_budget_usd": TOTAL_CAP_USD,
        "initial_committed_usd": INHERITED_SETTLED_USD + INHERITED_UNKNOWN_USD,
        "initial_available_usd": INITIAL_AVAILABLE_USD,
        "credential_file_path": manifest["credential_file_path"],
        "credential_file_sha256": manifest["credential_file_sha256"],
        "credential_resolution": manifest["credential_resolution"],
        "required_transport": "urllib_fallback",
        "concurrency": 1,
        "real_execution_allowed": True,
        "launch_requires_exact_flags": ["--execute-real-correction", "--confirm-route-count 175",
                                         "--confirm-total-budget-usd 15.0"],
    })


def prepare_and_validate(*, allow_create: bool = True):
    _assert_source_identity()
    manifest, reuse = _materialize_candidate()
    if allow_create:
        _create_namespace(manifest, reuse)
    if not MANIFEST_PATH.is_file() or load(MANIFEST_PATH) != manifest:
        raise RuntimeError("frozen formal manifest drift")
    if not REUSE_PATH.is_file() or load(REUSE_PATH) != reuse:
        raise RuntimeError("frozen reuse manifest drift")
    lock = load(LOCK_PATH)
    if lock.get("candidate_manifest_sha256") != sha(MANIFEST_PATH):
        raise RuntimeError("formal manifest lock drift")
    if lock.get("runner_sha256") != sha(Path(__file__).resolve()):
        raise RuntimeError("formal runner lock drift")
    credential_path = Path(manifest["credential_file_path"])
    if not credential_path.is_file() or sha(credential_path) != manifest["credential_file_sha256"]:
        raise RuntimeError("credential file identity drift")
    ledger = load(NAMESPACE / "budget_ledger.json")
    committed = float(ledger["spent_usd"]) + float(ledger["reserved_usd"])
    if float(ledger["cap_usd"]) != TOTAL_CAP_USD or committed > TOTAL_CAP_USD:
        raise RuntimeError("latest aggregate ledger exceeds formal cap")
    return manifest, reuse, ledger


def build_orchestrator(manifest: dict, *, agent_class=AnthropicDirectAgent,
                       orchestrator_class=FormalOrchestrator):
    config = load(ROOT / "config/direct_v1_anthropic_smoke.json")
    video_by_id = {row["video_id"]: row for row in manifest["videos"]}

    def resolver_factory(route):
        video = video_by_id[route["video_id"]]
        return FrozenFrameResolver.from_hierarchy(load(Path(video["hierarchy"])),
                                                   load(Path(video["frame_sha_manifest"])))

    def agent_factory(route, lifecycle, state):
        latest = load(NAMESPACE / "budget_ledger.json")
        available = float(latest["cap_usd"]) - float(latest["spent_usd"]) - float(latest["reserved_usd"])
        if available <= 0:
            raise RuntimeError("no aggregate correction budget remains")
        return agent_class(
            credential_env_path=Path(config["credential_env_path"]),
            model=config["model"],
            pricing=config["anthropic_cache_aware_pricing_usd_per_million_tokens"],
            max_output_tokens=config["max_output_tokens"],
            timeout_sec=config["timeout_sec"],
            max_retries=config["max_retries"],
            max_total_usd=available,
            max_billable_input_tokens=200000,
            lifecycle=lifecycle,
            transport_mode="urllib_fallback",
            credential_file_sha256=manifest["credential_file_sha256"],
            require_file_credential=True,
        )

    return orchestrator_class(manifest_path=MANIFEST_PATH, namespace=NAMESPACE,
                              agent_factory=agent_factory, resolver_factory=resolver_factory,
                              cap_usd=TOTAL_CAP_USD)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-real-correction", action="store_true")
    parser.add_argument("--confirm-route-count", type=int)
    parser.add_argument("--confirm-total-budget-usd", type=float)
    args = parser.parse_args()
    manifest, reuse, ledger = prepare_and_validate(allow_create=True)
    available = float(ledger["cap_usd"]) - float(ledger["spent_usd"]) - float(ledger["reserved_usd"])
    if not args.execute_real_correction:
        print(json.dumps({
            "valid": True, "mode": "offline_preflight_no_api", "real_api_calls": 0,
            "experiment_id": EXPERIMENT, "route_count": len(manifest["routes"]),
            "namespace": str(NAMESPACE), "manifest": str(MANIFEST_PATH),
            "required_transport": "urllib_fallback", "credential_file_identity_locked": True,
            "cap_usd": TOTAL_CAP_USD, "spent_usd": ledger["spent_usd"],
            "reserved_usd": ledger["reserved_usd"], "available_usd": available,
            "merge_partition": {"new_r1": 175, "reused_r1": 125, "reused_r3": 300},
        }, indent=2))
        return
    if args.confirm_route_count != ROUTE_COUNT or args.confirm_total_budget_usd != TOTAL_CAP_USD:
        raise SystemExit("real launch requires exact 175-route/$15 confirmations")
    print(json.dumps(build_orchestrator(manifest).execute(), indent=2))


if __name__ == "__main__":
    main()
