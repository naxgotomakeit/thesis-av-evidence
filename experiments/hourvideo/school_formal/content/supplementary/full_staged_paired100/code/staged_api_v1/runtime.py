"""Route-boundary-safe orchestration for Full Staged API."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .config import StagedConfig, canonical_sha, sha256_file
from .legacy_bridge import LegacyV662R3Executor, route_input_sha
from .preflight import legacy_dependency_files, live_dependency_versions, runtime_fingerprint_payload
from .provider import AnthropicStagedProvider
from .store import BudgetExceeded, StagedStore, TERMINAL_STATES


APPROVAL_TOKEN = "APPROVE_FULL_STAGED_R3_PILOT5_V1"


def verify_manifest(config: StagedConfig, config_path: Path, manifest_path: Path, workspace_root: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != config.experiment_id or manifest.get("config_sha256") != sha256_file(config_path):
        raise RuntimeError("Full Staged manifest/config fingerprint mismatch")
    expected = manifest.get("runtime_fingerprint")
    recorded_legacy = [Path(path) for path in manifest.get("fingerprinted_legacy_files", [])]
    current_legacy = legacy_dependency_files(config)
    if [str(path) for path in current_legacy] != [str(path) for path in recorded_legacy]:
        raise RuntimeError("Full Staged legacy dynamic-import graph mismatch")
    current_dependencies = live_dependency_versions(config)
    if current_dependencies != manifest.get("live_dependency_versions"):
        raise RuntimeError("Full Staged live dependency version mismatch")
    observed = canonical_sha(runtime_fingerprint_payload(
        config_path=config_path, workspace_root=workspace_root,
        workspace_files=list(manifest["fingerprinted_workspace_files"]),
        legacy_files=current_legacy, dependency_versions=current_dependencies,
    ))
    if observed != expected:
        raise RuntimeError("Full Staged runtime fingerprint mismatch")
    for source in manifest.get("frozen_population_sources", {}).values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise RuntimeError("Full Staged population source fingerprint mismatch")
    for row in manifest["planner_reuse"]["rows"]:
        if sha256_file(Path(row["planner_path"])) != row["planner_sha256"]:
            raise RuntimeError("Full Staged frozen Planner fingerprint mismatch")
    for assets in manifest["assets"].values():
        for name, record in assets.items():
            if name == "fine_frame_count":
                continue
            if sha256_file(Path(record["path"])) != record["sha256"]:
                raise RuntimeError("Full Staged scientific asset fingerprint mismatch")
    return manifest


class FullStagedRunner:
    def __init__(self, *, config: StagedConfig, config_path: Path, manifest_path: Path, workspace_root: Path,
                 store: StagedStore, provider_factory: Callable[[], AnthropicStagedProvider], legacy_config_path: Path,
                 executor_factory: Callable[..., Any] = LegacyV662R3Executor) -> None:
        self.config, self.config_path, self.manifest_path = config, config_path, manifest_path
        self.workspace_root, self.store = workspace_root, store
        self.provider_factory, self.legacy_config_path = provider_factory, legacy_config_path
        self.executor_factory = executor_factory

    def reconcile(self) -> dict[str, int]:
        manifest = verify_manifest(self.config, self.config_path, self.manifest_path, self.workspace_root)
        return self.store.reconcile_interrupted(list(manifest["ordered_question_ids"]))

    def run_one(self, question_id: str) -> dict[str, Any]:
        manifest = verify_manifest(self.config, self.config_path, self.manifest_path, self.workspace_root)
        by_id = {row["question_id"]: row for row in manifest["planner_reuse"]["rows"]}
        if question_id not in by_id:
            raise RuntimeError("question is outside frozen Full Staged population")
        prior = self.store.status(question_id)
        if prior and prior.get("state") in TERMINAL_STATES:
            return {"question_id": question_id, "skipped_terminal": True, "state": prior["state"]}
        if prior:
            raise RuntimeError("active route must be reconciled, never replayed")
        input_sha = route_input_sha(question_id, by_id[question_id]["planner_sha256"], manifest["runtime_fingerprint"])
        self.store.start_route(question_id, input_sha256=input_sha)
        started = time.perf_counter()
        try:
            provider = self.provider_factory()
            executor = self.executor_factory(self.config, provider, self.store, self.legacy_config_path)
            result = executor.run_one(question_id)
            if result["success"]:
                return self.store.terminal(question_id, success=True, extra={**result, "route_wall_time_sec": time.perf_counter() - started})
            return self.store.terminal(question_id, success=False, extra={**result, "route_wall_time_sec": time.perf_counter() - started})
        except BudgetExceeded:
            # A global budget stop must remain distinguishable and must not consume a request.
            self.store.terminal(question_id, success=False, extra={"failure_category": "global_budget_stop", "prediction": None, "route_wall_time_sec": time.perf_counter() - started})
            raise
        except Exception as error:
            return self.store.terminal(question_id, success=False, extra={"failure_category": f"{type(error).__name__}: {error}", "prediction": None, "route_wall_time_sec": time.perf_counter() - started})

    def run_ordered(self, question_ids: list[str]) -> dict[str, Any]:
        reconciliation = self.reconcile()
        results = []
        for question_id in question_ids:
            results.append(self.run_one(question_id))
        return {"reconciliation": reconciliation, "routes": results}
