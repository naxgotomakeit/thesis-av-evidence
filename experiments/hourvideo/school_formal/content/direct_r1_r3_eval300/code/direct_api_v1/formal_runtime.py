"""Crash-safe orchestration for the frozen Direct-v1.2 formal protocol.

The runtime deliberately supports route-boundary resume only.  A route left in
``running`` state is materialised as ``interrupted_active_route`` and is never
sent to a provider again.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .anthropic_provider import direct_action_tools
from .controller import DirectController
from .maps import load_direct_input
from .policy import MAX_NEW_IMAGES_PER_TURN, MAX_UNIQUE_IMAGES_PER_QUESTION
from .prompt import DIRECT_V1_PROMPT_VERSION, DIRECT_V1_SYSTEM_PROMPT
from .state import DirectSessionState


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


class InjectedCrash(BaseException):
    """Test-only hard crash which provider/controller exception handlers cannot absorb."""


class CrashInjector:
    """One-shot crash injection at a named durable boundary."""

    def __init__(self, boundary: str | None = None) -> None:
        self.boundary = boundary
        self.triggered = False

    def checkpoint(self, boundary: str) -> None:
        if self.boundary == boundary and not self.triggered:
            self.triggered = True
            raise InjectedCrash(boundary)


class FormalStore:
    def __init__(self, root: Path, experiment_id: str, cap: float = 50.0) -> None:
        self.root, self.experiment_id, self.cap = root, experiment_id, cap

    @property
    def ledger_path(self) -> Path:
        return self.root / "budget_ledger.json"

    def initialise(self) -> None:
        if not self.ledger_path.exists():
            atomic_json(self.ledger_path, {"cap_usd": self.cap, "spent_usd": 0.0, "attempt_ids": [], "charges": []})

    def status(self, route_id: str) -> dict[str, Any] | None:
        path = self.root / "route_status" / f"{route_id}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_status(self, route_id: str, value: dict[str, Any]) -> None:
        atomic_json(self.root / "route_status" / f"{route_id}.json", value)

    def start(self, route_id: str, **metadata: Any) -> None:
        existing = self.status(route_id)
        if existing and str(existing.get("state", "")).startswith("terminal_"):
            raise RuntimeError("terminal route cannot restart")
        self.write_status(route_id, {"state": "running", "route_id": route_id, "started_at": now(), **metadata})

    def request_start(self, route_id: str, **record: Any) -> None:
        append_jsonl(self.root / "journals" / "request_start.jsonl", {"timestamp": now(), "experiment_id": self.experiment_id, "route_id": route_id, **record})

    def attempt_end(self, route_id: str, **record: Any) -> None:
        append_jsonl(self.root / "journals" / "attempt_end.jsonl", {"timestamp": now(), "experiment_id": self.experiment_id, "route_id": route_id, **record})

    def controller_result(self, route_id: str, **record: Any) -> None:
        append_jsonl(self.root / "journals" / "controller_result.jsonl", {"timestamp": now(), "experiment_id": self.experiment_id, "route_id": route_id, **record})

    def terminal(self, route_id: str, success: bool, **record: Any) -> None:
        self.write_status(route_id, {"state": "terminal_success" if success else "terminal_failed", "route_id": route_id, "terminal_at": now(), **record})

    def spend(self) -> float:
        return float(json.loads(self.ledger_path.read_text(encoding="utf-8"))["spent_usd"])

    def assert_can_attempt(self, estimated_upper_bound_usd: float) -> None:
        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if float(data["spent_usd"]) + float(estimated_upper_bound_usd) > float(data["cap_usd"]) + 1e-12:
            raise RuntimeError("formal_hard_budget_would_be_exceeded")

    def charge_once(self, attempt_id: str, cost: float, **metadata: Any) -> bool:
        if cost < 0:
            raise RuntimeError("negative provider cost")
        data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
        if attempt_id in data["attempt_ids"]:
            return False
        next_spend = float(data["spent_usd"]) + float(cost)
        if next_spend > float(data["cap_usd"]) + 1e-12:
            raise RuntimeError("formal_hard_budget_would_be_exceeded")
        data["attempt_ids"].append(attempt_id)
        data.setdefault("charges", []).append({"attempt_id": attempt_id, "cost_usd": float(cost), **metadata})
        data["spent_usd"] = next_spend
        atomic_json(self.ledger_path, data)
        return True

    def reconcile_ledger(self) -> int:
        """Idempotently charge every durable completed attempt."""
        reconciled = 0
        for record in read_jsonl(self.root / "journals" / "attempt_end.jsonl"):
            if self.charge_once(
                str(record["attempt_id"]), float(record.get("cache_aware_usd") or 0.0),
                route_id=record.get("route_id"), provider_status=record.get("provider_status"),
            ):
                reconciled += 1
        return reconciled

    def result_path(self, route_id: str) -> Path:
        return self.root / "route_artifacts" / f"{route_id}.json"

    def materialise_interrupted(self, route_id: str, status: dict[str, Any]) -> None:
        completed_attempts = [row for row in read_jsonl(self.root / "journals" / "attempt_end.jsonl")
                              if row.get("route_id") == route_id]
        resource_totals = {
            **status.get("resource_totals", {}),
            "provider_attempts": len(completed_attempts),
            "usd": sum(float(row.get("cache_aware_usd") or 0.0) for row in completed_attempts),
        }
        artifact = {
            "route_id": route_id, "question_id": status.get("question_id"), "method": status.get("method"),
            "terminal_status": "interrupted_active_route", "final_prediction": None,
            "resource_totals": resource_totals, "materialised_at": now(),
        }
        atomic_json(self.result_path(route_id), artifact)
        self.terminal(route_id, False, category="interrupted_active_route", prediction=None,
                      artifact_path=str(self.result_path(route_id)), resource_totals=artifact["resource_totals"])

    def reconcile_active_routes(self, route_ids: Iterable[str]) -> int:
        reconciled = 0
        for route_id in route_ids:
            status = self.status(route_id)
            if status and status.get("state") == "running":
                self.materialise_interrupted(route_id, status)
                reconciled += 1
        return reconciled

    # Backward-compatible name retained for the already-materialised primitive.
    def reconcile(self, route_ids: list[str]) -> None:
        self.reconcile_active_routes(route_ids)


class FormalRouteLifecycle:
    """Lifecycle shared by a route's provider and controller."""

    def __init__(self, store: FormalStore, route_id: str, crash: CrashInjector | None = None) -> None:
        self.store, self.route_id = store, route_id
        self.crash = crash or CrashInjector()
        self._sequence = len([row for row in read_jsonl(store.root / "journals" / "request_start.jsonl") if row.get("route_id") == route_id])
        self._attempt_ids: dict[tuple[int, int], str] = {}

    def _event(self, event: str, **record: Any) -> None:
        append_jsonl(self.store.root / "journals" / "lifecycle.jsonl",
                     {"timestamp": now(), "experiment_id": self.store.experiment_id,
                      "route_id": self.route_id, "event": event, **record})

    def provider_request_start(self, *, turn_index: int, attempt_index: int, provider: str, model: str,
                               estimated_upper_bound_usd: float = 0.0) -> str:
        self.store.assert_can_attempt(estimated_upper_bound_usd)
        self._sequence += 1
        attempt_id = f"{self.route_id}:attempt:{self._sequence}"
        self._attempt_ids[(turn_index, attempt_index)] = attempt_id
        self.store.request_start(self.route_id, attempt_id=attempt_id, turn_index=turn_index,
                                 attempt_index=attempt_index, provider=provider, model=model)
        self._event("request_start", attempt_id=attempt_id)
        self.crash.checkpoint("after_request_start")
        return attempt_id

    def provider_response_received(self) -> None:
        self.crash.checkpoint("after_provider_call")

    def provider_attempt_end(self, attempt_id: str, record: Any) -> None:
        payload = _jsonable(record)
        self.store.attempt_end(self.route_id, attempt_id=attempt_id, **payload)
        self._event("attempt_end", attempt_id=attempt_id)
        self.crash.checkpoint("after_attempt_end")
        self.store.charge_once(attempt_id, float(payload.get("cache_aware_usd") or 0.0),
                               route_id=self.route_id, provider_status=payload.get("provider_status"))
        self._event("ledger_reconciliation", attempt_id=attempt_id)
        self.crash.checkpoint("after_ledger_reconciliation")

    def controller_validation(self, *, validation: str, state: DirectSessionState,
                              action: Any = None, provider_attempts: Iterable[Any] = ()) -> None:
        attempt_ids = []
        for record in provider_attempts:
            # IDs are deliberately recoverable from route/turn/attempt ordering.
            turn = getattr(record, "turn_index", None)
            attempt = getattr(record, "attempt_index", None)
            attempt_ids.append(self._attempt_ids.get((turn, attempt), {"turn_index": turn, "attempt_index": attempt}))
        self.store.controller_result(
            self.route_id, turn_index=state.rounds + 1, validation=validation,
            action=_jsonable(action), provider_attempts=attempt_ids,
        )
        self._event("controller_result", validation=validation)
        self.crash.checkpoint("after_controller_result")


class FormalOrchestrator:
    """Execute a frozen manifest with an injectable provider factory."""

    def __init__(self, *, manifest_path: Path, namespace: Path,
                 agent_factory: Callable[[dict[str, Any], FormalRouteLifecycle, DirectSessionState], Any],
                 resolver_factory: Callable[[dict[str, Any]], Any], cap_usd: float = 50.0,
                 crash: CrashInjector | None = None) -> None:
        self.manifest_path, self.namespace = manifest_path, namespace
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.agent_factory, self.resolver_factory = agent_factory, resolver_factory
        self.store = FormalStore(namespace, str(self.manifest["experiment_id"]), cap_usd)
        self.crash = crash or CrashInjector()

    def execute(self) -> dict[str, Any]:
        self.store.initialise()
        routes = list(self.manifest["routes"])
        route_ids = [str(route["route_id"]) for route in routes]
        ledger_reconciled = self.store.reconcile_ledger()
        interrupted = self.store.reconcile_active_routes(route_ids)
        summary = {"route_count": len(routes), "executed": 0, "skipped_terminal": 0,
                   "interrupted_materialised": interrupted, "ledger_attempts_reconciled": ledger_reconciled,
                   "provider_attempts_before": len(read_jsonl(self.namespace / "journals" / "attempt_end.jsonl")),
                   "spend_before_usd": self.store.spend()}
        input_root = self.manifest_path.parent / "inputs"
        for route in routes:
            route_id = str(route["route_id"])
            status = self.store.status(route_id)
            if status and str(status.get("state", "")).startswith("terminal_"):
                summary["skipped_terminal"] += 1
                continue
            self.store.start(route_id, question_id=route["question_id"], method=route["method"], resource_totals={})
            append_jsonl(self.namespace / "journals" / "lifecycle.jsonl",
                         {"timestamp": now(), "experiment_id": self.store.experiment_id,
                          "route_id": route_id, "event": "route_running"})
            self.crash.checkpoint("after_route_running")
            try:
                direct_input = load_direct_input(
                    question_path=input_root / f"{route['question_id']}.json", map_path=Path(route["map_path"]),
                    method=route["method"], expected_map_sha256=route["map_sha256"],
                )
                state = DirectSessionState(direct_input)
                lifecycle = FormalRouteLifecycle(self.store, route_id, self.crash)
                agent = self.agent_factory(route, lifecycle, state)
                controller = DirectController(resolver=self.resolver_factory(route), max_turns=32,
                                              enable_action_correction=True, lifecycle=lifecycle)
                telemetry = controller.run(state=state, agent=agent)
                artifact = telemetry.as_dict()
                atomic_json(self.store.result_path(route_id), artifact)
                append_jsonl(self.namespace / "journals" / "lifecycle.jsonl",
                             {"timestamp": now(), "experiment_id": self.store.experiment_id,
                              "route_id": route_id, "event": "terminal_artifact"})
                self.crash.checkpoint("after_terminal_artifact")
                success = telemetry.terminal_status == "final_answer" and telemetry.final_prediction is not None
                self.store.terminal(route_id, success, category=telemetry.terminal_status,
                                    prediction=telemetry.final_prediction, artifact_path=str(self.store.result_path(route_id)),
                                    resource_totals={"provider_attempts": telemetry.total_api_attempts,
                                                     "usd": telemetry.total_usd,
                                                     "images": telemetry.unique_images_transmitted,
                                                     "rounds": telemetry.rounds})
                append_jsonl(self.namespace / "journals" / "lifecycle.jsonl",
                             {"timestamp": now(), "experiment_id": self.store.experiment_id,
                              "route_id": route_id, "event": "terminal_status"})
                self.crash.checkpoint("after_terminal_status")
                summary["executed"] += 1
            except InjectedCrash:
                raise
            except Exception as error:
                artifact = {"route_id": route_id, "question_id": route["question_id"], "method": route["method"],
                            "terminal_status": f"runtime_failure:orchestrator:{type(error).__name__}",
                            "final_prediction": None, "resource_totals": {}, "materialised_at": now()}
                atomic_json(self.store.result_path(route_id), artifact)
                self.store.terminal(route_id, False, category=artifact["terminal_status"], prediction=None,
                                    artifact_path=str(self.store.result_path(route_id)), resource_totals={})
                summary["executed"] += 1
        summary["provider_attempts_after"] = len(read_jsonl(self.namespace / "journals" / "attempt_end.jsonl"))
        summary["new_provider_attempts"] = summary["provider_attempts_after"] - summary["provider_attempts_before"]
        summary["spend_after_usd"] = self.store.spend()
        summary["additional_spend_usd"] = summary["spend_after_usd"] - summary["spend_before_usd"]
        summary["terminal_routes"] = sum(
            bool(self.store.status(route_id)) and str(self.store.status(route_id).get("state", "")).startswith("terminal_")
            for route_id in route_ids
        )
        summary["completed_at"] = now()
        append_jsonl(self.namespace / "execution_history.jsonl", summary)
        atomic_json(self.namespace / "last_execution.json", summary)
        return summary


def fingerprints(root: Path, config: dict[str, Any], manifest: dict[str, Any] | None = None,
                 input_root: Path | None = None) -> dict[str, Any]:
    """Recompute every explicit implementation/configuration launch fingerprint."""
    source = root / "src/direct_api_v1"
    result = {
        "provider_name": config["provider"],
        "model_id": config["model"],
        "provider_config_hash": sha(root / "config/direct_v1_anthropic_smoke.json"),
        "prompt_hash": sha(source / "prompt.py"),
        "system_prompt_text_hash": hashlib.sha256(DIRECT_V1_SYSTEM_PROMPT.encode()).hexdigest(),
        "policy_hash": sha(source / "policy.py"),
        "controller_hash": sha(source / "controller.py"),
        "provider_hash": sha(source / "anthropic_provider.py"),
        "actions_hash": sha(source / "actions.py"),
        "state_machine_hash": sha(source / "state.py"),
        "maps_loader_hash": sha(source / "maps.py"),
        "telemetry_hash": sha(source / "telemetry.py"),
        "frame_resolver_hash": sha(source / "frame_resolver.py"),
        "pricing_hash": sha(source / "pricing.py"),
        "formal_runtime_hash": sha(source / "formal_runtime.py"),
        "formal_launch_gate_hash": sha(source / "formal_launch_gate.py"),
        "real_formal_runner_hash": sha(root / "scripts/execute_direct_v1_2_3x16_r1_r3_eval300_formal_v1.py"),
        "final_candidate_builder_hash": sha(root / "scripts/materialize_direct_v1_2_final_candidate_manifest.py"),
        "action_schema_hash": hashlib.sha256(json.dumps({str(n): direct_action_tools(n) for n in (3, 2, 1, 0)}, sort_keys=True).encode()).hexdigest(),
        "max_new_images_per_turn": MAX_NEW_IMAGES_PER_TURN,
        "max_unique_images_per_question": MAX_UNIQUE_IMAGES_PER_QUESTION,
        "provider_transport_retries": config["max_retries"],
        "structural_correction_retries": 1,
        "max_route_turns": 32,
        "max_output_tokens": config["max_output_tokens"],
        "provider_timeout_sec": config["timeout_sec"],
        "provider_temperature": 0.0,
        "cache_configuration": config["prompt_caching"],
        "pricing_configuration": config["anthropic_cache_aware_pricing_usd_per_million_tokens"],
        "pricing_catalog_path": config["pricing_catalog_path"],
        "configuration_schema_version": config["schema_version"],
        "prompt_version": DIRECT_V1_PROMPT_VERSION,
        "formal_hard_budget_usd": 50.0,
    }
    if manifest is not None:
        routes = list(manifest["routes"])
        if input_root is None:
            input_root = root / "outputs/direct_v1_formal" / str(manifest["experiment_id"]) / "inputs"
        question_ids = list(dict.fromkeys(str(route["question_id"]) for route in routes))
        population_path = Path(manifest["population_manifest"])
        population = json.loads(population_path.read_text(encoding="utf-8"))
        input_rows = [{"question_id": question_id, "path": str(input_root / f"{question_id}.json"),
                       "sha256": sha(input_root / f"{question_id}.json")} for question_id in question_ids]
        map_rows = [{"route_id": route["route_id"], "method": route["method"],
                     "path": route["map_path"], "sha256": sha(Path(route["map_path"]))} for route in routes]
        hierarchy_rows = [{"video_id": video["video_id"], "path": video["hierarchy"],
                           "sha256": sha(Path(video["hierarchy"]))} for video in manifest["videos"]]
        result.update({
            "eval300_population_manifest_hash": sha(population_path),
            "eval300_ordered_routes_source_hash": population["source_ordered_routes_sha256"],
            "question_input_closure_hash": canonical_sha(input_rows),
            "route_order_hash": canonical_sha([
                {key: route[key] for key in ("route_id", "method", "question_id", "video_id")} for route in routes
            ]),
            "r1_map_closure_hash": canonical_sha([row for row in map_rows if row["method"] == "R1"]),
            "r3_map_closure_hash": canonical_sha([row for row in map_rows if row["method"] == "R3"]),
            "hierarchy_closure_hash": canonical_sha(hierarchy_rows),
        })
    return result
