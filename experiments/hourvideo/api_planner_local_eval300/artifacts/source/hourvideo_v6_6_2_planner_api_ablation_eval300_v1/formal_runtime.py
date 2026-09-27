from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import (
    load_json,
    sha256_file,
    write_json,
)
from experiments.hourvideo_v6_6_1_contract_telemetry_v1 import core as v661
from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1 import formal_runtime as frozen
from experiments.hourvideo_v6_6_2_local_context_limited_eval300_v1.prompt_contract import (
    PLANNER_R3_SYSTEM,
)
from experiments.hourvideo_v6_6_2_shared_coarse_contract_v1 import core as v662

from .anthropic_transport import AnthropicPlannerTransport, anthropic_sdk_preflight
from .budget import BudgetGuard
from .completion import wait_for_terminal_route
from .config_contract import validate_phase2_config
from .credentials import load_credential
from .pricing import PricingCatalog
from .providers import PlannerProviderAdapter, ProviderRequest
from .telemetry import AttemptTelemetryJournal


APPROVAL_TOKEN = "APPROVE_SINGLE_R3_API_GATE"


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _config(root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = load_json(config_path)
    baseline = load_json(root / cfg["frozen_baseline_config"])
    return cfg, baseline


def _planner_profile(cfg: dict[str, Any]) -> dict[str, Any]:
    profile_name = cfg["roles"]["planner"]["provider_profile"]
    return cfg["provider_profiles"][profile_name]


def _capacity_row(root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    target = cfg["single_question_gate"]
    path = root / cfg["frozen_inputs"]["local_route_capacity_reference"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    matches = [
        row for row in rows
        if row["question_id"] == target["question_id"] and row["route"] == target["route"]
    ]
    if len(matches) != 1:
        raise RuntimeError("single gate does not resolve to exactly one frozen capacity route")
    row = matches[0]
    if (
        row["formal_status"] != "context_overflow_pre_model"
        or row["eligible"] is not False
        or row["model_request_count"] != 0
        or row["planner_input_tokens"] != target["previous_planner_input_tokens"]
        or row["required_total_tokens"] != target["previous_required_total_tokens"]
        or row["max_model_len"] != target["previous_max_model_len"]
    ):
        raise RuntimeError("single gate is not the frozen local-context overflow route")
    return row


def _build_gate_request(
    root: Path, cfg: dict[str, Any], baseline: dict[str, Any],
) -> tuple[ProviderRequest, dict[str, Any]]:
    gate = cfg["single_question_gate"]
    qid = gate["question_id"]
    source = root / cfg["frozen_inputs"]["source_experiment"]
    question_path = source / "cases" / qid / "question_input.json"
    case_config_path = source / "case_configs" / f"{qid}.json"
    question = load_json(question_path)
    case_cfg = load_json(case_config_path)
    if question["question_id"] != qid or case_cfg["video_uid"] != gate["video_uid"]:
        raise RuntimeError("single gate source identity mismatch")
    map_path = Path(case_cfg["asset_paths"]["r3_2_navigation_map.json"])
    map_doc = load_json(map_path)
    coarse_ids = [row["coarse_id"] for row in map_doc["coarse_regions"]]
    requirements = v661.option_requirements(question)
    canonical_payload = {
        "question": question,
        "requirements": requirements,
        "navigation_map": v661._base._planner_map(map_doc),
    }
    provider_payload, provider_ids = v661._base._local_indexed_planner_contract(
        canonical_payload, coarse_ids,
    )
    schema = frozen._selected_schema(question, requirements, provider_ids)
    profile = _planner_profile(cfg)
    request = ProviderRequest(
        logical_call_id=f"{qid}::r3_2::planner",
        role="planner",
        provider=profile["provider"],
        model=profile["model"],
        system_prompt=PLANNER_R3_SYSTEM,
        payload=provider_payload,
        schema=schema,
        max_output_tokens=int(cfg["planner_contract"]["planner_max_output_tokens"]),
        estimated_input_tokens=int(cfg["api_budget"]["reservation_input_tokens_per_attempt"]),
        temperature=float(cfg["planner_contract"]["temperature"]),
        retry_feedback_instruction=(
            "Regenerate the complete JSON for the same question and unchanged navigation map; "
            "fix only this contract error."
        ),
    )
    context = {
        "question": question,
        "requirements": requirements,
        "coarse_ids": coarse_ids,
        "map_path": map_path,
        "map_sha256": sha256_file(map_path),
        "question_input_sha256": sha256_file(question_path),
        "provider_payload_sha256": _digest(provider_payload),
        "schema_sha256": _digest(schema),
        "prompt_sha256": hashlib.sha256(PLANNER_R3_SYSTEM.encode()).hexdigest(),
        "baseline": baseline,
    }
    return request, context


def phase2_preflight(
    root: Path,
    config_path: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    contract = validate_phase2_config(root, config_path)
    cfg, baseline = _config(root, config_path)
    profile = _planner_profile(cfg)
    credential = load_credential(profile["credential_env"], environ)
    sdk = anthropic_sdk_preflight()
    pricing = PricingCatalog.load(root / cfg["pricing_catalog"])
    price = pricing.estimate(profile["provider"], profile["model"], 1_000_000, 1_000_000)
    if not price.priced or str(price.estimated_cost_usd) != "6.00":
        raise RuntimeError("Claude Haiku 4.5 pricing lookup failed")
    capacity = _capacity_row(root, cfg)
    request, context = _build_gate_request(root, cfg, baseline)
    reservation = pricing.estimate(
        request.provider, request.model,
        request.estimated_input_tokens, request.max_output_tokens,
    )
    secret = credential.reveal_for_request()
    with tempfile.TemporaryDirectory(prefix="hourvideo_phase2_preflight_") as temp:
        temp_root = Path(temp)
        guard = BudgetGuard(
            temp_root / "budget_ledger.jsonl",
            cfg["api_budget"]["max_cost_usd"],
            hard_stop=cfg["api_budget"]["hard_stop"],
        )
        guard.reserve("synthetic-preflight", reservation.estimated_cost_usd)
        guard.reconcile("synthetic-preflight", "0")
        journal = AttemptTelemetryJournal(temp_root / "telemetry")
        common = {
            "request_id": "synthetic-preflight",
            "logical_call_id": request.logical_call_id,
            "role": "planner",
            "provider": request.provider,
            "model": request.model,
            "attempt_index": 0,
            "is_retry": False,
            "retry_reason": None,
            "pricing_version": pricing.pricing_version,
        }
        journal.record_start({**common, "leak_canary": secret}, (secret,))
        persisted = journal.starts_path.read_text(encoding="utf-8")
        if secret in persisted:
            raise RuntimeError("credential leakage guard failed")

    result = {
        **contract,
        "credential": credential.safe_record(),
        "provider_preflight": sdk,
        "pricing": {
            "status": "PASS",
            "pricing_version": pricing.pricing_version,
            "one_million_input_plus_output_cost_usd": str(price.estimated_cost_usd),
            "single_attempt_worst_case_reservation_usd": str(reservation.estimated_cost_usd),
        },
        "budget": {
            "status": "PASS",
            "max_cost_usd": cfg["api_budget"]["max_cost_usd"],
            "hard_stop": cfg["api_budget"]["hard_stop"],
            "max_provider_attempts": cfg["api_budget"]["max_provider_attempts"],
            "synthetic_ledger_only": True,
        },
        "telemetry": {"status": "PASS", "credential_leakage": False, "synthetic_only": True},
        "gate": {
            "status": "PREPARED_NOT_EXECUTED",
            "question_id": capacity["question_id"],
            "video_uid": capacity["video_uid"],
            "route": capacity["route"],
            "previous_formal_status": capacity["formal_status"],
            "previous_required_total_tokens": capacity["required_total_tokens"],
            "map_sha256": context["map_sha256"],
            "payload_sha256": context["provider_payload_sha256"],
            "schema_sha256": context["schema_sha256"],
            "prompt_sha256": context["prompt_sha256"],
        },
        "network_request_made": False,
        "experiment_output_written": False,
    }
    if secret in json.dumps(result, ensure_ascii=False):
        raise RuntimeError("credential appeared in preflight result")
    return result


def prepare_single_gate(root: Path, config_path: Path) -> dict[str, Any]:
    """Read-only gate preparation. Credential loading and network access are intentionally omitted."""
    validate_phase2_config(root, config_path)
    cfg, baseline = _config(root, config_path)
    capacity = _capacity_row(root, cfg)
    request, context = _build_gate_request(root, cfg, baseline)
    return {
        "status": "PREPARED_NOT_EXECUTED",
        "question_id": capacity["question_id"],
        "video_uid": capacity["video_uid"],
        "route": "r3_2",
        "provider": request.provider,
        "model": request.model,
        "max_output_tokens": request.max_output_tokens,
        "max_attempts": 1 + int(cfg["planner_contract"]["max_validation_retries"]),
        "payload_sha256": context["provider_payload_sha256"],
        "schema_sha256": context["schema_sha256"],
        "prompt_sha256": context["prompt_sha256"],
        "planner_destination": str(
            root / cfg["write_roots"]["planner"] / "cases"
            / capacity["question_id"] / "r3_2" / "planner.json"
        ),
        "approval_required": True,
        "network_request_made": False,
    }


def _write_downstream_gate_config(
    root: Path, cfg: dict[str, Any], baseline: dict[str, Any], config_path: Path,
) -> Path:
    downstream = json.loads(json.dumps(baseline))
    downstream.update({
        "experiment": cfg["experiment"],
        "planner_source_experiment": cfg["write_roots"]["planner"],
        "output_root": cfg["write_roots"]["live"],
        "phase2_source_config": str(config_path),
        "phase2_source_config_sha256": sha256_file(config_path),
    })
    destination = root / cfg["write_roots"]["manifests"] / "single_gate_downstream_config.json"
    write_json(destination, downstream)
    return destination


def _run_frozen_downstream_single_r3(root: Path, downstream_config: Path, qid: str) -> dict[str, Any]:
    original_case_paths = v661._case_paths
    original_sides = v661._configured_sides
    original_terminal_summarize = v662.summarize

    def filtered_case_paths(cfg_: dict[str, Any], root_: Path, identity_filter: str | None):
        return [row for row in original_case_paths(cfg_, root_, None) if row[0]["question_id"] == qid]

    v661._case_paths = filtered_case_paths
    v661._configured_sides = lambda cfg_: ("r3_2",)
    v662.summarize = frozen._deferred_versioned_canonical_summary
    try:
        return v662.run_live(root, downstream_config, video_uid=qid)
    finally:
        v661._case_paths = original_case_paths
        v661._configured_sides = original_sides
        v662.summarize = original_terminal_summarize


def execute_single_gate(root: Path, config_path: Path, approval_token: str) -> dict[str, Any]:
    """The only Phase 2 network entry point; guarded to one frozen R3 identity."""
    if approval_token != APPROVAL_TOKEN:
        raise RuntimeError("single R3 API gate lacks the required explicit approval token")
    validate_phase2_config(root, config_path)
    cfg, baseline = _config(root, config_path)
    capacity = _capacity_row(root, cfg)
    request, context = _build_gate_request(root, cfg, baseline)
    qid = capacity["question_id"]
    planner_path = root / cfg["write_roots"]["planner"] / "cases" / qid / "r3_2" / "planner.json"
    if not planner_path.is_file():
        profile = _planner_profile(cfg)
        pricing = PricingCatalog.load(root / cfg["pricing_catalog"])
        adapter = PlannerProviderAdapter(
            transports={
                "anthropic": AnthropicPlannerTransport(timeout_sec=profile["timeout_sec"]),
            },
            credential_env_by_provider={"anthropic": profile["credential_env"]},
            pricing=pricing,
            telemetry=AttemptTelemetryJournal(root / cfg["write_roots"]["telemetry"]),
            budget=BudgetGuard(
                root / cfg["write_roots"]["telemetry"] / "budget_ledger.jsonl",
                cfg["api_budget"]["max_cost_usd"],
                hard_stop=cfg["api_budget"]["hard_stop"],
            ),
        )

        def validate(raw: dict[str, Any], schema: dict[str, Any]) -> None:
            frozen._validate_and_project_selected(
                raw, context["question"], context["requirements"], context["coarse_ids"],
            )

        result = adapter.call(
            request,
            max_attempts=1 + int(cfg["planner_contract"]["max_validation_retries"]),
            schema_validator=validate,
        )
        plan = frozen._validate_and_project_selected(
            result.parsed_output,
            context["question"], context["requirements"], context["coarse_ids"],
        )
        write_json(planner_path, {
            "schema_version": "v6_6_2_planner_api_ablation_frozen_planner_v1",
            "question_id": qid,
            "video_uid": capacity["video_uid"],
            "route": "r3_2",
            "input_asset_path": str(context["map_path"]),
            "input_asset_sha256": context["map_sha256"],
            "input_payload_sha256": context["provider_payload_sha256"],
            "planner_model": request.model,
            "provider": request.provider,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "prompt_sha256": context["prompt_sha256"],
            "output": plan,
            "usage": {
                "provider": result.provider,
                "model": result.model,
                "request_id": result.request_id,
                "provider_request_id": result.provider_request_id,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_sec": result.latency_sec,
                "pricing_version": result.pricing_version,
                "estimated_cost_usd": str(result.estimated_cost_usd),
                "attempt_index": result.attempt_index,
            },
            "gold_loaded": False,
        })
    downstream_config = _write_downstream_gate_config(root, cfg, baseline, config_path)
    downstream = _run_frozen_downstream_single_r3(root, downstream_config, qid)
    wait_for_terminal_route(root / cfg["write_roots"]["live"], qid, "r3_2")
    return {
        "status": "SINGLE_R3_GATE_EXECUTED",
        "question_id": qid,
        "route": "r3_2",
        "planner_artifact": str(planner_path),
        "downstream": downstream,
        "eval300_run": False,
    }


def run_planner_generation(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("batch Planner generation is disabled; only the approved single R3 gate exists")


def run_eval300(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("Eval300 is disabled in Phase 2; separate approval is required")
