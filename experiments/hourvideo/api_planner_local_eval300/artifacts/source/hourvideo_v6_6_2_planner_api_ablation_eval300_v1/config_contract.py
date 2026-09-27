from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


EXPERIMENT = "hourvideo_v6_6_2_planner_api_ablation_eval300_v1"
BASELINE = "hourvideo_v6_6_2_local_context_limited_eval300_v1"
OUTPUT_PREFIX = PurePosixPath("outputs/experiments") / EXPERIMENT


class ConfigContractError(RuntimeError):
    pass


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _under(path: str, parent: PurePosixPath) -> bool:
    candidate = PurePosixPath(path)
    return candidate == parent or parent in candidate.parents


def validate_phase2_config(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = _load(config_path)
    if cfg.get("experiment") != EXPERIMENT:
        raise ConfigContractError("new experiment identity mismatch")
    if cfg.get("phase") != "phase2_single_gate_preparation":
        raise ConfigContractError("configuration is not Phase 2 single-gate preparation")
    baseline_rel = cfg.get("frozen_baseline_config")
    baseline_path = root / str(baseline_rel)
    baseline = _load(baseline_path)
    if baseline.get("experiment") != BASELINE:
        raise ConfigContractError("frozen baseline identity mismatch")
    frozen_inputs = cfg.get("frozen_inputs") or {}
    expected_inputs = {
        "source_experiment": baseline["source_experiment"],
        "prompt_freeze_manifest": baseline["prompt_freeze_manifest"],
        "local_capacity_reference": baseline["capacity_summary"],
        "local_route_capacity_reference": str(
            PurePosixPath(baseline["eligibility_root"]) / "route_capacity.jsonl"
        ),
    }
    if frozen_inputs != expected_inputs:
        raise ConfigContractError("frozen input references drift from the V6.6.2 baseline")

    if cfg.get("output_root") != str(OUTPUT_PREFIX):
        raise ConfigContractError("output_root is not the isolated experiment namespace")
    writes = cfg.get("write_roots") or {}
    if set(writes) != {"planner", "live", "telemetry", "logs", "manifests", "canonical_summary"}:
        raise ConfigContractError("write_roots are incomplete")
    for name, value in writes.items():
        if not _under(str(value), OUTPUT_PREFIX):
            raise ConfigContractError(f"write root escapes isolated namespace: {name}")
    old_output = PurePosixPath("outputs/experiments") / BASELINE
    if any(_under(str(value), old_output) for value in writes.values()):
        raise ConfigContractError("new experiment attempts to write inside frozen outputs")

    roles = cfg.get("roles") or {}
    if roles.get("planner", {}).get("mode") != "external_api":
        raise ConfigContractError("Planner must be the sole external API role")
    for role in ("shared", "fine", "final"):
        if roles.get(role, {}).get("mode") != "frozen_local":
            raise ConfigContractError(f"{role} must remain frozen_local")
    if set(roles) != {"planner", "shared", "fine", "final"}:
        raise ConfigContractError("unexpected role configuration")
    expected_frozen_profiles = {
        "shared": "frozen_baseline_config.anthropic",
        "fine": "frozen_baseline_config.gemini",
        "final": "frozen_baseline_config.final",
    }
    profiles = cfg.get("provider_profiles") or {}
    for role, source in expected_frozen_profiles.items():
        profile_name = roles[role]["provider_profile"]
        if profiles.get(profile_name) != {"source": source}:
            raise ConfigContractError(f"{role} provider does not reference the frozen baseline")

    guards = cfg.get("phase2_guards") or {}
    if guards != {
        "api_adapter_ready": True,
        "single_gate_prepared": True,
        "single_gate_execution_requires_approval_token": True,
        "batch_planner_generation_enabled": False,
        "eval300_enabled": False,
    }:
        raise ConfigContractError("Phase 2 execution guards are not single-gate-only")
    planner_profile = cfg["provider_profiles"][roles["planner"]["provider_profile"]]
    if planner_profile != {
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "credential_env": "HOURVIDEO_PLANNER_API_KEY",
        "timeout_sec": 600,
        "sdk_max_retries": 0,
        "structured_output": "output_config.format.json_schema",
        "real_network_transport_enabled": True,
    }:
        raise ConfigContractError("Planner provider profile differs from the approved Claude configuration")
    planner_contract = cfg.get("planner_contract") or {}
    if planner_contract.get("planner_max_output_tokens") != baseline["planner_max_output_tokens"]:
        raise ConfigContractError("Planner output budget drift")
    if planner_contract.get("max_validation_retries") != baseline["max_validation_retries"]:
        raise ConfigContractError("Planner retry-rule drift")
    if planner_contract.get("temperature") != baseline["anthropic"]["temperature"]:
        raise ConfigContractError("Planner temperature drift")
    def forbidden_secret_field(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                str(key).lower() in {"api_key", "apikey", "authorization", "secret"}
                or forbidden_secret_field(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(forbidden_secret_field(item) for item in value)
        return False

    if forbidden_secret_field(cfg):
        raise ConfigContractError("configuration contains a forbidden secret field")

    budget = cfg.get("api_budget") or {}
    if budget != {
        "max_cost_usd": "1.000000",
        "hard_stop": True,
        "scope": "single_r3_gate_only",
        "max_provider_attempts": 3,
        "reservation_input_tokens_per_attempt": 200000,
        "real_spending_requires_gate_approval": True,
    }:
        raise ConfigContractError("API budget is not frozen to the single-gate contract")
    pricing_path = root / str(cfg.get("pricing_catalog"))
    pricing = _load(pricing_path)
    expected_price = [{
        "provider": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "input_usd_per_million_tokens": "1.00",
        "output_usd_per_million_tokens": "5.00",
        "pricing_basis": "standard Claude API; no prompt caching or batch discount",
    }]
    if (
        pricing.get("pricing_version") != "anthropic_claude_api_2026_08_31_v1"
        or pricing.get("status") != "FROZEN_FOR_SINGLE_R3_GATE"
        or pricing.get("effective_date") != "2026-08-31"
        or pricing.get("entries") != expected_price
    ):
        raise ConfigContractError("Claude Haiku 4.5 pricing freeze mismatch")
    population = cfg.get("population") or {}
    if population.get("route_count") != 600 or population.get("reuse_local_overflow_as_execution_filter") is not False:
        raise ConfigContractError("new experiment population must be full Eval300 for both routes")
    gate = cfg.get("single_question_gate") or {}
    if gate != {
        "question_id": "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31",
        "video_uid": "6fd90f8d-7a4d-425d-a812-3268db0b0342",
        "route": "r3_2",
        "previous_formal_status": "context_overflow_pre_model",
        "previous_planner_input_tokens": 32422,
        "previous_required_total_tokens": 40614,
        "previous_max_model_len": 36544,
        "approval_token": "APPROVE_SINGLE_R3_API_GATE",
    }:
        raise ConfigContractError("single-question gate identity is not the frozen overflow route")

    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    baseline_sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    return {
        "status": "PHASE2_CONFIG_VALID",
        "experiment": EXPERIMENT,
        "config_sha256": config_sha,
        "frozen_baseline_config": str(baseline_path),
        "frozen_baseline_config_sha256": baseline_sha,
        "planner_provider": planner_profile["provider"],
        "planner_model": planner_profile["model"],
        "single_gate_requires_approval": True,
        "batch_planner_generation_enabled": False,
        "eval300_enabled": False,
        "write_roots": writes,
    }


def validate_phase1_config(root: Path, config_path: Path) -> dict[str, Any]:
    """Compatibility name retained for callers; validation now enforces the Phase 2 freeze."""
    return validate_phase2_config(root, config_path)
