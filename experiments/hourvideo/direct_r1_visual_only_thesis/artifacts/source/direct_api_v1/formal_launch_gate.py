"""Fail-closed, read-only preflight for the Direct-v1.2 real formal launch."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from direct_api_prep.evidence import assert_safe_direct_evidence

from .formal_runtime import fingerprints, read_jsonl, sha
from .frame_resolver import FrozenFrameResolver
from .maps import FORBIDDEN_QUESTION_KEYS


EXPECTED_EXPERIMENT = "direct_v1_2_3x16_r1_r3_eval300_formal_v1"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"
EXPECTED_CACHE = "anthropic_ephemeral_cache_boundary_after_verbatim_native_map"
EXPECTED_PRICING = {
    "ordinary_input": 1.0,
    "output": 5.0,
    "cache_write_5m": 1.25,
    "cache_write_1h": None,
    "cache_read": 0.1,
    "cache_write_ttl": "5m",
}
EXPECTED_COUNTS = {"questions": 300, "videos": 12, "routes": 600, "R1": 300, "R3": 300}
LOCK_SCHEMA = "direct_v1_2_formal_launch_lock_v3"


class LaunchPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidatedLaunch:
    manifest: dict[str, Any]
    config: dict[str, Any]
    mode: str
    manifest_sha256: str


def _object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LaunchPreflightError(f"{label} missing or malformed") from error
    if not isinstance(value, dict):
        raise LaunchPreflightError(f"{label} must be a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LaunchPreflightError(message)


def validate_population_and_inputs(manifest: dict[str, Any], manifest_path: Path) -> None:
    """Validate the complete no-gold population, route order, and asset closure."""
    population_path = Path(str(manifest.get("population_manifest", "")))
    population = _object(population_path, "Eval300 population manifest")
    _require(sha(population_path) == manifest.get("population_manifest_sha256"), "Eval300 population SHA mismatch")
    question_ids = population.get("eval300_question_ids")
    question_to_video = population.get("question_to_video")
    _require(isinstance(question_ids, list) and len(question_ids) == 300 and len(set(question_ids)) == 300,
             "Eval300 question population mismatch")
    _require(isinstance(question_to_video, dict) and set(question_to_video) == set(question_ids),
             "Eval300 question/video mapping mismatch")
    video_order = list(dict.fromkeys(question_to_video[question_id] for question_id in question_ids))
    _require(len(video_order) == 12, "Eval300 video population mismatch")

    videos = manifest.get("videos")
    routes = manifest.get("routes")
    _require(isinstance(videos, list) and [video.get("video_id") for video in videos] == video_order,
             "formal video order mismatch")
    _require(isinstance(routes, list) and len(routes) == 600, "formal route count mismatch")
    expected_routes: list[tuple[str, str, str, str]] = []
    video_by_id = {video["video_id"]: video for video in videos}
    for video_id in video_order:
        ordered_questions = [question_id for question_id in question_ids if question_to_video[question_id] == video_id]
        _require(video_by_id[video_id].get("question_ids") == ordered_questions,
                 f"question order mismatch for video {video_id}")
        for method in ("R1", "R3"):
            expected_routes.extend((f"{method}:{question_id}", method, question_id, video_id)
                                   for question_id in ordered_questions)
    observed_routes = [(route.get("route_id"), route.get("method"), route.get("question_id"), route.get("video_id"))
                       for route in routes]
    _require(observed_routes == expected_routes, "formal route identity/order mismatch")
    _require(len({route[0] for route in observed_routes}) == 600, "formal route identities are not unique")

    input_root = manifest_path.parent / "inputs"
    for question_id in question_ids:
        question = _object(input_root / f"{question_id}.json", f"question input {question_id}")
        _require(question.get("question_id") == question_id, f"question identity mismatch: {question_id}")
        _require(not (set(question) & FORBIDDEN_QUESTION_KEYS), f"gold field in question input: {question_id}")
        assert_safe_direct_evidence(question)
        options = question.get("answer_options")
        _require(isinstance(options, list) and len(options) == 5 and
                 [row.get("option_id") for row in options if isinstance(row, dict)] == list("ABCDE"),
                 f"answer option closure mismatch: {question_id}")

    checked_maps: set[Path] = set()
    for route in routes:
        video = video_by_id[route["video_id"]]
        prefix = route["method"].lower()
        map_path = Path(route["map_path"])
        _require(str(map_path) == video.get(f"{prefix}_map"), f"map pairing mismatch: {route['route_id']}")
        _require(route.get("map_sha256") == video.get(f"{prefix}_sha256"), f"map manifest SHA mismatch: {route['route_id']}")
        _require(sha(map_path) == route.get("map_sha256"), f"map content SHA mismatch: {route['route_id']}")
        if map_path not in checked_maps:
            assert_safe_direct_evidence(_object(map_path, f"native map {map_path}"))
            checked_maps.add(map_path)
    _require(len(checked_maps) == 24, "R1/R3 native-map closure mismatch")

    for video in videos:
        hierarchy_path = Path(video["hierarchy"])
        _require(sha(hierarchy_path) == video.get("hierarchy_sha256"),
                 f"hierarchy SHA mismatch: {video['video_id']}")
        resolver = FrozenFrameResolver.from_hierarchy(_object(hierarchy_path, "frozen hierarchy"))
        _require(resolver.video_uid == video["video_id"] and resolver.duration_sec >= 0,
                 f"frame resolver closure mismatch: {video['video_id']}")


def validate_namespace_state(namespace: Path, manifest: dict[str, Any]) -> str:
    """Validate a clean first launch or a conservative route-boundary resume."""
    route_ids = {route["route_id"] for route in manifest["routes"]}
    statuses = list((namespace / "route_status").glob("*.json")) if (namespace / "route_status").exists() else []
    artifacts = list((namespace / "route_artifacts").glob("*.json")) if (namespace / "route_artifacts").exists() else []
    starts = read_jsonl(namespace / "journals/request_start.jsonl")
    ends = read_jsonl(namespace / "journals/attempt_end.jsonl")
    controllers = read_jsonl(namespace / "journals/controller_result.jsonl")
    ledger = _object(namespace / "budget_ledger.json", "formal budget ledger")
    _require(float(ledger.get("cap_usd", -1)) == 50.0, "formal ledger cap mismatch")
    _require(0.0 <= float(ledger.get("spent_usd", -1)) <= 50.0, "formal ledger spend invalid")
    _require(isinstance(ledger.get("attempt_ids"), list) and len(ledger["attempt_ids"]) == len(set(ledger["attempt_ids"])),
             "formal ledger attempt IDs invalid")
    first_launch = not statuses and not artifacts and not starts and not ends and not controllers
    if first_launch:
        _require(float(ledger["spent_usd"]) == 0.0 and ledger["attempt_ids"] == [],
                 "first-launch ledger is not empty")
        return "first_launch"

    for path in statuses:
        status = _object(path, "route status")
        _require(status.get("route_id") in route_ids and status.get("state") in {"running", "terminal_success", "terminal_failed"},
                 "resume route status invalid")
    for record in starts + ends + controllers:
        _require(record.get("experiment_id") == manifest["experiment_id"] and record.get("route_id") in route_ids,
                 "foreign journal record in real formal namespace")
    for record in starts:
        _require(record.get("provider") == "anthropic" and record.get("model") == EXPECTED_MODEL,
                 "non-formal provider request in real namespace")
    for record in ends:
        _require(record.get("provider") == "anthropic" and record.get("model") == EXPECTED_MODEL,
                 "non-formal provider attempt in real namespace")
    ended_ids = {record.get("attempt_id") for record in ends}
    _require(set(ledger["attempt_ids"]) <= ended_ids, "ledger contains an attempt without durable attempt_end")
    return "resume"


def validate_launch_preflight(*, root: Path, namespace: Path, manifest_path: Path,
                              launch_lock_path: Path, config_path: Path,
                              fingerprint_factory: Callable[..., dict[str, Any]] = fingerprints) -> ValidatedLaunch:
    """Validate every lock before credentials or provider construction are reachable."""
    try:
        lock = _object(launch_lock_path, "formal launch lock")
        _require(lock.get("schema_version") == LOCK_SCHEMA, "launch-lock schema mismatch")
        _require(lock.get("experiment_id") == EXPECTED_EXPERIMENT, "launch-lock experiment mismatch")
        _require(lock.get("real_execution_allowed") is True, "launch lock does not allow real execution")
        _require(Path(str(lock.get("candidate_manifest_path", ""))).resolve() == manifest_path.resolve(),
                 "launch-lock candidate path mismatch")
        actual_manifest_sha = sha(manifest_path)
        _require(actual_manifest_sha == lock.get("candidate_manifest_sha256"), "candidate manifest SHA mismatch")
        manifest = _object(manifest_path, "final candidate manifest")
        _require(namespace.resolve() == manifest_path.parent.resolve(), "real formal namespace mismatch")
        _require(manifest.get("experiment_id") == EXPECTED_EXPERIMENT, "candidate experiment mismatch")
        _require(manifest.get("gold_loaded") is False, "candidate permits gold")
        _require(lock.get("expected_route_count") == 600 and lock.get("expected_question_count") == 300,
                 "launch-lock population mismatch")
        _require(lock.get("expected_model_id") == EXPECTED_MODEL, "launch-lock model mismatch")

        config = _object(config_path, "Direct provider configuration")
        contract = manifest.get("scientific_contract")
        _require(isinstance(contract, dict), "candidate scientific contract missing")
        expected_contract = {
            "model_id": EXPECTED_MODEL, "max_new_images_per_turn": 3,
            "max_unique_images_per_question": 16, "provider_transport_retries": 1,
            "structural_correction_retries": 1, "max_route_turns": 32,
            "question_count": 300, "video_count": 12, "route_count": 600,
            "r1_route_count": 300, "r3_route_count": 300, "formal_hard_budget_usd": 50.0,
            "cache_configuration": EXPECTED_CACHE, "pricing_configuration": EXPECTED_PRICING,
        }
        _require(contract == expected_contract, "candidate scientific scalar/config contract mismatch")
        _require(config.get("model") == EXPECTED_MODEL and config.get("provider") == "anthropic", "provider/model config mismatch")
        _require(config.get("max_retries") == 1 and config.get("prompt_caching") == EXPECTED_CACHE,
                 "retry/cache config mismatch")
        _require(config.get("budget_policy") == {"max_new_images_per_turn": 3, "max_unique_physical_images_per_question": 16},
                 "3/16 provider config mismatch")
        _require(config.get("anthropic_cache_aware_pricing_usd_per_million_tokens") == EXPECTED_PRICING,
                 "pricing config mismatch")
        _require(manifest.get("question_count") == 300 and manifest.get("route_count") == 600 and
                 manifest.get("formal_hard_budget_usd") == 50.0, "candidate population/budget mismatch")
        _require(sum(route.get("method") == "R1" for route in manifest["routes"]) == 300 and
                 sum(route.get("method") == "R3" for route in manifest["routes"]) == 300,
                 "R1/R3 route count mismatch")

        current = fingerprint_factory(root, config, manifest, manifest_path.parent / "inputs")
        recorded = manifest.get("fingerprints")
        _require(isinstance(recorded, dict) and recorded == current, "implementation/configuration fingerprint mismatch")
        validate_population_and_inputs(manifest, manifest_path)
        mode = validate_namespace_state(namespace, manifest)
        return ValidatedLaunch(manifest, config, mode, actual_manifest_sha)
    except LaunchPreflightError:
        raise
    except Exception as error:
        raise LaunchPreflightError(f"formal launch preflight failed closed: {type(error).__name__}") from error
