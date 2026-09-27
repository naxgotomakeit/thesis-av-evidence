"""Fail-closed preflight for the isolated 175-route R1 visual-only correction."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from direct_api_prep.evidence import assert_safe_direct_evidence
from direct_api_v1.actions import parse_action
from direct_api_v1.anthropic_provider import direct_action_tools
from direct_api_v1.frame_resolver import FrozenFrameResolver
from direct_api_v1.maps import FORBIDDEN_QUESTION_KEYS
from direct_api_v1.policy import MAX_NEW_IMAGES_PER_TURN, MAX_UNIQUE_IMAGES_PER_QUESTION
from direct_api_v1.prompt import DIRECT_V1_PROMPT_VERSION, DIRECT_V1_SYSTEM_PROMPT


EXPECTED_EXPERIMENT = "direct_r1_visual_only_correction_v3_urllib_frozen"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"
EXPECTED_ROUTE_COUNT = 175
EXPECTED_VIDEO_COUNT = 7
LOCK_SCHEMA = "direct_r1_visual_only_correction_launch_lock_v3_urllib_frozen"
ALLOWED_AUDIO_PATH = re.compile(r"^coarse_regions\[\d+\]\.audio_channel$")


class CorrectionPreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class ValidatedCorrection:
    root: Path
    namespace: Path
    manifest: dict[str, Any]
    config: dict[str, Any]
    control: dict[str, Any]
    mode: str
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorrectionPreflightError(f"{label} missing or malformed: {path}") from error
    if not isinstance(value, dict):
        raise CorrectionPreflightError(f"{label} must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CorrectionPreflightError(message)


def recursive_differences(old: Any, new: Any, path: str = "") -> list[dict[str, Any]]:
    if type(old) is not type(new):
        return [{"path": path, "old": old, "new": new}]
    if isinstance(old, dict):
        if set(old) != set(new):
            return [{"path": path, "old_keys": sorted(old), "new_keys": sorted(new)}]
        result: list[dict[str, Any]] = []
        for key in old:
            child = f"{path}.{key}" if path else key
            result.extend(recursive_differences(old[key], new[key], child))
        return result
    if isinstance(old, list):
        if ALLOWED_AUDIO_PATH.fullmatch(path) and old != new:
            return [{"path": path, "old_count": len(old), "new_count": len(new)}]
        if len(old) != len(new):
            return [{"path": path, "old_count": len(old), "new_count": len(new)}]
        result = []
        for index, (left, right) in enumerate(zip(old, new)):
            result.extend(recursive_differences(left, right, f"{path}[{index}]"))
        return result
    return [] if old == new else [{"path": path, "old": old, "new": new}]


def audio_inventory(value: Any, path: str = "") -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if any(term in str(key).casefold() for term in ("audio", "asr", "speech", "transcript", "whisper")):
                result.append({"path": child_path, "kind": "key", "nonempty": bool(child)})
            result.extend(audio_inventory(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(audio_inventory(child, f"{path}[{index}]"))
    elif isinstance(value, str) and any(
        term in value.casefold() for term in ("audio", "asr", "speech", "transcript", "whisper")
    ):
        result.append({"path": path, "kind": "string_value", "value": value})
    return result


def validate_visual_only_pair(old_path: Path, new_path: Path, expect_byte_identical: bool) -> list[dict[str, Any]]:
    require(old_path.is_file() and new_path.is_file(), "map source/output missing")
    require(not new_path.is_symlink(), f"new map must not be a symlink: {new_path}")
    old = load_object(old_path, "old R1 map")
    new = load_object(new_path, "new R1 visual-only map")
    differences = recursive_differences(old, new)
    for difference in differences:
        require(ALLOWED_AUDIO_PATH.fullmatch(str(difference.get("path", ""))) is not None,
                f"non-whitelisted map difference: {difference}")
        require(difference.get("new_count") == 0, f"audio list not emptied: {difference}")
    old_nonempty = sum(bool(region.get("audio_channel")) for region in old.get("coarse_regions", []))
    if expect_byte_identical:
        require(old_nonempty == 0, "byte-copy map unexpectedly had nonempty audio")
        require(old_path.read_bytes() == new_path.read_bytes(), f"empty-audio map is not byte-identical: {new_path}")
        require(not differences, f"byte-copy map differs structurally: {new_path}")
    else:
        require(old_nonempty > 0, "rerun map did not contain source audio")
        require(differences, "rerun map has no allowed transformation")
    require(all(region.get("audio_channel") == [] for region in new.get("coarse_regions", [])),
            f"nonempty audio_channel remains: {new_path}")
    outside = [row for row in audio_inventory(new)
               if "audio_channel" not in row["path"] and row.get("path") != "map_type"]
    require(not outside, f"unexpected audio-derived content outside audio_channel: {outside[:3]}")
    require(new.get("map_type") == old.get("map_type") == "r1_av_structural_audio_dual_channel",
            "static schema label changed")
    assert_safe_direct_evidence(new)
    return differences


def validate_correction(root: Path) -> ValidatedCorrection:
    root = root.resolve()
    control = load_object(root / "config/correction_control.json", "correction control")
    config = load_object(root / "config/direct_v1_anthropic_smoke.json", "frozen Direct config")
    namespace = (root / control["run_namespace"]).resolve()
    manifest_path = namespace / "formal_manifest_visual_only_correction_v3_urllib_frozen.json"
    lock_path = namespace / "formal_launch_lock_visual_only_correction_v3_urllib_frozen.json"
    manifest = load_object(manifest_path, "correction candidate")
    lock = load_object(lock_path, "correction launch lock")
    code_freeze = load_object(root / "manifests/code_freeze_manifest.json", "code freeze manifest")
    closure_path = root / "manifests/execution_input_closure_v3.json"
    closure = load_object(closure_path, "execution input closure")
    reuse = load_object(root / "manifests/route_reuse_manifest_v3.json", "route partition/merge manifest")

    require(control.get("experiment_id") == EXPECTED_EXPERIMENT, "control experiment mismatch")
    require(manifest.get("experiment_id") == EXPECTED_EXPERIMENT, "manifest experiment mismatch")
    require(lock.get("schema_version") == LOCK_SCHEMA and lock.get("experiment_id") == EXPECTED_EXPERIMENT,
            "launch lock identity mismatch")
    require(Path(lock["candidate_manifest_path"]).resolve() == manifest_path.resolve(), "locked manifest path mismatch")
    manifest_sha = sha256_file(manifest_path)
    require(lock.get("candidate_manifest_sha256") == manifest_sha, "candidate SHA mismatch")
    require(lock.get("real_execution_allowed") is True, "real launch is not explicitly prepared")
    require(lock.get("required_transport") == control.get("required_transport") == "urllib_fallback",
            "urllib fallback transport is not frozen")
    require(float(lock.get("hard_budget_usd", -1)) == float(control["hard_budget_usd"]), "lock budget drift")
    require(manifest.get("gold_loaded") is False, "manifest permits gold")
    require(manifest.get("state") == "prepared_not_started_no_api", "candidate state mismatch")
    require(manifest.get("api_calls") == 0 and manifest.get("model_calls") == 0, "preparation call count is nonzero")
    require(float(manifest.get("hard_budget_usd", -1)) == float(control["hard_budget_usd"]), "manifest budget drift")
    require(manifest.get("concurrency") == control.get("concurrency") == 1, "concurrency contract drift")
    require(code_freeze.get("correction_control", {}).get("sha256") ==
            sha256_file(root / "config/correction_control.json"), "correction control SHA drift")
    require(manifest.get("execution_input_closure_manifest_sha256") == sha256_file(closure_path),
            "execution input closure file SHA drift")
    require(manifest.get("execution_input_closure_canonical_sha256") == canonical_sha(closure),
            "execution input closure canonical SHA drift")
    require(closure.get("experiment_id") == EXPECTED_EXPERIMENT, "closure experiment mismatch")
    for dependency in closure.get("dependencies", []):
        path = Path(dependency["path"])
        require(path.is_file(), f"closure dependency missing: {path}")
        require(sha256_file(path) == dependency["sha256"], f"closure dependency SHA drift: {path}")

    require(config.get("provider") == "anthropic" and config.get("model") == EXPECTED_MODEL,
            "provider/model mismatch")
    require(config.get("max_output_tokens") == 512 and config.get("timeout_sec") == 120 and
            config.get("max_retries") == 1, "provider scalar drift")
    require(config.get("budget_policy") == {
        "max_new_images_per_turn": 3, "max_unique_physical_images_per_question": 16,
    }, "3/16 budget drift")
    require(MAX_NEW_IMAGES_PER_TURN == 3 and MAX_UNIQUE_IMAGES_PER_QUESTION == 16,
            "frozen policy drift")
    require(DIRECT_V1_PROMPT_VERSION == "direct_v1.2", "prompt version drift")
    require(hashlib.sha256(DIRECT_V1_SYSTEM_PROMPT.encode()).hexdigest() ==
            manifest["protocol_identity"]["system_prompt_text_hash"], "system prompt text drift")
    require(hashlib.sha256(json.dumps(
                {str(n): direct_action_tools(n) for n in (3, 2, 1, 0)}, sort_keys=True
            ).encode()).hexdigest() ==
            manifest["protocol_identity"]["action_schema_hash"], "tool schema drift")
    parse_action({"action": "final_answer", "timestamps_sec": [], "selected_option_id": "A", "reason": "x"})

    for relative, expected in manifest["code_snapshot_sha256"].items():
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"code snapshot missing/symlink: {relative}")
        require(sha256_file(path) == expected, f"code snapshot SHA drift: {relative}")
    require(sha256_file(root / "config/direct_v1_anthropic_smoke.json") ==
            manifest["protocol_identity"]["provider_config_hash"], "provider config SHA drift")

    routes = manifest.get("routes")
    videos = manifest.get("videos")
    require(isinstance(routes, list) and len(routes) == EXPECTED_ROUTE_COUNT, "active route count mismatch")
    require(isinstance(videos, list) and len(videos) == EXPECTED_VIDEO_COUNT, "active video count mismatch")
    require(len({row.get("route_id") for row in routes}) == EXPECTED_ROUTE_COUNT, "route IDs not unique")
    require(all(row.get("route_id") == "R1VO3:" + row.get("question_id", "") and row.get("method") == "R1"
                for row in routes), "route identity/method mismatch")
    active_uids = {row["video_id"] for row in routes}
    require(active_uids == {row["video_id"] for row in videos}, "active video set mismatch")
    require(reuse.get("rerun", {}).get("count") == 175 and
            reuse.get("reuse_r1", {}).get("count") == 125 and
            reuse.get("reuse_r3", {}).get("count") == 300, "175+125/300 partition drift")
    require({row["route_id"] for row in reuse["rerun"]["routes"]} ==
            {row["route_id"] for row in routes}, "rerun partition does not match candidate")
    rerun_qids = {row["question_id"] for row in routes}
    reused_qids = {row["question_id"] for row in reuse["reuse_r1"]["routes"]}
    r3_qids = {row["question_id"] for row in reuse["reuse_r3"]["routes"]}
    require(not (rerun_qids & reused_qids) and len(rerun_qids | reused_qids) == 300,
            "R1 partition is not disjoint Eval300")
    require(rerun_qids | reused_qids == r3_qids, "R1/R3 paired question set drift")
    for method, group in (("R1", reuse["reuse_r1"]["routes"]),
                          ("R3", reuse["reuse_r3"]["routes"])):
        for source in group:
            source_path = Path(source["source_artifact"])
            require(source_path.is_file() and sha256_file(source_path) == source["source_artifact_sha256"],
                    f"reused {method} artifact identity drift: {source_path}")
            source_artifact = load_object(source_path, f"reused {method} artifact")
            require(source_artifact.get("question_id") == source["question_id"] and
                    source_artifact.get("method") == method, f"reused {method} artifact content mismatch")
            require(source.get("reuse_eligible") is True, f"reused {method} route not frozen eligible")

    video_by_id = {row["video_id"]: row for row in videos}
    for video in videos:
        map_path = Path(video["r1_visual_only_map"])
        old_path = Path(video["original_r1_map"])
        require(map_path.resolve().is_relative_to((root / "maps/r1_visual_only").resolve()),
                "new map escaped isolated workspace")
        require(sha256_file(map_path) == video["r1_visual_only_sha256"], "new map SHA mismatch")
        require(sha256_file(old_path) == video["original_r1_sha256"], "old map SHA mismatch")
        validate_visual_only_pair(old_path, map_path, expect_byte_identical=False)
        hierarchy = Path(video["hierarchy"])
        require(sha256_file(hierarchy) == video["hierarchy_sha256"], "hierarchy SHA mismatch")
        frame_manifest_path = Path(video["frame_sha_manifest"])
        require(sha256_file(frame_manifest_path) == video["frame_sha_manifest_sha256"],
                "frame SHA manifest drift")
        frame_manifest = load_object(frame_manifest_path, "frame SHA manifest")
        require(frame_manifest.get("source_evidence") == "current_prelaunch_snapshot_not_historical_freeze",
                "frame evidence limitation not recorded")
        require(frame_manifest.get("historical_full_frame_sha_manifest_available") is False,
                "historical full-frame SHA evidence status mismatch")
        resolver = FrozenFrameResolver.from_hierarchy(load_object(hierarchy, "hierarchy"), frame_manifest)
        require(resolver.video_uid == video["video_id"], "resolver video mismatch")

    input_root = namespace / "inputs"
    for route in routes:
        require(route["video_id"] in video_by_id, "route video missing")
        require(route["map_path"] == video_by_id[route["video_id"]]["r1_visual_only_map"], "route map mismatch")
        require(route["map_sha256"] == video_by_id[route["video_id"]]["r1_visual_only_sha256"], "route map SHA mismatch")
        question_path = input_root / f"{route['question_id']}.json"
        require(question_path.is_file() and not question_path.is_symlink(), "question input missing/symlink")
        question = load_object(question_path, "question input")
        require(question.get("question_id") == route["question_id"], "question identity mismatch")
        require(sha256_file(question_path) == route["question_sha256"], "question/options SHA drift")
        require(not (set(question) & FORBIDDEN_QUESTION_KEYS), "gold field in question input")
        require([row.get("option_id") for row in question.get("answer_options", [])] == list("ABCDE"),
                "answer-option protocol mismatch")
        assert_safe_direct_evidence(question)

    ledger = load_object(namespace / "budget_ledger.json", "correction budget ledger")
    require(ledger.get("schema_version") == "reserved_budget_ledger_v2", "ledger schema mismatch")
    require(float(ledger.get("cap_usd", -1)) == float(control["hard_budget_usd"]), "ledger cap mismatch")
    require(0 <= float(ledger.get("spent_usd", -1)) <= float(ledger["cap_usd"]), "ledger spend invalid")
    require(0 <= float(ledger.get("reserved_usd", -1)) and
            float(ledger["spent_usd"]) + float(ledger["reserved_usd"]) <= float(ledger["cap_usd"]),
            "ledger reservation invalid")
    require(isinstance(ledger.get("attempts"), dict), "ledger attempt registry invalid")
    require(isinstance(ledger.get("attempt_ids"), list) and len(ledger["attempt_ids"]) == len(set(ledger["attempt_ids"])),
            "ledger attempt IDs invalid")
    statuses = list((namespace / "route_status").glob("*.json"))
    artifacts = list((namespace / "route_artifacts").glob("*.json"))
    journal_rows = []
    for name in ("request_start.jsonl", "attempt_end.jsonl", "controller_result.jsonl", "lifecycle.jsonl"):
        path = namespace / "journals" / name
        require(path.is_file() and not path.is_symlink(), f"journal missing/symlink: {name}")
        journal_rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    route_ids = {row["route_id"] for row in routes}
    require(all(row.get("experiment_id") == EXPECTED_EXPERIMENT and row.get("route_id") in route_ids
                for row in journal_rows), "foreign/old journal state present")
    require(all(load_object(path, "route status").get("route_id") in route_ids for path in statuses),
            "foreign/old route status present")
    require(all(path.stem in route_ids for path in artifacts), "foreign/old route artifact present")
    first_launch = not statuses and not artifacts and not journal_rows
    if first_launch:
        require(ledger["spent_usd"] == 0 and ledger["reserved_usd"] == 0 and ledger["attempts"] == {} and
                ledger["attempt_ids"] == [] and ledger.get("charges") == [],
                "first-launch ledger is not empty")
    return ValidatedCorrection(root, namespace, manifest, config, control,
                               "first_launch" if first_launch else "own_namespace_resume", manifest_sha)
