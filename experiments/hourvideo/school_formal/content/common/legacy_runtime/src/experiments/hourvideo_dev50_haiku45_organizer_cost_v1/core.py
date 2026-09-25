from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import anthropic

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.common import load_json, sha256_file, write_json
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import (
    ORGANIZER_SYSTEM,
    _api_schema,
    _organizer_schema,
)
from experiments.hourvideo_r1_av_r3_2_single_video_smoke.local_prepare import attach_audio


PRICING_SOURCE = "https://www.anthropic.com/claude/haiku"
PRICING_VERIFIED_DATE = "2026-08-11"


class OrganizerCallError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any], raw_text: str) -> None:
        super().__init__(message)
        self.usage = usage
        self.raw_text = raw_text


def _load_dedicated_key(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"dedicated API environment file missing: {path}")
    value = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("ANTHROPIC_API_KEY="):
            value = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not value:
        raise RuntimeError("ANTHROPIC_API_KEY is empty in the dedicated environment file")
    # Deliberately replace any inherited key so this experiment cannot be billed to the old key.
    os.environ["ANTHROPIC_API_KEY"] = value


def _cost_record(usage: dict[str, Any], pricing: dict[str, float]) -> dict[str, float]:
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    input_usd = input_tokens * float(pricing["input"]) / 1_000_000
    output_usd = output_tokens * float(pricing["output"]) / 1_000_000
    return {
        "input_usd": input_usd,
        "output_usd": output_usd,
        "total_usd": input_usd + output_usd,
    }


def _caption_cases(root: Path, cfg: dict[str, Any]) -> list[str]:
    caption_root = root / cfg["caption_root"]
    return sorted(path.stem for path in (caption_root / "case_configs").glob("*.json"))


def _audio_sources(root: Path, cfg: dict[str, Any], uids: list[str]) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    selected_hashes: dict[str, str] = {}
    allowed = set(uids)
    for relative_root in cfg["audio_source_roots"]:
        source_root = root / relative_root
        for path in sorted(source_root.glob("*/audio_asr.json")):
            uid = path.parent.name
            if uid not in allowed:
                continue
            segments = load_json(path).get("segments", [])
            if not segments:
                continue
            digest = sha256_file(path)
            if uid in selected and digest != selected_hashes[uid]:
                raise RuntimeError(f"conflicting non-empty ASR artifacts for {uid}")
            selected.setdefault(uid, path)
            selected_hashes.setdefault(uid, digest)
    return selected


def preflight(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    output_root = root / cfg["output_root"]
    caption_root = root / cfg["caption_root"]
    uids = _caption_cases(root, cfg)
    audio = _audio_sources(root, cfg, uids)
    cases: list[dict[str, Any]] = []
    failures: list[str] = []
    for uid in uids:
        case_root = caption_root / "cases" / uid
        required = [
            case_root / "shared_hierarchy.json",
            case_root / "r3_medium_captions.json",
            case_root / "r3_caption_validation.json",
        ]
        if any(not path.is_file() for path in required):
            failures.append(f"{uid}: missing caption inputs")
            continue
        hierarchy = load_json(required[0])
        captions = load_json(required[1])
        validation = load_json(required[2])
        if not validation.get("valid") or len(captions) != len(hierarchy["medium_nodes"]):
            failures.append(f"{uid}: invalid caption/Medium coverage")
            continue
        audio_path = audio.get(uid)
        audio_segments = load_json(audio_path)["segments"] if audio_path else []
        cases.append({
            "video_uid": uid,
            "duration_sec": hierarchy["duration_sec"],
            "medium_count": len(captions),
            "caption_path": str(required[1].resolve()),
            "caption_sha256": sha256_file(required[1]),
            "asr_available": bool(audio_path),
            "asr_segment_count": len(audio_segments),
            "asr_path": str(audio_path.resolve()) if audio_path else None,
            "asr_sha256": sha256_file(audio_path) if audio_path else None,
        })
    document = {
        "experiment": cfg["experiment"],
        "caption_source": "local_qwen2_5_vl_7b_three_ordered_frames_per_45s_medium",
        "videoseal_outputs_used": False,
        "api_navigation_maps_used": False,
        "visual_only_case_count": len(cases),
        "visual_audio_case_count": sum(row["asr_available"] for row in cases),
        "failures": failures,
        "cases": cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "preflight.json", document)
    if failures or len(cases) != int(cfg["expected_visual_only_cases"]):
        raise RuntimeError(f"Organizer preflight failed: {len(failures)} failures, {len(cases)} valid cases")
    if sum(row["asr_available"] for row in cases) != int(cfg["expected_visual_audio_cases"]):
        raise RuntimeError("Organizer preflight ASR case count changed")
    return document


def _organizer_payload(hierarchy: dict[str, Any], captions: list[dict[str, Any]], audio: list[dict[str, Any]]) -> dict[str, Any]:
    attached = attach_audio(hierarchy["medium_nodes"], audio)
    timeline = []
    for index, (medium, caption) in enumerate(zip(hierarchy["medium_nodes"], captions)):
        if medium["medium_id"] != caption["medium_id"]:
            raise RuntimeError("Caption/Medium order mismatch")
        timeline.append({
            "medium_index": index,
            "interval": [medium["start_sec"], medium["end_sec"]],
            "caption": caption["qwen_caption"],
            "overlapping_asr": attached[medium["medium_id"]],
        })
    return {"timeline": timeline, "contract": {"global_view": True, "storyline": False, "hard_filtering": False}}


def _map_from_output(
    hierarchy: dict[str, Any], captions: list[dict[str, Any]], audio: list[dict[str, Any]], raw: dict[str, Any],
) -> dict[str, Any]:
    timeline_count = len(hierarchy["medium_nodes"])
    ends = [row["end_medium_index"] for row in raw["groups"]]
    if ends != sorted(set(ends)) or not ends or ends[-1] != timeline_count - 1:
        raise RuntimeError(f"Invalid first-pass Organizer ends: {ends}")
    groups = []
    start = 0
    for index, item in enumerate(raw["groups"], 1):
        end = item["end_medium_index"]
        media = hierarchy["medium_nodes"][start:end + 1]
        group_captions = captions[start:end + 1]
        aud = [
            row for row in audio
            if float(row["start_sec"]) < float(media[-1]["end_sec"])
            and float(row["end_sec"]) > float(media[0]["start_sec"])
        ]
        groups.append({
            "coarse_id": f"C{index:02d}",
            "start_sec": media[0]["start_sec"],
            "end_sec": media[-1]["end_sec"],
            "source_medium_ids": [row["medium_id"] for row in media],
            "navigation_summary": item["navigation_summary"],
            "uncertainty_notes": item["uncertainty_notes"],
            "exact_source_captions": group_captions,
            "exact_source_asr": aud,
            "semantic_summary": True,
        })
        start = end + 1
    return {
        "map_type": "r3_2_global_av_semantic_coarse",
        "semantic_fields_available": True,
        "hard_filtering_allowed": False,
        "storyline_events": [],
        "has_storyline": False,
        "coarse_regions": groups,
    }


def _call_once(cfg: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    response = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"], timeout=float(cfg["anthropic"]["timeout_sec"]),
    ).messages.create(
        model=cfg["anthropic"]["model"],
        max_tokens=int(cfg["anthropic"]["organizer_max_tokens"]),
        temperature=float(cfg["anthropic"]["temperature"]),
        system=ORGANIZER_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
        output_config={"format": {"type": "json_schema", "schema": _api_schema(_organizer_schema(len(payload["timeline"]))) }},
    )
    raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    usage = {
        "provider": "anthropic",
        "model": cfg["anthropic"]["model"],
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "cache_creation_input_tokens": int(getattr(response.usage, "cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(response.usage, "cache_read_input_tokens", 0) or 0),
        "latency_sec": time.perf_counter() - started,
        "response_id": str(response.id),
        "stop_reason": str(response.stop_reason),
        "raw_text": raw_text,
    }
    if response.stop_reason == "max_tokens":
        raise OrganizerCallError("Anthropic Organizer response reached max_tokens", usage, raw_text)
    try:
        return json.loads(raw_text), usage
    except json.JSONDecodeError as error:
        raise OrganizerCallError("Anthropic Organizer returned invalid JSON", usage, raw_text) from error


def run_condition(
    root: Path, config_path: Path, condition: str, resume: bool = True, case_uid: str | None = None,
) -> dict[str, Any]:
    if condition not in {"visual_only", "visual_audio"}:
        raise ValueError(condition)
    cfg = load_json(config_path)
    _load_dedicated_key(root / cfg["dedicated_env_file"])
    pre = preflight(root, config_path)
    caption_root = root / cfg["caption_root"]
    output_root = root / cfg["output_root"] / condition
    selected = [row for row in pre["cases"] if condition == "visual_only" or row["asr_available"]]
    if case_uid is not None:
        selected = [row for row in selected if row["video_uid"] == case_uid]
        if len(selected) != 1:
            raise RuntimeError(f"case {case_uid} is unavailable for condition {condition}")
    completed = 0
    failed = 0
    for row in selected:
        uid = row["video_uid"]
        case_out = output_root / "cases" / uid
        usage_path = case_out / "organizer_usage.json"
        map_path = case_out / "r3_2_navigation_map.json"
        if resume and usage_path.is_file() and map_path.is_file():
            completed += 1
            continue
        case_out.mkdir(parents=True, exist_ok=True)
        source = caption_root / "cases" / uid
        hierarchy = load_json(source / "shared_hierarchy.json")
        captions = load_json(source / "r3_medium_captions.json")
        audio = load_json(Path(row["asr_path"]))["segments"] if condition == "visual_audio" else []
        payload = _organizer_payload(hierarchy, captions, audio)
        attempts = []
        success = False
        for attempt in range(1, int(cfg["max_attempts_per_case"]) + 1):
            attempt_started = time.perf_counter()
            raw: dict[str, Any] | None = None
            usage: dict[str, Any] | None = None
            try:
                raw, usage = _call_once(cfg, payload)
                map_doc = _map_from_output(hierarchy, captions, audio, raw)
                usage.update({
                    "stage": "organizer",
                    "condition": condition,
                    "video_uid": uid,
                    "attempt": attempt,
                    "medium_count": len(captions),
                    "asr_segment_count": len(audio),
                    "coarse_count": len(map_doc["coarse_regions"]),
                    "pricing_usd_per_million": cfg["anthropic"]["pricing_usd_per_million"],
                    "pricing_source": PRICING_SOURCE,
                    "pricing_verified_date": PRICING_VERIFIED_DATE,
                    "estimated_cost": _cost_record(usage, cfg["anthropic"]["pricing_usd_per_million"]),
                })
                attempts.append({"attempt": attempt, "status": "completed", "response_id": usage["response_id"], "elapsed_sec": usage["latency_sec"]})
                write_json(case_out / "organizer_input.json", payload)
                write_json(case_out / "organizer_raw_output.json", raw)
                write_json(map_path, map_doc)
                write_json(usage_path, usage)
                write_json(case_out / "attempts.json", attempts)
                success = True
                completed += 1
                break
            except Exception as error:
                # Preserve usage/raw output whether the failure happened inside the provider call
                # or afterwards during the unchanged Organizer map-contract validation.
                failed_usage = getattr(error, "usage", None) or usage
                failed_raw_text = getattr(error, "raw_text", "")
                if not failed_raw_text and raw is not None:
                    failed_raw_text = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
                failed_record = {
                    "attempt": attempt,
                    "status": "failed",
                    "elapsed_sec": time.perf_counter() - attempt_started,
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                    "usage_available": failed_usage is not None,
                }
                if failed_usage is not None:
                    failed_usage = dict(failed_usage)
                    failed_usage["estimated_cost"] = _cost_record(
                        failed_usage, cfg["anthropic"]["pricing_usd_per_million"],
                    )
                    failed_record["usage"] = failed_usage
                    write_json(case_out / f"failed_response_attempt_{attempt}.json", {
                        "usage": failed_usage,
                        "raw_text": failed_raw_text,
                    })
                attempts.append(failed_record)
                write_json(case_out / "attempts.json", attempts)
        if not success:
            failed += 1
            write_json(case_out / "error.json", {"video_uid": uid, "condition": condition, "attempts": attempts})
        write_json(output_root / "progress.json", {
            "condition": condition,
            "selected": len(selected),
            "completed": completed,
            "failed": failed,
        })
    result = summarize(root, config_path)
    if failed:
        raise RuntimeError(f"{condition} Organizer failed for {failed} cases")
    return result


def summarize(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load_json(config_path)
    experiment_root = root / cfg["output_root"]
    conditions: dict[str, Any] = {}
    per_uid: dict[str, dict[str, Any]] = {}
    for condition in ("visual_only", "visual_audio"):
        rows = []
        for path in sorted((experiment_root / condition / "cases").glob("*/organizer_usage.json")):
            usage = load_json(path)
            rows.append(usage)
            per_uid.setdefault(usage["video_uid"], {})[condition] = usage
        conditions[condition] = {
            "completed_cases": len(rows),
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "latency_sec": sum(row["latency_sec"] for row in rows),
            "estimated_usd": sum(row["estimated_cost"]["total_usd"] for row in rows),
        }
    paired = []
    for uid, modes in sorted(per_uid.items()):
        if not {"visual_only", "visual_audio"} <= modes.keys():
            continue
        visual = modes["visual_only"]
        av = modes["visual_audio"]
        paired.append({
            "video_uid": uid,
            "asr_segment_count": av["asr_segment_count"],
            "visual_only_input_tokens": visual["input_tokens"],
            "visual_audio_input_tokens": av["input_tokens"],
            "input_token_delta": av["input_tokens"] - visual["input_tokens"],
            "visual_only_output_tokens": visual["output_tokens"],
            "visual_audio_output_tokens": av["output_tokens"],
            "output_token_delta": av["output_tokens"] - visual["output_tokens"],
            "visual_only_usd": visual["estimated_cost"]["total_usd"],
            "visual_audio_usd": av["estimated_cost"]["total_usd"],
            "cost_delta_usd": av["estimated_cost"]["total_usd"] - visual["estimated_cost"]["total_usd"],
            "visual_only_coarse_count": visual["coarse_count"],
            "visual_audio_coarse_count": av["coarse_count"],
        })
    summary = {
        "experiment": cfg["experiment"],
        "model": cfg["anthropic"]["model"],
        "pricing_usd_per_million": cfg["anthropic"]["pricing_usd_per_million"],
        "pricing_source": PRICING_SOURCE,
        "pricing_verified_date": PRICING_VERIFIED_DATE,
        "conditions": conditions,
        "paired_case_count": len(paired),
        "paired_comparison": paired,
        "provider_dashboard_reconciliation_note": "Failed requests without a response usage object are not locally billable; reconcile the dedicated key dashboard total against this summary.",
    }
    experiment_root.mkdir(parents=True, exist_ok=True)
    write_json(experiment_root / "summary.json", summary)
    return summary
