"""Single-call Claude Haiku audio-visual answer baseline for one frozen video."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments/egopolice_av_answer_v0_1.json"

SYSTEM_PROMPT = """You are generating a provisional structured incident answer from supplied evidence.
Use only the supplied visual and speech evidence.
Do not infer intent, legality, identity, causation, or unseen events.
Distinguish visual evidence from ASR evidence. ASR may contain transcription errors.
When evidence is incomplete or conflicting, answer unclear.
Every important claim must include a supporting timestamp.
Do not treat the existing visual storyline as ground truth; it is only another potentially imperfect input.
Ignore irrelevant visual changes such as camera occlusion, sleeves near the lens, repetitive driving scenes, or motion blur unless relevant to the incident.
Use the unified timestamps to reason about overlap, lead, lag, and sequence. A spoken command does not prove the corresponding object or action occurred.
Return JSON only, matching the requested schema. Do not include Markdown fences. Keep the answer compact: no repeated evidence, maximum 5 summary sentences, and at most 8 timeline items, 8 officer actions, 8 civilian actions, and 10 speech items. Supporting evidence is one sentence maximum. Do not return an empty timeline unless the supplied evidence contains no identifiable event; use the visual and ASR timestamps to construct the timeline. Do not use placeholder objects, empty strings, or empty arrays when the supplied evidence supports an answer."""

QUESTIONS = """Fixed provisional schema v0.1. The JSON schema below is authoritative:
Q1 incident_summary: What happened in this incident? Use 3–5 sentences; ignore irrelevant driving, occlusion, and repetition.
Q2 timeline: Main stages in chronological order, maximum 8 items. Each item must contain only time_range (string), event, visual_evidence, speech_evidence, temporal_relation (overlap|lead|lag|sequence|unclear), and confidence (high|medium|low). Do not return empty timeline items.
Q3 officer_actions and civilian_actions: maximum 8 items each; observable facts only; do not infer motive, legality, or reasonableness. Supporting evidence is one sentence maximum.
Q4 important_speech: maximum 10 items; timestamp and original ASR text; status clear or unclear; do not silently correct unreliable ASR.
Q5 critical_events: include all six fixed fields weapon_visible, gunshot_or_firearm_discharge, physical_confrontation, force_or_restraint, arrest_or_handcuffing, injury_or_medical_assistance. Each value must be one compact string exactly in the form "status | timestamp | evidence | modality". Status is yes/no/unclear. Timestamp and evidence must be concrete or explicitly "unknown". Use no only when evidence supports absence; otherwise unclear.
Q6 uncertainty_or_missing_information: maximum 6 concise items; explicitly cover ASR uncertainty, visually obscured events, missing temporal context, conflicts between speech and visual captions, and unverifiable claims. Do not repeat evidence.

Required top-level JSON keys:
video_id, incident_summary, timeline, officer_actions, civilian_actions, important_speech, critical_events, uncertainty_or_missing_information, evidence_conflicts."""

_TIMELINE_ITEM = {"type": "object", "additionalProperties": False, "required": ["time_range", "event", "visual_evidence", "speech_evidence", "temporal_relation", "confidence"], "properties": {"time_range": {"type": "string"}, "event": {"type": "string"}, "visual_evidence": {"type": "string"}, "speech_evidence": {"type": "string"}, "temporal_relation": {"type": "string"}, "confidence": {"type": "string"}}}
_ACTION_ITEM = {"type": "object", "additionalProperties": False, "required": ["timestamp_sec", "action", "supporting_evidence"], "properties": {"timestamp_sec": {"type": "number"}, "action": {"type": "string"}, "supporting_evidence": {"type": "string"}}}
_SPEECH_ITEM = {"type": "object", "additionalProperties": False, "required": ["start_sec", "end_sec", "text", "interpretation", "status"], "properties": {"start_sec": {"type": "number"}, "end_sec": {"type": "number"}, "text": {"type": "string"}, "interpretation": {"type": "string"}, "status": {"type": "string"}}}
_CRITICAL_NAMES = ["weapon_visible", "gunshot_or_firearm_discharge", "physical_confrontation", "force_or_restraint", "arrest_or_handcuffing", "injury_or_medical_assistance"]
OUTPUT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["video_id", "incident_summary", "timeline", "officer_actions", "civilian_actions", "important_speech", "critical_events", "uncertainty_or_missing_information", "evidence_conflicts"], "properties": {
    "video_id": {"type": "string"}, "incident_summary": {"type": "string"},
    "timeline": {"type": "array", "items": _TIMELINE_ITEM},
    "officer_actions": {"type": "array", "items": _ACTION_ITEM}, "civilian_actions": {"type": "array", "items": _ACTION_ITEM},
    "important_speech": {"type": "array", "items": _SPEECH_ITEM},
    "critical_events": {"type": "object", "additionalProperties": False, "required": _CRITICAL_NAMES, "properties": {key: {"type": "string"} for key in _CRITICAL_NAMES}},
    "uncertainty_or_missing_information": {"type": "array", "items": {"type": "string"}}, "evidence_conflicts": {"type": "array", "items": {"type": "string"}}
}}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_id(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value.replace("/", "__")).strip("._-")


def load_json(path_value: str) -> tuple[Path, Any]:
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def dedupe_rows(rows: list[dict[str, Any]], text_key: str) -> list[dict[str, Any]]:
    seen: set[str] = set(); result = []
    for row in sorted(rows, key=lambda x: (float(x.get("start_sec", x.get("start", x.get("start_time", 0.0)))), str(x.get("medium_id", x.get("transcript_segment_id", ""))))):
        text = str(row.get(text_key, "")).strip()
        if not text or text in seen:
            continue
        seen.add(text); result.append(row)
    return result


def build_evidence(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    medium_path, medium_raw = load_json(config["visual_medium_captions"])
    story_path, story_raw = load_json(config["visual_storyline"])
    asr_path, asr_raw = load_json(config["timestamped_asr"])
    align_path, align_raw = load_json(config["audio_medium_alignment"])
    medium_rows = dedupe_rows(medium_raw if isinstance(medium_raw, list) else medium_raw.get("records", []), "cleaned_caption")
    medium = [{"start_sec": float(x["start"]), "end_sec": float(x["end"]), "medium_id": x["medium_id"], "caption": str(x.get("cleaned_caption", "")).strip()} for x in medium_rows]
    asr_rows = dedupe_rows(asr_raw.get("transcript_segments", []), "text")
    asr = []
    for x in asr_rows:
        metadata = x.get("whisper_metadata", {})
        asr.append({"start_sec": float(x["start_time"]), "end_sec": float(x["end_time"]), "text": str(x.get("text", "")).strip(), "status": "unclear" if x.get("warnings") or metadata.get("no_speech_prob", 0.0) > 0.6 else "clear"})
    alignment = align_raw.get("medium_alignments", align_raw.get("alignments", []))
    alignment_by_id = {str(x.get("medium_id")): x for x in alignment}
    windows = []
    for item in medium:
        overlaps = [x for x in asr if x["end_sec"] > item["start_sec"] and x["start_sec"] < item["end_sec"]]
        relation = alignment_by_id.get(str(item["medium_id"]), {})
        # Keep one compact per-Medium evidence window. The alignment transcript is
        # intentionally omitted when it duplicates these overlapping ASR segments.
        relation_fields = {key: relation.get(key) for key in ("speech_duration_sec", "speech_overlap_ratio", "relation", "temporal_relation") if relation.get(key) is not None}
        windows.append({"medium_id": item["medium_id"], "start_sec": item["start_sec"], "end_sec": item["end_sec"], "visual_caption": item["caption"], "overlapping_asr": overlaps, "audio_visual_relation": relation_fields})
    storyline = {"overall_story": story_raw.get("overall_story", ""), "high_level_storyline": story_raw.get("high_level_storyline", [])}
    evidence = {"video_id": config["dataset_video_id"], "duration_sec": float(config["duration_sec"]), "medium_windows": windows, "chronological_visual_storyline": storyline}
    hashes = {"medium_captions": sha256_file(medium_path), "visual_storyline": sha256_file(story_path), "timestamped_asr": sha256_file(asr_path), "audio_medium_alignment": sha256_file(align_path)}
    return evidence, hashes


def validate_answer(answer: Any, video_id: str) -> list[str]:
    errors: list[str] = []
    required = ["video_id", "incident_summary", "timeline", "officer_actions", "civilian_actions", "important_speech", "critical_events", "uncertainty_or_missing_information", "evidence_conflicts"]
    if not isinstance(answer, dict): return ["top_level_not_object"]
    errors.extend(f"missing:{key}" for key in required if key not in answer)
    if answer.get("video_id") != video_id: errors.append("video_id_mismatch")
    if not isinstance(answer.get("incident_summary"), str) or not answer.get("incident_summary", "").strip(): errors.append("incident_summary_empty")
    for key, limit in [("timeline", 8), ("officer_actions", 8), ("civilian_actions", 8), ("important_speech", 10), ("uncertainty_or_missing_information", 6), ("evidence_conflicts", 6)]:
        if not isinstance(answer.get(key), list): errors.append(f"{key}_not_array")
        elif len(answer[key]) > limit: errors.append(f"{key}_exceeds_{limit}")
    critical = ["weapon_visible", "gunshot_or_firearm_discharge", "physical_confrontation", "force_or_restraint", "arrest_or_handcuffing", "injury_or_medical_assistance"]
    critical_events = answer.get("critical_events")
    if not isinstance(critical_events, dict):
        errors.append("critical_events_not_object")
        critical_events = {}
    for key in critical:
        value = critical_events.get(key)
        if not isinstance(value, str):
            errors.append(f"critical_event_missing_or_not_string:{key}")
            continue
        parts = [part.strip() for part in value.split("|", 3)]
        if len(parts) != 4: errors.append(f"critical_event_contract:{key}"); continue
        if parts[0] not in {"yes", "no", "unclear"}: errors.append(f"invalid_status:{key}")
        if not parts[1] or not parts[2]: errors.append(f"critical_event_timestamp_evidence_empty:{key}")
    for index, item in enumerate(answer.get("timeline", [])):
        if not isinstance(item, dict): errors.append(f"timeline_{index}_not_object"); continue
        for key in ["time_range", "event", "visual_evidence", "speech_evidence", "temporal_relation", "confidence"]:
            if key not in item: errors.append(f"timeline_{index}_missing:{key}")
        for key in ["time_range", "event", "visual_evidence", "speech_evidence"]:
            if key in item and (not isinstance(item[key], str) or not item[key].strip()): errors.append(f"timeline_{index}_empty:{key}")
        if item.get("confidence") not in {"high", "medium", "low"}: errors.append(f"timeline_{index}_confidence")
        if item.get("temporal_relation") not in {"overlap", "lead", "lag", "sequence", "unclear"}: errors.append(f"timeline_{index}_temporal_relation")
    for group in ["officer_actions", "civilian_actions"]:
        for index, item in enumerate(answer.get(group, [])):
            if not isinstance(item, dict): errors.append(f"{group}_{index}_not_object"); continue
            for key in ["timestamp_sec", "action", "supporting_evidence"]:
                if key not in item or item[key] in (None, ""): errors.append(f"{group}_{index}_missing:{key}")
    for index, item in enumerate(answer.get("important_speech", [])):
        if not isinstance(item, dict): errors.append(f"speech_{index}_not_object"); continue
        for key in ["start_sec", "end_sec", "text", "interpretation", "status"]:
            if key not in item: errors.append(f"speech_{index}_missing:{key}")
        if item.get("status") not in {"clear", "unclear"}: errors.append(f"speech_{index}_status")
    return errors


def parse_json(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(cleaned)


def append_api_call(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def api_records_for_run(path: Path, video_id: str, started_at: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("video_id") == video_id and str(record.get("timestamp_utc", "")) >= started_at:
            records.append(record)
    return records


def call_haiku(client: Any, *, model: str, system: str, prompt: str, purpose: str, retry_index: int, api_log: Path, video_id: str, prompt_hash: str, evidence_hash: str, raw_path: Path, structured_path: Path, max_tokens: int) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter(); record: dict[str, Any] = {"call_id": f"{video_id.replace('/', '__')}__{purpose}__{retry_index}", "video_id": video_id, "purpose": purpose, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "model": model, "input_tokens": None, "output_tokens": None, "total_tokens": None, "cache_creation_input_tokens": None, "cache_read_input_tokens": None, "latency_seconds": None, "stop_reason": None, "request_id": None, "http_status": None, "retry_index": retry_index, "success": False, "error_type": None, "error_message": None, "prompt_version": "egopolice_av_answer_v0.1", "prompt_sha256": prompt_hash, "evidence_packet_sha256": evidence_hash, "raw_response_path": raw_path.as_posix(), "structured_output_path": structured_path.as_posix(), "structured_outputs": True, "max_tokens": max_tokens}
    try:
        message = client.messages.create(model=model, max_tokens=max_tokens, temperature=0, system=system, messages=[{"role": "user", "content": prompt}], output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}})
        text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
        usage = getattr(message, "usage", None); stop_reason = getattr(message, "stop_reason", None); record.update({"input_tokens": getattr(usage, "input_tokens", None), "output_tokens": getattr(usage, "output_tokens", None), "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None), "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None), "stop_reason": stop_reason, "request_id": getattr(message, "_request_id", None), "http_status": 200, "success": stop_reason != "max_tokens", "error_type": "max_tokens_truncation" if stop_reason == "max_tokens" else None, "error_message": "structured output reached max_tokens" if stop_reason == "max_tokens" else None})
        return text, record
    except Exception as exc:
        record.update({"http_status": getattr(exc, "status_code", None), "request_id": getattr(exc, "request_id", None), "error_type": type(exc).__name__, "error_message": str(exc)[:1000]})
        raise
    finally:
        record["latency_seconds"] = time.perf_counter() - started
        record["total_tokens"] = (record["input_tokens"] or 0) + (record["output_tokens"] or 0) if record["input_tokens"] is not None or record["output_tokens"] is not None else None
        append_api_call(api_log, record)


def main() -> int:
    # Shell variables take precedence because override=False is explicit.
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is missing from the shell environment and project-root .env")
    import anthropic
    config_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else CONFIG
    if not config_path.is_absolute(): config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8")); output = ROOT / config["output_root"] / safe_id(config["dataset_video_id"]); output.mkdir(parents=True, exist_ok=True); api_log = ROOT / config["output_root"] / "api_calls.jsonl"; model = os.environ.get("ANTHROPIC_MODEL", config.get("default_model", "claude-haiku-4-5"))
    evidence, source_hashes = build_evidence(config); evidence_text = json.dumps(evidence, ensure_ascii=False, indent=2); evidence_hash = sha256_text(evidence_text); prompt = f"{QUESTIONS}\n\nSupplied evidence packet:\n{evidence_text}\n\nReturn the fixed JSON schema now. Every important claim must be timestamped."
    prompt_hash = sha256_text(SYSTEM_PROMPT + "\n" + prompt); write_json = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_json(output / "input_evidence_packet.json", evidence); (output / "prompt_snapshot.txt").write_text(SYSTEM_PROMPT + "\n\n" + QUESTIONS + "\n\n[SUPPLIED EVIDENCE PACKET]\n" + evidence_text, encoding="utf-8")
    manifest = {"status": "running", "video_id": config["dataset_video_id"], "model": model, "prompt_version": config["prompt_version"], "input_file_hashes": source_hashes, "evidence_packet_sha256": evidence_hash, "code_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(), "api_call_count": 0, "successful_call_count": 0, "retry_count": 0, "total_input_tokens": 0, "total_output_tokens": 0, "total_api_latency_seconds": 0.0}
    write_json(output / "run_manifest.json", manifest); client = anthropic.Anthropic(api_key=api_key); raw_path = output / "raw_response.txt"; structured_path = output / "structured_answer.json"; started = time.perf_counter(); run_started_at = datetime.now(timezone.utc).isoformat()
    try:
        raw, record = call_haiku(client, model=model, system=SYSTEM_PROMPT, prompt=prompt, purpose="initial_answer", retry_index=0, api_log=api_log, video_id=config["dataset_video_id"], prompt_hash=prompt_hash, evidence_hash=evidence_hash, raw_path=raw_path, structured_path=structured_path, max_tokens=int(config["max_tokens"]))
        raw_path.write_text(raw, encoding="utf-8")
        retry = 0
        # Structured Outputs should make ordinary JSON repair unnecessary. A
        # single complete retry is reserved exclusively for truncation.
        if record.get("stop_reason") == "max_tokens" and config.get("allow_retry", True):
            raw_retry, retry_record = call_haiku(client, model=model, system=SYSTEM_PROMPT, prompt=prompt, purpose="initial_answer_retry", retry_index=1, api_log=api_log, video_id=config["dataset_video_id"], prompt_hash=prompt_hash, evidence_hash=evidence_hash, raw_path=output / "raw_response_retry.txt", structured_path=structured_path, max_tokens=9000)
            (output / "raw_response_retry.txt").write_text(raw_retry, encoding="utf-8")
            raw, record = raw_retry, retry_record
            retry = 1
            if record.get("stop_reason") == "max_tokens":
                raise RuntimeError("structured output truncated at max_tokens after one 9000-token retry")
        answer = parse_json(raw)
        validation_errors = validate_answer(answer, config["dataset_video_id"]); write_json(structured_path, answer); validation = {"valid": not validation_errors, "errors": validation_errors, "schema_version": "provisional_v0.1"}; write_json(output / "validation_report.json", validation)
        records = api_records_for_run(api_log, config["dataset_video_id"], run_started_at); manifest.update({"status": "completed" if not validation_errors else "completed_with_validation_errors", "retry_count": retry, "api_call_count": len(records), "successful_call_count": sum(bool(x.get("success")) for x in records), "total_input_tokens": sum(x.get("input_tokens") or 0 for x in records), "total_output_tokens": sum(x.get("output_tokens") or 0 for x in records), "total_api_latency_seconds": sum(x.get("latency_seconds") or 0 for x in records), "validation": validation, "total_elapsed_seconds": time.perf_counter() - started}); write_json(output / "run_manifest.json", manifest)
        (output / "REPORT.md").write_text(f"# EgoPolice AV Answer V0\n\n- Video: `{config['dataset_video_id']}`\n- Model: `{model}`\n- JSON valid: **{validation['valid']}**\n- API calls: {manifest['api_call_count']} (repair retries: {manifest['retry_count']})\n- API latency: {manifest['total_api_latency_seconds']:.2f}s\n- Input/output tokens: {manifest['total_input_tokens']} / {manifest['total_output_tokens']}\n\nThis V0 uses only visual Medium captions, visual storyline, timestamped ASR, and audio–Medium alignment. No raw video was provided to Claude.\n", encoding="utf-8")
        print(json.dumps({"status": manifest["status"], "output": str(output), "api_calls": manifest["api_call_count"], "validation_valid": validation["valid"], "model": model}, ensure_ascii=False, indent=2)); return 0 if validation["valid"] else 1
    except Exception as exc:
        records = api_records_for_run(api_log, config["dataset_video_id"], run_started_at)
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error_message": str(exc)[:1000], "api_call_count": len(records), "successful_call_count": sum(bool(x.get("success")) for x in records), "retry_count": sum(1 for x in records if x.get("purpose") == "initial_answer_retry"), "total_input_tokens": sum(x.get("input_tokens") or 0 for x in records), "total_output_tokens": sum(x.get("output_tokens") or 0 for x in records), "total_api_latency_seconds": sum(x.get("latency_seconds") or 0 for x in records), "total_elapsed_seconds": time.perf_counter() - started}); write_json(output / "run_manifest.json", manifest); raise


if __name__ == "__main__":
    raise SystemExit(main())
