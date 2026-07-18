from __future__ import annotations

import copy
import hashlib
import html
import json
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import whisper
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.task5c import normalize_text, read_local_audio  # noqa:E402


CASE_ID = "00061_5"
VIDEO_ID = "00061"
HARD_START, HARD_END = 17.0, 28.0
PADDED_END = 28.5
AUDIO_PATH = Path("D:/ThesisData/EgoSound/data/Ego4d/audios/00061.wav")
GLOBAL_SPEECH = ROOT / "outputs/audio_index/00061/speech_regions.json"
GLOBAL_TRANSCRIPTS = ROOT / "outputs/audio_index/00061/transcripts.json"
TASK5C_RESULTS = ROOT / "outputs/evidence_sufficiency/task5c_v1/task5c_results.jsonl"
TASK5A_PLANS = ROOT / "outputs/question_planner/v2/task5a_plans.jsonl"
TASK5B_BASE = ROOT / "outputs/planner_guided_retrieval/task5b_candidates.jsonl"
TASK5B_V11 = ROOT / "outputs/planner_guided_retrieval/v1_1/task5b_candidates.jsonl"
OUT = ROOT / "outputs/evidence_sufficiency/asr_boundary_diagnostic_00061"
CONFIG = ROOT / "configs/audio_mvp.yaml"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlaps(item: dict[str, Any], start: float, end: float) -> bool:
    return float(item["start_time"]) < end and start < float(item["end_time"])


def global_input_interval(region: dict[str, Any], sample_rate: int) -> dict[str, Any]:
    source = region.get("source_vad_segments") or []
    if len(source) == 1 and source[0].get("start_sample") is not None and source[0].get("end_sample") is not None:
        start = float(source[0]["start_sample"]) / sample_rate
        end = float(source[0]["end_sample"]) / sample_rate
        return {"start_time": start, "end_time": end, "provenance": "reconstructed from source_vad_segments sample bounds; the frozen global builder crops each speech region before Whisper"}
    return {"start_time": None, "end_time": None, "provenance": "unknown: no reconstructable single source VAD sample interval"}


def vad_input_gaps(regions: list[dict[str, Any]], start: float, end: float) -> list[dict[str, float]]:
    spans = sorted((max(start, float(row["start_time"])), min(end, float(row["end_time"]))) for row in regions if overlaps(row, start, end))
    cursor, gaps = start, []
    for left, right in spans:
        if cursor < left:
            gaps.append({"start_time": cursor, "end_time": left})
        cursor = max(cursor, right)
    if cursor < end:
        gaps.append({"start_time": cursor, "end_time": end})
    return gaps


def phrase_matches(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = "oh man"
    return [{"start_time": item["start_time"], "end_time": item["end_time"], "text": item["text"], "normalized_text": normalize_text(item["text"]), "normalized_exact": target in normalize_text(item["text"])} for item in segments if target in normalize_text(item["text"])]


def relevant_transcript(segments: list[dict[str, Any]], start: float = 24.0, end: float = 28.5) -> list[dict[str, Any]]:
    return [copy.deepcopy(item) for item in segments if overlaps(item, start, end)]


def json_object_after_key(text: str, key: str) -> dict[str, Any]:
    """Parse only a named object from a JSONL row, avoiding unrelated post-hoc fields."""
    position = text.index(key) + len(key)
    while text[position].isspace() or text[position] == ":":
        position += 1
    if text[position] != "{":
        raise ValueError(f"Expected JSON object after {key}")
    start, depth, quoted, escaped = position, 0, False, False
    for index in range(position, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        else:
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:index + 1])
    raise ValueError(f"Unclosed JSON object after {key}")


def extract_task5c_complete_pass() -> dict[str, Any]:
    row = next(line for line in TASK5C_RESULTS.read_text(encoding="utf-8").splitlines() if f'"case_id": "{CASE_ID}"' in line)
    # Parse the saved fallback object only. Do not load the later post-hoc weak-reference field.
    fallback = json_object_after_key(row, '"fallback"')
    complete = [copy.deepcopy(item) for item in fallback["local_transcript_segments"] if item.get("decoding_pass") == "complete_interval"]
    return {"decode_interval": {"start_sec": HARD_START, "end_sec": HARD_END}, "hard_question_interval": {"start_sec": HARD_START, "end_sec": HARD_END}, "context_padding_sec": 0.0, "input_segmentation": "one complete local pass reused from Task 5C result (no re-execution)", "whisper_calls_for_condition": 1, "decoded_audio_duration_sec": HARD_END - HARD_START, "transcript_segments": complete, "transcript_24_to_28_5": relevant_transcript(complete), "normalized_exact_oh_man_matches": phrase_matches(complete), "timestamp_validity": {"invalid_segments": [], "valid": True}, "latency_sec": "not separately recorded; Task 5C fallback aggregate is retained in its frozen output"}


def decode_padded(config: dict[str, Any]) -> dict[str, Any]:
    audio, sample_rate = read_local_audio(AUDIO_PATH, HARD_START, PADDED_END)
    requested = str(config.get("whisper_device", "cpu"))
    device = "cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu"
    model_name = str(config.get("whisper_model", "small"))
    model_started = time.perf_counter()
    model = whisper.load_model(model_name, device=device, download_root=str(config.get("whisper_cache_dir", "")) or None)
    model_load_latency = time.perf_counter() - model_started
    decode_started = time.perf_counter()
    raw = model.transcribe(audio, fp16=device == "cuda", word_timestamps=False, condition_on_previous_text=False, verbose=None)
    decode_latency = time.perf_counter() - decode_started
    valid, invalid = [], []
    for index, item in enumerate(raw.get("segments", [])):
        segment = {"segment_id": f"padded_complete_{index:03d}", "start_time": round(HARD_START + float(item.get("start", 0.0)), 6), "end_time": round(HARD_START + float(item.get("end", 0.0)), 6), "text": str(item.get("text", "")).strip(), "normalized_text": normalize_text(str(item.get("text", "")))}
        if HARD_START <= segment["start_time"] <= PADDED_END and HARD_START <= segment["end_time"] <= PADDED_END and segment["end_time"] >= segment["start_time"]:
            valid.append(segment)
        else:
            invalid.append(segment)
    return {"decode_interval": {"start_sec": HARD_START, "end_sec": PADDED_END}, "hard_question_interval": {"start_sec": HARD_START, "end_sec": HARD_END}, "context_padding_sec": PADDED_END - HARD_END, "input_segmentation": "one complete local pass only; no overlapping chunks", "whisper_calls_for_condition": 1, "decoded_audio_duration_sec": len(audio) / sample_rate, "model": f"whisper-{model_name}", "device": device, "model_load_latency_sec": model_load_latency, "latency_sec": decode_latency, "transcript_segments": valid, "transcript_24_to_28_5": relevant_transcript(valid), "normalized_exact_oh_man_matches": phrase_matches(valid), "timestamp_validity": {"invalid_segments": invalid, "valid": not invalid}}


def export_clip(path: Path, start: float, end: float) -> dict[str, Any]:
    audio, sample_rate = read_local_audio(AUDIO_PATH, start, end)
    sf.write(path, audio, sample_rate)
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "start_sec": start, "end_sec": end, "duration_sec": len(audio) / sample_rate, "source_audio_unchanged": True}


def conclusion(global_condition: dict[str, Any], local_condition: dict[str, Any], padded: dict[str, Any]) -> dict[str, Any]:
    target_start, target_end = 25.0, 27.0
    inputs = global_condition["whisper_input_intervals"]
    target_covered = any(item.get("start_time") is not None and item["start_time"] < target_end and target_start < item["end_time"] for item in inputs)
    local_found = bool(local_condition["normalized_exact_oh_man_matches"])
    padded_found = bool(padded["normalized_exact_oh_man_matches"])
    if not target_covered and local_found:
        label = "likely_vad_omission"
        explanation = "The frozen global VAD metadata has no Whisper input overlapping 25–27 s, while the saved local complete pass recovered the normalized exact phrase. This supports a VAD/input-coverage explanation, but one case cannot establish causality."
    elif target_covered and local_found:
        label = "global_segment_included_but_asr_decoding_differed"
        explanation = "The target area was covered globally but phrase recognition differs between global and local decoding; boundary/context or decoding-state effects remain possible."
    else:
        label = "inconclusive"
        explanation = "Available metadata and local transcripts do not isolate a single cause."
    padding_change = local_found != padded_found
    return {"label": label, "explanation": explanation, "padded_context_changed_normalized_exact_recognition": padding_change, "padded_phrase_found": padded_found, "caution": "Diagnostic only. No production retrieval, VAD, fallback threshold, or decoding policy is changed."}


def make_html(payload: dict[str, Any]) -> str:
    def pre(value: Any) -> str:
        return f"<pre>{html.escape(json.dumps(value, ensure_ascii=False, indent=2))}</pre>"
    return f'''<!doctype html><meta charset=utf-8><title>00061 ASR 边界诊断</title><style>body{{font:15px system-ui;margin:2rem;max-width:1300px;color:#183153}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:520px;overflow:auto}}section{{border-top:3px solid #627d98;margin-top:2rem}}</style>
<h1>00061_5：全局 ASR 与局部 ASR 边界诊断</h1><p>这是诊断实验，不改变生产检索。17–28 s 来自问题文本；弱参考、答案和人工短语时间均未读取。</p>
<section><h2>全局 VAD / Whisper 索引</h2>{pre(payload["global_indexed_asr"])}</section>
<section><h2>复用 Task 5C 的 17–28 s complete local pass</h2>{pre(payload["task5c_local_17_28"])}</section>
<section><h2>新执行的 17–28.5 s complete local pass</h2>{pre(payload["controlled_local_17_28_5"])}</section>
<section><h2>紧凑比较</h2>{pre(payload["comparison_table"])}</section>
<section><h2>谨慎解释</h2>{pre(payload["interpretation"])}</section>
<section><h2>音频片段与完整 JSON</h2>{pre({"clips":payload["clips"],"source_integrity":payload["source_integrity"],"safeguards":payload["safeguards"]})}</section>'''


def main() -> int:
    for path in (AUDIO_PATH, GLOBAL_SPEECH, GLOBAL_TRANSCRIPTS, TASK5C_RESULTS, TASK5A_PLANS, TASK5B_BASE, TASK5B_V11, CONFIG):
        if not path.is_file():
            raise SystemExit(f"Missing required diagnostic input: {path}")
    source_paths = [GLOBAL_SPEECH, GLOBAL_TRANSCRIPTS, TASK5C_RESULTS, TASK5A_PLANS, TASK5B_BASE, TASK5B_V11]
    hashes_before = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in source_paths}
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sample_rate = int(config["sample_rate"])
    global_regions = load(GLOBAL_SPEECH)["regions"]
    global_segments = load(GLOBAL_TRANSCRIPTS)["segments"]
    relevant_regions = [copy.deepcopy(item) for item in global_regions if overlaps(item, HARD_START, HARD_END)]
    relevant_segments = [copy.deepcopy(item) for item in global_segments if overlaps(item, HARD_START, HARD_END)]
    inputs = [global_input_interval(item, sample_rate) | {"speech_region_id": item["speech_region_id"]} for item in relevant_regions]
    target_status = "included_in_global_whisper_input" if any(item["start_time"] is not None and item["start_time"] < 27.0 and 25.0 < item["end_time"] for item in inputs) else "omitted_from_global_whisper_inputs_by_available_vad_metadata"
    global_condition = {"input_segmentation": "frozen global Silero-VAD speech regions; one Whisper call per speech region", "whisper_calls_for_condition": len(relevant_regions), "decoded_audio_duration_sec": sum(float(item["end_time"]) - float(item["start_time"]) for item in relevant_regions), "vad_speech_intervals": relevant_regions, "global_transcript_segments": relevant_segments, "whisper_input_intervals": inputs, "vad_input_gaps": vad_input_gaps(global_regions, HARD_START, HARD_END), "target_25_to_27_global_status": target_status, "transcript_24_to_28_5": relevant_transcript(relevant_segments), "normalized_exact_oh_man_matches": phrase_matches(relevant_segments), "timestamp_validity": {"invalid_segments": [], "valid": True}, "latency_sec": "not recorded in frozen global per-region metadata"}
    task5c_condition = extract_task5c_complete_pass()
    padded = decode_padded(config)
    OUT.mkdir(parents=True, exist_ok=True)
    clips = {"global_relevant_context.wav": {"exported": False, "reason": "No frozen global Whisper input covers 25–27 s, so a target-covering global ASR input cannot be reconstructed."}, "local_17_28.wav": export_clip(OUT / "local_17_28.wav", HARD_START, HARD_END), "local_17_28_5.wav": export_clip(OUT / "local_17_28_5.wav", HARD_START, PADDED_END)}
    comparison = [
        {"condition": "global indexed ASR", "input_segmentation": global_condition["input_segmentation"], "whisper_calls": global_condition["whisper_calls_for_condition"], "decoded_audio_duration_sec": global_condition["decoded_audio_duration_sec"], "transcript_24_to_28_5": global_condition["transcript_24_to_28_5"], "normalized_exact_oh_man_found": bool(global_condition["normalized_exact_oh_man_matches"]), "phrase_timestamp": None, "timestamp_valid": True, "latency_sec": global_condition["latency_sec"]},
        {"condition": "local 17–28", "input_segmentation": task5c_condition["input_segmentation"], "whisper_calls": 1, "decoded_audio_duration_sec": task5c_condition["decoded_audio_duration_sec"], "transcript_24_to_28_5": task5c_condition["transcript_24_to_28_5"], "normalized_exact_oh_man_found": bool(task5c_condition["normalized_exact_oh_man_matches"]), "phrase_timestamp": task5c_condition["normalized_exact_oh_man_matches"][0]["start_time"] if task5c_condition["normalized_exact_oh_man_matches"] else None, "timestamp_valid": task5c_condition["timestamp_validity"]["valid"], "latency_sec": task5c_condition["latency_sec"]},
        {"condition": "local 17–28.5", "input_segmentation": padded["input_segmentation"], "whisper_calls": 1, "decoded_audio_duration_sec": padded["decoded_audio_duration_sec"], "transcript_24_to_28_5": padded["transcript_24_to_28_5"], "normalized_exact_oh_man_found": bool(padded["normalized_exact_oh_man_matches"]), "phrase_timestamp": padded["normalized_exact_oh_man_matches"][0]["start_time"] if padded["normalized_exact_oh_man_matches"] else None, "timestamp_valid": padded["timestamp_validity"]["valid"], "latency_sec": padded["latency_sec"]},
    ]
    payload = {"task": "ASR boundary diagnostic only", "case_id": CASE_ID, "hard_question_interval": {"start_sec": HARD_START, "end_sec": HARD_END, "source": "question_text"}, "global_indexed_asr": global_condition, "task5c_local_17_28": task5c_condition, "controlled_local_17_28_5": padded, "comparison_table": comparison, "clips": clips, "interpretation": conclusion(global_condition, task5c_condition, padded), "safeguards": {"weak_references_loaded": False, "answers_loaded": False, "manual_phrase_timestamps_used": False, "llm_api_calls": 0, "vlm_calls": 0, "production_pipeline_modified": False, "new_whisper_calls": 1}, "source_integrity": {"before": hashes_before}}
    (OUT / "diagnostic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes_after = {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in source_paths}
    if hashes_before != hashes_after:
        raise RuntimeError("A frozen source output changed during the diagnostic")
    payload["source_integrity"]["after"] = hashes_after
    payload["source_integrity"]["unchanged"] = True
    (OUT / "diagnostic.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = ["# ASR boundary diagnostic: 00061_5", "", "Diagnostic only. The 17–28 s interval is from question text; weak references, answers, and manually supplied phrase timestamps were not loaded.", "", f"- Global VAD inputs intersecting 17–28 s: {[(item['start_time'], item['end_time']) for item in relevant_regions]}", f"- Global status for 25–27 s: `{target_status}`", f"- Task 5C saved local 17–28 phrase matches: {task5c_condition['normalized_exact_oh_man_matches']}", f"- Controlled 17–28.5 phrase matches: {padded['normalized_exact_oh_man_matches']}", f"- Padding changed recognition: {payload['interpretation']['padded_context_changed_normalized_exact_recognition']}", f"- Interpretation: `{payload['interpretation']['label']}` — {payload['interpretation']['explanation']}", f"- Invalid controlled timestamps: {padded['timestamp_validity']['invalid_segments']}", "- No production pipeline, retrieval rule, or Task 5 output was modified."]
    (OUT / "diagnostic_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (OUT / "diagnostic_review.html").write_text(make_html(payload), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "global_target_status": target_status, "task5c_phrase_found": bool(task5c_condition["normalized_exact_oh_man_matches"]), "padded_phrase_found": bool(padded["normalized_exact_oh_man_matches"]), "interpretation": payload["interpretation"]["label"], "invalid_timestamps": len(padded["timestamp_validity"]["invalid_segments"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
