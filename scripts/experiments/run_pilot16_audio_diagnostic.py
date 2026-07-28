"""Pilot16 audio-only diagnostic, migrated from main:c21c5cb audio baseline."""

from __future__ import annotations

import json
import os
import platform
import resource
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.index import decode_wav, interval_overlap_ratio, merge_intervals, write_json  # noqa: E402

CONFIG = ROOT / "config/experiments/pilot16_audio_diagnostic_v0_1.json"


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", value.replace("/", "__")).strip("._-")


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def mps_memory() -> dict[str, int | None]:
    if not hasattr(torch, "mps"):
        return {"current_allocated_bytes": None, "driver_allocated_bytes": None}
    current = getattr(torch.mps, "current_allocated_memory", None)
    driver = getattr(torch.mps, "driver_allocated_memory", None)
    return {"current_allocated_bytes": int(current()) if current else None, "driver_allocated_bytes": int(driver()) if driver else None}


def stage(t: float) -> dict[str, Any]:
    return {"elapsed_sec": time.perf_counter() - t, "peak_rss_bytes": rss_bytes(), "mps_memory": mps_memory()}


def runtime_preflight(config: dict[str, Any]) -> dict[str, Any]:
    mount = Path(config["required_mount"]); runtime = Path(config["runtime_root"])
    if not mount.is_dir():
        raise RuntimeError(f"External mount unavailable: {mount}")
    if not runtime.is_relative_to(mount):
        raise RuntimeError(f"Audio runtime must remain under external mount: {runtime}")
    runtime.mkdir(parents=True, exist_ok=True)
    stat = os.statvfs(runtime); free = stat.f_bavail * stat.f_frsize
    if free < 20 * 2**30:
        raise RuntimeError(f"External volume has less than 20 GiB free: {free / 2**30:.2f} GiB")
    probe = runtime / ".audio_diagnostic_write_probe"
    probe.write_text("ok", encoding="utf-8"); probe.unlink()
    return {"mount": mount.as_posix(), "runtime_root": runtime.as_posix(), "free_gib": free / 2**30, "fallback": False}


def probe_video(video: Path, ffprobe: Path) -> dict[str, Any]:
    command = [str(ffprobe), "-v", "error", "-show_entries", "stream=index,codec_type,codec_name:format=duration", "-of", "json", str(video)]
    payload = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    return {"duration_sec": float(payload["format"]["duration"]), "streams": payload.get("streams", []), "audio_present": any(x.get("codec_type") == "audio" for x in payload.get("streams", []))}


def extract_audio(video: Path, wav: Path, ffmpeg: Path) -> dict[str, Any]:
    command = [str(ffmpeg), "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return {"audio_path": wav.as_posix(), "bytes": wav.stat().st_size}


def vad_regions(model: Any, waveform: np.ndarray, sr: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from silero_vad import get_speech_timestamps
    raw = get_speech_timestamps(torch.from_numpy(waveform), model, threshold=float(cfg["vad_threshold"]), sampling_rate=sr,
                                min_speech_duration_ms=int(cfg["minimum_speech_duration_ms"]), speech_pad_ms=int(cfg["speech_pad_ms"]), return_seconds=False)
    intervals = [{"start_time": float(x["start"]) / sr, "end_time": min(len(waveform) / sr, float(x["end"]) / sr), "source_vad_segments": [{"start_sample": int(x["start"]), "end_sample": int(x["end"])}]} for x in raw]
    merged = merge_intervals(intervals, float(cfg["speech_merge_gap_sec"]))
    regions = []
    for index, item in enumerate(merged):
        regions.append({"speech_region_id": f"speech_{index:04d}", "start_time": round(item["start_time"], 3), "end_time": round(item["end_time"], 3), "duration": round(item["end_time"] - item["start_time"], 3), "transcript": "", "asr_language": None, "whisper_segments": [], "source_vad_segments": item.get("source_vad_segments", []), "warnings": []})
    return regions


def transcribe_regions(model: Any, waveform: np.ndarray, sr: int, regions: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    records = []; next_id = 0; fp16 = cfg["whisper_precision"] == "float16" and cfg["whisper_device"] == "cuda"
    for region in regions:
        start_sample = max(0, int(round(region["start_time"] * sr))); end_sample = min(len(waveform), int(round(region["end_time"] * sr)))
        clip = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
        if clip.size < sr // 10:
            region["warnings"].append("speech_region_too_short_for_asr"); continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = model.transcribe(clip, fp16=fp16, word_timestamps=bool(cfg["whisper_word_timestamps"]), condition_on_previous_text=False, verbose=None)
            region["warnings"].extend(str(item.message) for item in caught)
            region["asr_language"] = result.get("language"); region["transcript"] = result.get("text", "").strip()
            for item in result.get("segments", []):
                language = result.get("language"); abs_start = region["start_time"] + float(item["start"]); abs_end = min(region["end_time"], region["start_time"] + float(item["end"]))
                words = [{"word": word.get("word"), "start_time": round(region["start_time"] + float(word.get("start", 0.0)), 3), "end_time": round(region["start_time"] + float(word.get("end", 0.0)), 3), "probability": word.get("probability")} for word in item.get("words", []) or []]
                record = {"transcript_segment_id": f"transcript_{next_id:04d}", "speech_region_id": region["speech_region_id"], "start_time": round(abs_start, 3), "end_time": round(abs_end, 3), "text": item.get("text", "").strip(), "language": language, "whisper_metadata": {"whisper_segment_id": item.get("id"), "temperature": item.get("temperature"), "avg_logprob": item.get("avg_logprob"), "compression_ratio": item.get("compression_ratio"), "no_speech_prob": item.get("no_speech_prob"), "words": words}, "warnings": []}
                records.append(record); region["whisper_segments"].append(record["transcript_segment_id"]); next_id += 1
        except Exception as exc:
            region["warnings"].append(f"asr_error: {type(exc).__name__}: {exc}")
    return records


def overlap_duration(start: float, end: float, regions: list[dict[str, Any]]) -> float:
    return sum(max(0.0, min(end, x["end_time"]) - max(start, x["start_time"])) for x in regions)


def main() -> int:
    config_path = Path(os.environ.get("PILOT16_AUDIO_CONFIG", str(CONFIG)))
    config = json.loads(config_path.read_text(encoding="utf-8")); output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True); video = Path(config["video_path"]); video_id = config["dataset_video_id"]; safe = safe_id(video_id)
    audio_root = Path(config["runtime_root"]) / safe / config["audio_root_name"]
    audio_root.mkdir(parents=True, exist_ok=True); preflight = runtime_preflight(config)
    manifest: dict[str, Any] = {"status": "running", "experiment_id": config["experiment_id"], "dataset_video_id": video_id, "video_path": video.as_posix(), "runtime_audio_root": audio_root.as_posix(), "python": {"executable": sys.executable, "version": sys.version.split()[0], "machine": platform.machine()}, "torch": {"version": torch.__version__, "mps_available": bool(torch.backends.mps.is_available())}, "preflight": preflight, "stages": {}}
    write_json(output / "audio_run_manifest.json", manifest); current = "preflight"; total = time.perf_counter()
    try:
        current = "ffprobe"; t = time.perf_counter(); media = probe_video(video, Path(config["ffprobe"])); manifest["media"] = media; manifest["stages"][current] = stage(t); write_json(output / "audio_run_manifest.json", manifest)
        if not media["audio_present"]: raise RuntimeError("Input video has no audio stream")
        current = "audio_extraction"; t = time.perf_counter(); audio_path = audio_root / "audio_16khz_mono.wav"; extraction = extract_audio(video, audio_path, Path(config["ffmpeg"])); manifest["stages"][current] = {**stage(t), **extraction}; write_json(output / "audio_run_manifest.json", manifest)
        current = "decode"; t = time.perf_counter(); waveform, audio_meta = decode_wav(audio_path, int(config["sample_rate"])); sr = int(config["sample_rate"]); manifest["stages"][current] = {**stage(t), **audio_meta}; write_json(output / "audio_run_manifest.json", manifest)
        current = "vad"; t = time.perf_counter(); import silero_vad; torch.hub.set_dir(str(audio_root / "torch_cache")); vad_model = silero_vad.load_silero_vad(onnx=False); speech = vad_regions(vad_model, waveform, sr, config); manifest["stages"][current] = {**stage(t), "vad_segment_count": len(speech), "device": "cpu"}; write_json(output / "speech_segments.json", {"video_id": video_id, "audio_duration_sec": len(waveform) / sr, "vad": speech}); write_json(output / "audio_run_manifest.json", manifest)
        current = "timestamped_asr"; t = time.perf_counter(); import whisper; requested_device = str(config["whisper_device"]); device = "mps" if requested_device == "mps" and torch.backends.mps.is_available() else "cpu"; manifest["audio_device"] = {"requested": requested_device, "actual": device, "fallback": requested_device != device, "reason": "MPS unavailable on current macOS" if requested_device != device else None}; whisper_cache = Path(config.get("whisper_cache_dir", audio_root / "whisper_cache")); model = whisper.load_model(config["whisper_model"], device=device, download_root=str(whisper_cache)); transcripts = transcribe_regions(model, waveform, sr, speech, {**config, "whisper_device": device}); success = sum(bool(x["transcript"].strip()) for x in speech); manifest["stages"][current] = {**stage(t), "device": device, "vad_region_count": len(speech), "asr_success_count": success, "asr_success_rate": success / max(len(speech), 1)}; write_json(output / "speech_segments.json", {"video_id": video_id, "audio_duration_sec": len(waveform) / sr, "vad": speech, "transcript_segments": transcripts}); write_json(output / "audio_run_manifest.json", manifest)
        current = "speech_timeline_alignment"; t = time.perf_counter(); visual = json.loads(Path(config["visual_medium_events"]).read_text(encoding="utf-8")); medium_nodes = visual.get("nodes", []); speech_duration = sum(max(0.0, x["end_time"] - x["start_time"]) for x in speech); alignments = []
        for medium in medium_nodes:
            start, end = float(medium["start"]), float(medium["end"]); overlaps = [x for x in speech if x["end_time"] > start and x["start_time"] < end]; matching = [x for x in transcripts if x["end_time"] > start and x["start_time"] < end]; unique_intervals = merge_intervals([{ "start_time": max(start, x["start_time"]), "end_time": min(end, x["end_time"]) } for x in overlaps], 0.0); duration = sum(x["end_time"] - x["start_time"] for x in unique_intervals); alignments.append({"medium_id": medium["node_id"], "start_time": start, "end_time": end, "speech_duration_sec": round(duration, 3), "speech_overlap_ratio": round(duration / max(end - start, 1e-9), 4), "transcript": " ".join(x["text"] for x in matching if x["text"]), "transcript_segment_ids": [x["transcript_segment_id"] for x in matching], "has_speech": bool(matching or duration > 0), "visual_event_without_speech": not bool(matching or duration > 0)}); timeline = {"video_id": video_id, "audio_duration_sec": len(waveform) / sr, "speech_duration_sec": round(speech_duration, 3), "speech_ratio": round(speech_duration / max(len(waveform) / sr, 1e-9), 4), "vad_segment_count": len(speech), "asr_success_count": success, "asr_success_rate": success / max(len(speech), 1), "medium_count": len(alignments), "medium_without_speech_count": sum(x["visual_event_without_speech"] for x in alignments), "alignments": alignments}; manifest["stages"][current] = stage(t); write_json(output / "speech_timeline.json", timeline); write_json(output / "audio_medium_alignment.json", {"video_id": video_id, "visual_events_source": config["visual_medium_events"], "medium_alignments": alignments}); write_json(output / "audio_run_manifest.json", manifest)
        manifest["status"] = "completed"; manifest["total_elapsed_sec"] = time.perf_counter() - total; manifest["peak_rss_bytes"] = rss_bytes(); manifest["mps_memory_at_end"] = mps_memory(); write_json(output / "audio_run_manifest.json", manifest)
        report = ["# EgoPolice audio diagnostic", "", f"- Video: `{video_id}`", f"- Duration: {len(waveform) / sr:.3f}s", f"- Speech ratio: {timeline['speech_ratio']:.2%}", f"- VAD segments: {len(speech)}", f"- ASR success: {success}/{len(speech)} ({timeline['asr_success_rate']:.2%})", f"- Medium events without speech: {timeline['medium_without_speech_count']}/{len(alignments)}", f"- Total elapsed: {manifest['total_elapsed_sec']:.2f}s", f"- Peak RSS: {manifest['peak_rss_bytes'] / 2**30:.2f} GiB", f"- Whisper device: {device}", "", "This is an independent audio diagnostic. It does not modify visual segmentation, captions, or storyline."]
        (output / "audio_diagnostic_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        cards = "".join(f'<tr><td>{x["medium_id"]}</td><td>{x["start_time"]:.1f}–{x["end_time"]:.1f}s</td><td>{x["speech_duration_sec"]:.2f}s</td><td>{"yes" if x["visual_event_without_speech"] else "no"}</td><td>{x["transcript"]}</td></tr>' for x in alignments)
        (output / "review.html").write_text(f'<!doctype html><meta charset="utf-8"><title>EgoPolice audio diagnostic</title><style>body{{font:14px system-ui;max-width:1400px;margin:2rem;color:#183153}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5dd;padding:7px;text-align:left;vertical-align:top}}th{{background:#e6edf3}}pre{{white-space:pre-wrap}}</style><h1>EgoPolice audio diagnostic</h1><p>Independent audio branch: video → audio extraction → Silero VAD → timestamped Whisper ASR → Medium alignment.</p><h2>Summary</h2><pre>{json.dumps(timeline,ensure_ascii=False,indent=2)}</pre><h2>Medium alignment</h2><table><tr><th>Medium</th><th>Time</th><th>Speech duration</th><th>Visual event without speech</th><th>Transcript</th></tr>{cards}</table>', encoding="utf-8")
        print(json.dumps({"status": "completed", "output": str(output), "audio_root": str(audio_root), "speech_ratio": timeline["speech_ratio"], "vad_segments": len(speech), "asr_success_rate": timeline["asr_success_rate"], "medium_without_speech": timeline["medium_without_speech_count"]}, ensure_ascii=False, indent=2)); return 0
    except Exception as exc:
        manifest.update({"status": "failed", "failed_stage": current, "error": f"{type(exc).__name__}: {exc}", "total_elapsed_sec": time.perf_counter() - total, "peak_rss_bytes": rss_bytes(), "mps_memory_at_failure": mps_memory()}); write_json(output / "audio_run_manifest.json", manifest); raise


if __name__ == "__main__":
    raise SystemExit(main())
