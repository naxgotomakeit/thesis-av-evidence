from __future__ import annotations

import argparse
import json
import sys
import time
import warnings as python_warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.index import (  # noqa: E402
    REFERENCE_LABEL,
    build_acoustic_regions,
    compute_acoustic_features,
    decode_wav,
    merge_intervals,
    propose_acoustic_boundaries,
    write_json,
)


def resolve_audio_path(case: dict[str, Any], dataset_root: Path) -> Path:
    path = Path(case["audio_path"])
    return path if path.is_absolute() else dataset_root / path


def vad_probabilities(model: Any, waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    chunk = 512 if sample_rate == 16000 else 256
    usable = len(waveform) - len(waveform) % chunk
    model.reset_states()
    probabilities = []
    with torch.inference_mode():
        for start in range(0, usable, chunk):
            probabilities.append(float(model(torch.from_numpy(waveform[start:start + chunk]), sample_rate).item()))
    times = (np.arange(len(probabilities)) * chunk + chunk / 2) / sample_rate
    return times, np.asarray(probabilities, dtype=np.float32)


def speech_regions_from_vad(model: Any, waveform: np.ndarray, sample_rate: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from silero_vad import get_speech_timestamps

    raw = get_speech_timestamps(
        torch.from_numpy(waveform), model,
        threshold=float(cfg["vad_threshold"]), sampling_rate=sample_rate,
        min_speech_duration_ms=int(cfg["minimum_speech_duration_ms"]),
        speech_pad_ms=int(cfg["speech_pad_ms"]), return_seconds=False,
    )
    intervals = [
        {
            "start_time": float(item["start"]) / sample_rate,
            "end_time": min(len(waveform) / sample_rate, float(item["end"]) / sample_rate),
            "source_vad_segments": [{"start_sample": int(item["start"]), "end_sample": int(item["end"])}],
        }
        for item in raw
    ]
    merged = merge_intervals(intervals, float(cfg["speech_merge_gap_sec"]))
    prob_times, probabilities = vad_probabilities(model, waveform, sample_rate)
    regions = []
    for index, item in enumerate(merged):
        mask = (prob_times >= item["start_time"]) & (prob_times <= item["end_time"])
        local = probabilities[mask]
        summary = {
            "mean": float(np.mean(local)) if local.size else None,
            "minimum": float(np.min(local)) if local.size else None,
            "maximum": float(np.max(local)) if local.size else None,
            "analysis_chunk_samples": 512,
        }
        regions.append({
            "speech_region_id": f"speech_{index:04d}",
            "start_time": round(item["start_time"], 3),
            "end_time": round(item["end_time"], 3),
            "duration": round(item["end_time"] - item["start_time"], 3),
            "transcript": "",
            "asr_language": None,
            "whisper_segments": [],
            "source_vad_segments": item["source_vad_segments"],
            "vad_probability_summary": summary,
            "warnings": [],
        })
    return regions


def transcribe_regions(model: Any, waveform: np.ndarray, sample_rate: int, speech_regions: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    transcript_segments = []
    fp16 = cfg["whisper_precision"] == "float16" and cfg["whisper_device"] == "cuda"
    segment_index = 0
    for region in speech_regions:
        start_sample = max(0, int(round(region["start_time"] * sample_rate)))
        end_sample = min(len(waveform), int(round(region["end_time"] * sample_rate)))
        clip = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
        if clip.size < sample_rate // 10:
            region["warnings"].append("speech_region_too_short_for_asr")
            continue
        try:
            with python_warnings.catch_warnings(record=True) as caught:
                python_warnings.simplefilter("always")
                result = model.transcribe(
                    clip, fp16=fp16, word_timestamps=bool(cfg["whisper_word_timestamps"]),
                    condition_on_previous_text=False, verbose=None,
                )
            for warning in caught:
                message = str(warning.message)
                if message not in region["warnings"]:
                    region["warnings"].append(message)
            language = result.get("language")
            region["asr_language"] = language
            region["transcript"] = result.get("text", "").strip()
            for item in result.get("segments", []):
                absolute_start = region["start_time"] + float(item["start"])
                absolute_end = min(region["end_time"], region["start_time"] + float(item["end"]))
                words = []
                for word in item.get("words", []) or []:
                    words.append({
                        "word": word.get("word"),
                        "start_time": round(region["start_time"] + float(word.get("start", 0.0)), 3),
                        "end_time": round(region["start_time"] + float(word.get("end", 0.0)), 3),
                        "probability": word.get("probability"),
                    })
                metadata = {
                    "whisper_segment_id": item.get("id"),
                    "seek": item.get("seek"),
                    "temperature": item.get("temperature"),
                    "avg_logprob": item.get("avg_logprob"),
                    "compression_ratio": item.get("compression_ratio"),
                    "no_speech_prob": item.get("no_speech_prob"),
                    "words": words,
                }
                record = {
                    "transcript_segment_id": f"transcript_{segment_index:04d}",
                    "speech_region_id": region["speech_region_id"],
                    "start_time": round(absolute_start, 3),
                    "end_time": round(absolute_end, 3),
                    "text": item.get("text", "").strip(),
                    "language": language,
                    "whisper_metadata": metadata,
                    "warnings": [],
                }
                transcript_segments.append(record)
                region["whisper_segments"].append(record["transcript_segment_id"])
                segment_index += 1
        except Exception as exc:  # keep processing remaining regions/cases
            region["warnings"].append(f"asr_error: {type(exc).__name__}: {exc}")
    return transcript_segments


def add_reference(ax: Any, case: dict[str, Any]) -> None:
    start, end = case.get("provided_timestamp_start"), case.get("provided_timestamp_end")
    if start is not None and end is not None:
        ax.axvspan(float(start), float(end), color="#e83e8c", alpha=0.22, label=REFERENCE_LABEL)


def save_plots(out: Path, waveform: np.ndarray, sr: int, features: dict[str, np.ndarray], speech: list[dict[str, Any]], acoustic: list[dict[str, Any]], case: dict[str, Any], threshold: float) -> None:
    duration = len(waveform) / sr
    times = np.arange(len(waveform)) / sr
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.plot(times, waveform, linewidth=0.35, color="#334155")
    add_reference(ax, case); ax.set(xlim=(0, duration), xlabel="Time (s)", ylabel="Amplitude", title=f"{case['video_id']} full waveform")
    ax.legend(loc="upper right", fontsize=7); fig.tight_layout(); fig.savefig(out / "waveform.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.specgram(waveform, NFFT=1024, Fs=sr, noverlap=768, cmap="magma", scale="dB")
    add_reference(ax, case); ax.set(xlim=(0, duration), ylim=(0, 8000), xlabel="Time (s)", ylabel="Frequency (Hz)", title=f"{case['video_id']} full spectrogram")
    ax.legend(loc="upper right", fontsize=7); fig.tight_layout(); fig.savefig(out / "spectrogram.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(features["times"], features["raw_change"], alpha=.35, label="raw composite change")
    ax.plot(features["times"], features["smoothed_change"], linewidth=1.4, label="smoothed change")
    if np.isfinite(threshold): ax.axhline(threshold, linestyle="--", color="red", label="median+MAD threshold")
    for region in acoustic[1:]: ax.axvline(region["start_time"], color="#64748b", alpha=.35)
    add_reference(ax, case); ax.set(xlim=(0, duration), xlabel="Time (s)", ylabel="Non-semantic change score", title="Acoustic boundary proposal signals")
    ax.legend(loc="upper right", fontsize=7); fig.tight_layout(); fig.savefig(out / "acoustic_change_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = {False: "#9bd4a5", True: "#cbd5e1"}
    for region in acoustic:
        y = 0.15
        ax.broken_barh([(region["start_time"], region["duration"])], (y, .25), facecolors=colors[region["low_information_or_silence"]], edgecolors="white")
        label = f"{region['acoustic_region_id'].replace('acoustic_', 'A')}\n{region['start_time']:.1f}–{region['end_time']:.1f}s"
        ax.text((region["start_time"] + region["end_time"]) / 2, y+.125, label, ha="center", va="center", fontsize=5.5, rotation=90 if region["duration"] < 5 else 0)
    for region in speech:
        ax.broken_barh([(region["start_time"], region["duration"])], (.58, .25), facecolors="#3b82f6")
        label = f"{region['speech_region_id'].replace('speech_', 'S')} {region['start_time']:.1f}–{region['end_time']:.1f}s"
        ax.text((region["start_time"] + region["end_time"]) / 2, .705, label, ha="center", va="center", fontsize=5.5, color="white", rotation=90 if region["duration"] < 4 else 0)
    add_reference(ax, case)
    from matplotlib.patches import Patch
    handles = [Patch(color="#3b82f6", label="speech_regions"), Patch(color="#9bd4a5", label="acoustic_regions"), Patch(color="#cbd5e1", label="low-information / silence"), Patch(color="#e83e8c", alpha=.25, label=REFERENCE_LABEL)]
    ax.legend(handles=handles, loc="upper right", fontsize=7)
    ax.set(xlim=(0, duration), ylim=(0, 1.12), yticks=[.275, .705], yticklabels=["acoustic_regions", "speech_regions"], xlabel="Time (s)", title=f"Question-independent audio timeline — {case['video_id']}")
    fig.tight_layout(); fig.savefig(out / "audio_timeline.png", dpi=160); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/audio_mvp.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    cases = json.loads((ROOT / cfg["mvp_cases_path"]).read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset_root"])
    output_root = ROOT / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)

    requested_device = cfg["whisper_device"]
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    if device != requested_device:
        cfg["whisper_device"] = device
        cfg["whisper_precision"] = "float32"
    from silero_vad import load_silero_vad
    import whisper
    vad_model = load_silero_vad(onnx=False)
    whisper_model = whisper.load_model(cfg["whisper_model"], device=device, download_root=cfg["whisper_cache_dir"])
    run_start = time.perf_counter()
    summaries = []
    all_warnings = []
    for case in cases:
        case_start = time.perf_counter()
        video_id = case["video_id"]
        out = output_root / video_id
        out.mkdir(parents=True, exist_ok=True)
        warnings = []
        try:
            waveform, metadata = decode_wav(resolve_audio_path(case, dataset_root), int(cfg["sample_rate"]))
            speech = speech_regions_from_vad(vad_model, waveform, int(cfg["sample_rate"]), cfg)
            transcripts = transcribe_regions(whisper_model, waveform, int(cfg["sample_rate"]), speech, cfg)
            features = compute_acoustic_features(waveform, int(cfg["sample_rate"]), float(cfg["acoustic_window_length_sec"]), float(cfg["acoustic_hop_length_sec"]), float(cfg["smoothing_window_sec"]))
            boundaries, threshold = propose_acoustic_boundaries(features["times"], features["smoothed_change"], metadata["duration"], float(cfg["minimum_acoustic_region_duration_sec"]), cfg["boundary_threshold_method"], float(cfg["boundary_mad_multiplier"]))
            acoustic = build_acoustic_regions(features, boundaries, speech, float(cfg["low_information_rms_quantile"]))
            metadata.update({
                "case_ids": [case["case_id"]], "video_id": video_id,
                "construction_is_question_independent": True,
                "construction_excluded_fields": ["question", "answer", "provided_context", "provided_timestamp_start", "provided_timestamp_end"],
                "dataset_provided_qa_reference_interval": {"start": case.get("provided_timestamp_start"), "end": case.get("provided_timestamp_end"), "usage": "visualization overlay only", "label": REFERENCE_LABEL},
                "device": device, "whisper_precision": cfg["whisper_precision"], "warnings": metadata["warnings"] + warnings,
            })
            write_json(out / "speech_regions.json", {"video_id": video_id, "region_type": "speech_regions", "question_independent": True, "regions": speech})
            write_json(out / "transcripts.json", {"video_id": video_id, "question_independent": True, "segments": transcripts})
            write_json(out / "acoustic_regions.json", {"video_id": video_id, "region_type": "acoustic_regions", "semantic_events": False, "question_independent": True, "complete_timeline_coverage": True, "boundary_threshold": threshold if np.isfinite(threshold) else None, "regions": acoustic})
            write_json(out / "audio_metadata.json", metadata)
            save_plots(out, waveform, int(cfg["sample_rate"]), features, speech, acoustic, case, threshold)
            languages = [s["asr_language"] for s in speech if s["asr_language"]]
            language = Counter(languages).most_common(1)[0][0] if languages else None
            case_warnings = sorted(set(warnings + [w for s in speech for w in s["warnings"]]))
            summaries.append({"case_id": case["case_id"], "video_id": video_id, "status": "ok", "duration": metadata["duration"], "speech_region_count": len(speech), "acoustic_region_count": len(acoustic), "transcript_segment_count": len(transcripts), "transcript_language": language, "processing_time_sec": round(time.perf_counter()-case_start, 2), "warnings": case_warnings})
        except Exception as exc:
            warning = f"{case['case_id']}: {type(exc).__name__}: {exc}"
            all_warnings.append(warning)
            summaries.append({"case_id": case["case_id"], "video_id": video_id, "status": "failed", "speech_region_count": 0, "acoustic_region_count": 0, "transcript_language": None, "processing_time_sec": round(time.perf_counter()-case_start, 2), "warnings": [warning]})
    total_time = round(time.perf_counter()-run_start, 2)
    summary = {
        "task": "Task 3A question-independent audio blue-land baseline",
        "processed_audio_count": sum(item["status"] == "ok" for item in summaries),
        "requested_audio_count": len(cases), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "processing_time_sec": total_time,
        "model_paths": {"silero_vad": str(Path(__import__('silero_vad').__file__).parent / 'data' / 'silero_vad.jit'), "whisper": str(Path(cfg["whisper_cache_dir"]) / 'small.pt')},
        "question_independent": True,
        "reference_interval_usage": "visualization overlay only; not used in VAD, ASR, acoustic boundaries, or region statistics",
        "cases": summaries, "warnings": sorted(set(all_warnings + [w for item in summaries for w in item["warnings"]])),
    }
    write_json(output_root / "audio_index_summary.json", summary)
    lines = ["# Task 3A audio index summary", "", "This baseline processes each complete WAV independently of the question, answer, annotation context, and dataset-provided timestamp.", "", "Pipeline: full audio → parallel Silero VAD/Whisper speech branch and non-semantic acoustic boundary proposal branch.", "", f"- Processed: {summary['processed_audio_count']} / {len(cases)} audio files", f"- Device: `{device}`" + (f" ({summary['gpu']})" if summary['gpu'] else ""), f"- Total processing time: {total_time:.2f} s", "- Speech and acoustic regions may overlap; acoustic content inside speech is retained.", "- `acoustic_regions` are signal-change proposals, not semantic sound events.", f"- Reference overlay label: **{REFERENCE_LABEL.replace(chr(10), ' ')}**", "", "| case_id | video_id | speech regions | acoustic regions | transcript segments | language | time (s) | status |", "|---|---:|---:|---:|---:|---|---:|---|"]
    for item in summaries:
        lines.append(f"| {item['case_id']} | {item['video_id']} | {item['speech_region_count']} | {item['acoustic_region_count']} | {item.get('transcript_segment_count', 0)} | {item.get('transcript_language') or 'unknown'} | {item['processing_time_sec']} | {item['status']} |")
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in summary["warnings"]] or ["- None."]
    lines += ["", "## Scope boundary", "", "No transcript embeddings, semantic non-speech embeddings, retrieval, visual-audio linking, reranking, final QA, or agents were implemented."]
    (output_root / "audio_index_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["processed_audio_count"] == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
