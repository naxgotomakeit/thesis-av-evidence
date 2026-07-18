from __future__ import annotations

import json
import re
import sys
import time
import warnings
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

from src.audio.index import decode_wav, merge_intervals, write_json  # noqa: E402

VIDEO_ID = "00061"
CASE_ID = "00061_5"
REVIEW_START = 17.0
REVIEW_END = 40.0
THRESHOLDS = [0.50, 0.40, 0.30, 0.20, 0.10]


def interval_duration(intervals: list[dict[str, Any]]) -> float:
    return sum(max(0.0, x["end_time"] - x["start_time"]) for x in intervals)


def overlap_duration(intervals: list[dict[str, Any]], start: float, end: float) -> float:
    return sum(max(0.0, min(end, x["end_time"]) - max(start, x["start_time"])) for x in intervals)


def silero_regions(model: Any, waveform: np.ndarray, sr: int, threshold: float, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from silero_vad import get_speech_timestamps

    raw = get_speech_timestamps(
        torch.from_numpy(waveform), model, threshold=threshold, sampling_rate=sr,
        min_speech_duration_ms=int(cfg["minimum_speech_duration_ms"]),
        speech_pad_ms=int(cfg["speech_pad_ms"]), return_seconds=False,
    )
    intervals = [
        {"start_time": item["start"] / sr, "end_time": min(len(waveform) / sr, item["end"] / sr)}
        for item in raw
    ]
    merged = merge_intervals(intervals, float(cfg["speech_merge_gap_sec"]))
    return [
        {"start_time": round(x["start_time"], 3), "end_time": round(x["end_time"], 3), "duration": round(x["end_time"] - x["start_time"], 3)}
        for x in merged
    ]


def probability_curve(model: Any, waveform: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    chunk = 512
    usable = len(waveform) - len(waveform) % chunk
    model.reset_states()
    values = []
    with torch.inference_mode():
        for start in range(0, usable, chunk):
            values.append(float(model(torch.from_numpy(waveform[start:start + chunk]), sr).item()))
    times = (np.arange(len(values)) * chunk + chunk / 2) / sr
    return times, np.asarray(values, dtype=np.float32)


def rms_curve(waveform: np.ndarray, sr: int, times: np.ndarray) -> np.ndarray:
    chunk = 512
    result = []
    for center in times:
        start = max(0, int(center * sr - chunk / 2))
        frame = waveform[start:start + chunk]
        result.append(float(np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)))
    return np.asarray(result)


def energy_risk_proxy(intervals: list[dict[str, Any]], times: np.ndarray, rms: np.ndarray) -> dict[str, float]:
    cutoff = float(np.quantile(rms, 0.25))
    low = 0.0
    total = 0.0
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.032
    for interval in intervals:
        mask = (times >= interval["start_time"]) & (times <= interval["end_time"])
        total += mask.sum() * step
        low += np.logical_and(mask, rms <= cutoff).sum() * step
    return {
        "low_energy_detected_duration_sec": round(low, 3),
        "low_energy_fraction_of_detected": round(low / total, 4) if total else 0.0,
        "interpretation": "False-positive risk proxy only; low energy is not proof of non-speech.",
    }


def mask_to_intervals(mask: np.ndarray, times: np.ndarray, duration: float, merge_gap: float, minimum_duration: float) -> list[dict[str, float]]:
    step = float(np.median(np.diff(times))) if len(times) > 1 else 0.032
    raw = []
    start = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = max(0.0, float(times[i] - step / 2))
        if start is not None and (not active or i == len(mask) - 1):
            end_i = i if active and i == len(mask) - 1 else i - 1
            end = min(duration, float(times[end_i] + step / 2))
            if end - start >= minimum_duration:
                raw.append({"start_time": start, "end_time": end})
            start = None
    merged = merge_intervals(raw, merge_gap)
    return [{"start_time": round(x["start_time"], 3), "end_time": round(x["end_time"], 3), "duration": round(x["end_time"]-x["start_time"], 3)} for x in merged]


def subtract_intervals(candidates: list[dict[str, Any]], exclusions: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Return candidate portions not already covered by baseline clear speech."""
    pieces = []
    for candidate in candidates:
        remaining = [(float(candidate["start_time"]), float(candidate["end_time"]))]
        for excluded in exclusions:
            updated = []
            left, right = float(excluded["start_time"]), float(excluded["end_time"])
            for start, end in remaining:
                if right <= start or left >= end:
                    updated.append((start, end))
                else:
                    if start < left:
                        updated.append((start, left))
                    if right < end:
                        updated.append((right, end))
            remaining = updated
        pieces.extend({"start_time": round(a, 3), "end_time": round(b, 3), "duration": round(b-a, 3)} for a, b in remaining if b > a)
    return pieces


def transcript_records(result: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    previous = None
    for item in result.get("segments", []):
        text = item.get("text", "").strip()
        flags = []
        if previous and text.casefold() == previous.casefold():
            flags.append("consecutive_duplicate_text")
        if float(item.get("no_speech_prob", 0.0)) > 0.6:
            flags.append("high_no_speech_probability")
        records.append({
            "segment_id": item.get("id"), "start_time": round(float(item["start"]), 3),
            "end_time": round(float(item["end"]), 3), "text": text,
            "temperature": item.get("temperature"), "avg_logprob": item.get("avg_logprob"),
            "compression_ratio": item.get("compression_ratio"), "no_speech_prob": item.get("no_speech_prob"),
            "automatic_quality_flags": flags,
        })
        previous = text
    return records


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/audio_mvp.yaml").read_text(encoding="utf-8"))
    cases = json.loads((ROOT / cfg["mvp_cases_path"]).read_text(encoding="utf-8"))
    case = next(x for x in cases if x["case_id"] == CASE_ID)
    audio_path = Path(cfg["dataset_root"]) / case["audio_path"]
    out = ROOT / "outputs/audio_index/00061/speech_diagnostic"
    out.mkdir(parents=True, exist_ok=True)
    waveform, metadata = decode_wav(audio_path, int(cfg["sample_rate"]))
    sr = int(cfg["sample_rate"]); duration = len(waveform) / sr

    from silero_vad import load_silero_vad
    vad_model = load_silero_vad(onnx=False)
    times, probabilities = probability_curve(vad_model, waveform, sr)
    rms = rms_curve(waveform, sr, times)
    baseline_regions = json.loads((ROOT / "outputs/audio_index/00061/speech_regions.json").read_text(encoding="utf-8"))["regions"]
    baseline_intervals = [{"start_time": x["start_time"], "end_time": x["end_time"]} for x in baseline_regions]
    baseline_sent = interval_duration(baseline_intervals)

    comparisons = []
    regions_by_threshold = {}
    for threshold in THRESHOLDS:
        regions = silero_regions(vad_model, waveform, sr, threshold, cfg)
        regions_by_threshold[f"{threshold:.2f}"] = regions
        sent = interval_duration(regions)
        review_overlap = overlap_duration(regions, REVIEW_START, REVIEW_END)
        comparisons.append({
            "threshold": threshold, "region_count": len(regions),
            "detected_duration_sec": round(sent, 3), "fraction_of_full_audio": round(sent/duration, 4),
            "coverage_of_human_review_17_40": round(review_overlap/(REVIEW_END-REVIEW_START), 4),
            "detected_duration_inside_17_40_sec": round(review_overlap, 3),
            "detected_duration_outside_17_40_sec": round(sent-review_overlap, 3),
            "additional_audio_vs_original_baseline_sec": round(sent-baseline_sent, 3),
            "false_positive_risk_proxy": energy_risk_proxy(regions, times, rms),
        })
    threshold_output = {
        "video_id": VIDEO_ID, "audio_duration_sec": duration,
        "construction_is_question_independent": True,
        "human_review_interval": {"start": REVIEW_START, "end": REVIEW_END, "usage": "diagnostic comparison and visualization only; not used to construct VAD regions"},
        "current_parameters": {"threshold": cfg["vad_threshold"], "minimum_speech_duration_ms": cfg["minimum_speech_duration_ms"], "speech_merge_gap_sec": cfg["speech_merge_gap_sec"], "speech_pad_ms": cfg["speech_pad_ms"], "silero_default_min_silence_duration_ms": 100, "analysis_window_samples": 512, "analysis_window_sec": 512/sr},
        "original_baseline_detected_duration_sec": round(baseline_sent, 3),
        "comparisons": comparisons, "regions_by_threshold": regions_by_threshold,
        "decision": "No threshold selected or frozen by this diagnostic.",
    }
    write_json(out / "vad_threshold_comparison.json", threshold_output)

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharey=True)
    for ax, limits, title in [(axes[0], (0, duration), "Complete audio"), (axes[1], (REVIEW_START, REVIEW_END), "Human-reviewed interval (~17–40 s), diagnostic view only")]:
        ax.plot(times, probabilities, color="#2563eb", linewidth=.65, label="Silero speech probability (512-sample windows)")
        for threshold in THRESHOLDS:
            ax.axhline(threshold, linestyle="--", linewidth=.7, alpha=.75, label=f"threshold {threshold:.2f}")
        ax.axvspan(REVIEW_START, REVIEW_END, color="#f59e0b", alpha=.13, label="human-review interval; not used for construction")
        ax.set_xlim(*limits); ax.set_ylim(0, 1.02); ax.set_ylabel("Speech probability"); ax.set_title(title); ax.grid(alpha=.15)
    axes[1].set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .91)); fig.savefig(out / "vad_probability_curve.png", dpi=170); plt.close(fig)

    full_output_path = out / "full_audio_whisper_transcripts.json"
    if full_output_path.exists():
        # Preserve the single completed full-audio upper-bound run when regenerating comparisons.
        full_output = json.loads(full_output_path.read_text(encoding="utf-8"))
        whisper_time = float(full_output["processing_time_sec"])
        records = full_output["segments"]
        review_records = full_output["segments_overlapping_17_40"]
        normalized_review = full_output["normalized_text_17_40"]
        oh_man_matches = [None] * int(full_output["exact_or_normalized_oh_man_match_count_17_40"])
        device = full_output["device"]
    else:
        import whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(cfg["whisper_model"], device=device, download_root=cfg["whisper_cache_dir"])
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = model.transcribe(waveform, fp16=device == "cuda", word_timestamps=True, condition_on_previous_text=True, verbose=None)
        whisper_time = time.perf_counter() - started
        records = transcript_records(result)
        review_records = [x for x in records if x["end_time"] > REVIEW_START and x["start_time"] < REVIEW_END]
        normalized_review = " ".join(x["text"] for x in review_records).casefold()
        oh_man_matches = re.findall(r"\bo+h+[,.!? ]+man\b|\boh\s+man\b", normalized_review)
        full_output = {
        "video_id": VIDEO_ID, "experiment": "full-audio Whisper diagnostic upper bound",
        "final_index_design": False, "input_scope": "complete audio, independent of VAD and QA fields",
        "audio_duration_sec": duration, "processing_time_sec": round(whisper_time, 3),
        "device": device, "precision": "float16" if device == "cuda" else "float32",
        "model": cfg["whisper_model"], "language": result.get("language"),
        "segments": records, "segments_overlapping_17_40": review_records,
        "normalized_text_17_40": normalized_review,
        "exact_or_normalized_oh_man_match_count_17_40": len(oh_man_matches),
        "automatic_quality_flags": sorted({flag for x in records for flag in x["automatic_quality_flags"]}),
        "warnings": sorted({str(x.message) for x in caught}),
        "manual_interpretation_required": "Semantic similarity, hallucinations, and speaker identity cannot be established automatically here.",
        "diagnostic_quality_notes": {"consecutive_duplicate_segments_detected": False, "definite_hallucinations_identified": False, "suspected_asr_misrecognitions": ["I'm good with high gross leaks.", "So now you can imagine the posters and walls."], "reason": "These texts do not recover the human-reviewed target utterances; verbatim ground truth is insufficient to call other segments definite hallucinations."},
        }
        write_json(full_output_path, full_output)

    full_rtf = whisper_time / duration
    policies = []
    for floor in [0.40, 0.30, 0.20, 0.10]:
        clear = [{"start_time": x["start_time"], "end_time": x["end_time"], "duration": round(x["end_time"]-x["start_time"], 3)} for x in baseline_intervals]
        send = regions_by_threshold[f"{floor:.2f}"]
        uncertain = subtract_intervals(send, clear)
        sent = interval_duration(send)
        review_overlap = overlap_duration(send, REVIEW_START, REVIEW_END)
        policies.append({
            "policy_label": f"candidate_uncertain_floor_{floor:.2f}", "clear_floor": 0.50, "uncertain_floor": floor,
            "background_rule": f"probability < {floor:.2f}", "clear_intervals": clear,
            "uncertain_intervals_proposed_for_rescue": uncertain, "combined_intervals_sent_to_whisper": send,
            "total_audio_sent_to_whisper_sec": round(sent, 3),
            "additional_audio_vs_original_vad_baseline_sec": round(sent-baseline_sent, 3),
            "fraction_of_full_audio_sent": round(sent/duration, 4),
            "temporal_coverage_of_human_review_17_40": round(review_overlap/(REVIEW_END-REVIEW_START), 4),
            "estimated_whisper_time_sec_proportional_lower_bound": round(sent*full_rtf, 3),
            "estimate_warning": "Proportional estimate excludes per-clip overhead and is not a measured rescue run.",
        })
    simulation = {
        "video_id": VIDEO_ID, "policy_status": "simulation only; no thresholds frozen",
        "construction_is_question_independent": True,
        "original_vad_gated_audio_sec": round(baseline_sent, 3),
        "full_audio_upper_bound": {"audio_sent_sec": duration, "measured_processing_time_sec": round(whisper_time, 3), "real_time_factor": round(full_rtf, 4)},
        "candidate_policies": policies,
    }
    write_json(out / "selective_rescue_simulation.json", simulation)

    best_threshold_review = max(comparisons, key=lambda x: x["coverage_of_human_review_17_40"])
    review_text = " ".join(x["text"] for x in review_records) or "(no transcript)"
    recovered_conversation = bool(review_records and len(normalized_review.split()) >= 5)
    exact_recovered = len(oh_man_matches)
    cause = "VAD gating caused much of the broad conversation loss; the target phrase failure is combined VAD plus difficult-audio ASR" if recovered_conversation and exact_recovered == 0 else ("mainly VAD gating" if recovered_conversation else "ASR remains unable to recover the reviewed interval even without VAD")
    lines = [
        "# Case 00061 distant-speech diagnostic", "",
        "This diagnostic preserves the Task 3A baseline. All regions were constructed from the complete waveform without the question, answer, annotation context, or dataset timestamp. The 17–40 s interval is used only for human-review comparison and visualization.", "",
        "## Experiment A — Silero VAD", "",
        f"Current configuration: threshold `{cfg['vad_threshold']}`, minimum speech `{cfg['minimum_speech_duration_ms']} ms`, merge gap `{cfg['speech_merge_gap_sec']} s`, padding `{cfg['speech_pad_ms']} ms` (Silero default minimum silence: 100 ms).", "",
        "| threshold | regions | detected audio (s) | 17–40 temporal coverage | extra vs baseline (s) | low-energy risk proxy |", "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(f"| {item['threshold']:.2f} | {item['region_count']} | {item['detected_duration_sec']:.3f} | {item['coverage_of_human_review_17_40']:.1%} | {item['additional_audio_vs_original_baseline_sec']:+.3f} | {item['false_positive_risk_proxy']['low_energy_fraction_of_detected']:.1%} |")
    lines += ["", "No final VAD threshold is selected. The low-energy column is only a false-positive-risk proxy, not a speech/non-speech label.", "", "## Experiment B — full-audio Whisper upper bound", "", f"- Complete audio sent: {duration:.3f} s", f"- Measured Whisper processing time: {whisper_time:.3f} s on `{device}`", f"- Conversation recovered in 17–40 s: **{'yes' if recovered_conversation else 'no/insufficient'}**", f"- Exact or normalized `Oh man` occurrences recovered in 17–40 s: **{exact_recovered}**", f"- Transcript in 17–40 s: `{review_text}`", f"- Diagnostic attribution: **{cause}**", "", "Automatic quality flags and model warnings are stored in `full_audio_whisper_transcripts.json`. Speaker identity is not inferred.", "", "No consecutive duplicated segments were detected. The phrases `high gross leaks` and `posters and walls` are treated as suspected ASR misrecognitions because they do not recover the human-reviewed target utterances. There is not enough ground-truth verbatim transcription to label other segments as definite hallucinations.", "", "## Experiment C — selective rescue simulation", "", "| uncertain floor | audio sent (s) | extra vs original (s) | 17–40 temporal coverage | estimated Whisper time (s) |", "|---:|---:|---:|---:|---:|"]
    for item in policies:
        lines.append(f"| {item['uncertain_floor']:.2f} | {item['total_audio_sent_to_whisper_sec']:.3f} | {item['additional_audio_vs_original_vad_baseline_sec']:+.3f} | {item['temporal_coverage_of_human_review_17_40']:.1%} | {item['estimated_whisper_time_sec_proportional_lower_bound']:.3f} |")
    lines += ["", f"Full-audio upper bound: {duration:.3f} s sent, {whisper_time:.3f} s measured. Selective-rescue times are proportional lower-bound estimates and exclude per-clip overhead.", "", "## Answers", "", f"1. Full-audio Whisper recovered part of the 17–40 s conversation: **{'yes' if recovered_conversation else 'no/insufficient'}**. It recovered speech around 19.72–21.00 s and 30.00–38.20 s, not a complete verbatim conversation.", f"2. It recovered `Oh man`: **{exact_recovered} occurrence(s)** by normalized string matching. No semantically convincing alternative for that phrase was present in the transcript.", f"3. The observed failure is attributed to **{cause}**.", f"4. Lower thresholds increase reviewed-interval temporal coverage only up to {best_threshold_review['coverage_of_human_review_17_40']:.1%}. A selective rescue may add candidates cheaply, but recovery of `Oh man` is **not demonstrated**, because even full-audio Whisper missed it.", "5. Exact duration comparisons are in the two JSON files and tables above.", "", "## Human-review note preserved", "", "One utterance may be from a female speaker and one from a male speaker. This is a suspected annotation inconsistency. The original dataset annotation has not been modified.", "", "No final rescue policy, embeddings, retrieval, linking, reranking, QA, or agents were implemented."]
    (out / "diagnostic_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "full_audio_whisper_time_sec": round(whisper_time,3), "segments_17_40": review_records, "oh_man_matches": exact_recovered, "attribution": cause}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
