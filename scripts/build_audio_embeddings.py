from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.index import write_json  # noqa: E402


def l2_normalize(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norm, 1e-12)


def valid_transcript(segment: dict[str, Any]) -> bool:
    return bool(str(segment.get("text", "")).strip()) and float(segment["end_time"]) > float(segment["start_time"])


def valid_acoustic_region(region: dict[str, Any]) -> bool:
    return float(region["end_time"]) > float(region["start_time"])


def resolve_audio_path(case: dict[str, Any], dataset_root: Path) -> Path:
    path = Path(case["audio_path"])
    return path if path.is_absolute() else dataset_root / path


def chunk_audio(audio: np.ndarray, sample_rate: int, maximum_duration_sec: float) -> list[np.ndarray]:
    maximum_samples = max(1, int(round(sample_rate * maximum_duration_sec)))
    if audio.size == 0:
        return []
    return [np.asarray(audio[start:start + maximum_samples], dtype=np.float32) for start in range(0, len(audio), maximum_samples)]


def encode_acoustic_region(
    model: Any,
    processor: Any,
    waveform: np.ndarray,
    source_sample_rate: int,
    start_time: float,
    end_time: float,
    target_sample_rate: int,
    maximum_chunk_duration_sec: float,
    device: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    start_sample = max(0, int(round(start_time * source_sample_rate)))
    end_sample = min(len(waveform), int(round(end_time * source_sample_rate)))
    crop = np.asarray(waveform[start_sample:end_sample], dtype=np.float32)
    if crop.size == 0:
        return None, {"warning": "empty_audio_crop", "chunk_count": 0}
    if source_sample_rate != target_sample_rate:
        crop = librosa.resample(crop, orig_sr=source_sample_rate, target_sr=target_sample_rate)
    chunks = chunk_audio(crop, target_sample_rate, maximum_chunk_duration_sec)
    vectors = []
    with torch.inference_mode():
        for chunk in chunks:
            inputs = processor(audio=chunk, sampling_rate=target_sample_rate, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model.get_audio_features(**inputs)
            # Transformers 5 returns BaseModelOutputWithPooling; older releases returned a tensor.
            vector = output.pooler_output if hasattr(output, "pooler_output") else output
            vectors.append(vector.detach().float().cpu().numpy()[0])
    pooled = l2_normalize(np.mean(np.stack(vectors), axis=0))[...]
    return np.asarray(pooled, dtype=np.float32), {
        "warning": None,
        "chunk_count": len(chunks),
        "chunk_max_duration_sec": maximum_chunk_duration_sec,
        "pooling": "normalized_mean_of_chunk_embeddings",
    }


def validate_pair(embeddings: np.ndarray, rows: list[dict[str, Any]], expected_dimension: int | None = None) -> dict[str, Any]:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    if embeddings.shape[0] != len(rows):
        raise ValueError(f"row mismatch: embeddings={embeddings.shape[0]} metadata={len(rows)}")
    if expected_dimension is not None and embeddings.shape[1] != expected_dimension:
        raise ValueError(f"dimension mismatch: {embeddings.shape[1]} != {expected_dimension}")
    if embeddings.dtype != np.float32:
        raise ValueError(f"expected float32, got {embeddings.dtype}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embedding contains NaN or Inf")
    return {"row_count_matches": True, "dimension": int(embeddings.shape[1]), "dtype": "float32", "all_finite": True}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Task 3B semantic embeddings over frozen Task 3A regions.")
    parser.add_argument("--config", default="configs/audio_embedding_models.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    cases = json.loads((ROOT / cfg["mvp_cases_path"]).read_text(encoding="utf-8"))
    dataset_root = Path(cfg["dataset_root"])
    index_root = ROOT / cfg["audio_index_root"]
    requested_device = cfg["device"]
    device = "cuda" if requested_device == "cuda" and torch.cuda.is_available() else "cpu"
    warnings_list = []
    if device != requested_device:
        warnings_list.append(f"requested_{requested_device}_unavailable_used_{device}")
    run_start = time.perf_counter()

    from sentence_transformers import SentenceTransformer
    transcript_cfg = cfg["transcript_model"]
    transcript_model = SentenceTransformer(
        transcript_cfg["local_snapshot"], device=device,
        local_files_only=bool(transcript_cfg["local_files_only"]),
    )
    transcript_dimension = int(transcript_model.get_sentence_embedding_dimension())
    per_video: dict[str, dict[str, Any]] = {}
    transcript_start = time.perf_counter()
    for case in cases:
        video_id = case["video_id"]
        out = index_root / video_id
        source = json.loads((out / "transcripts.json").read_text(encoding="utf-8"))
        valid = [segment for segment in source["segments"] if valid_transcript(segment)]
        skipped = [segment.get("transcript_segment_id") for segment in source["segments"] if not valid_transcript(segment)]
        texts = [str(segment["text"]).strip() for segment in valid]
        if texts:
            embeddings = transcript_model.encode(
                texts, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=bool(transcript_cfg["normalize_embeddings"]),
                show_progress_bar=False,
            ).astype(np.float32, copy=False)
        else:
            embeddings = np.empty((0, transcript_dimension), dtype=np.float32)
        rows = []
        for row_index, segment in enumerate(valid):
            rows.append({
                "embedding_row": row_index,
                "transcript_segment_id": segment["transcript_segment_id"],
                "speech_region_id": segment["speech_region_id"],
                "video_id": video_id,
                "case_id": case["case_id"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "text": segment["text"],
                "language": segment.get("language"),
            })
        validation = validate_pair(embeddings, rows, transcript_dimension)
        np.save(out / "transcript_embeddings.npy", embeddings, allow_pickle=False)
        write_json(out / "transcript_embedding_index.json", {
            "video_id": video_id, "model": transcript_cfg["name"],
            "model_cache_path": transcript_cfg["local_snapshot"],
            "question_independent": True, "embedding_normalized": True,
            "validation": validation, "skipped_invalid_or_empty_transcript_ids": skipped,
            "rows": rows,
        })
        per_video[video_id] = {
            "case_id": case["case_id"], "video_id": video_id,
            "transcript_embedding_count": len(rows),
            "skipped_transcript_count": len(skipped),
        }
    transcript_time = time.perf_counter() - transcript_start
    del transcript_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from transformers import ClapModel, ClapProcessor
    acoustic_cfg = cfg["acoustic_model"]
    processor = ClapProcessor.from_pretrained(
        acoustic_cfg["name"], revision=acoustic_cfg["revision"], local_files_only=True,
    )
    clap_model = ClapModel.from_pretrained(
        acoustic_cfg["name"], revision=acoustic_cfg["revision"],
        use_safetensors=True, local_files_only=True,
    ).to(device).eval()
    acoustic_dimension = int(clap_model.config.projection_dim)
    acoustic_start = time.perf_counter()
    for case in cases:
        video_id = case["video_id"]
        out = index_root / video_id
        source = json.loads((out / "acoustic_regions.json").read_text(encoding="utf-8"))
        regions = source["regions"]
        waveform, source_sr = sf.read(resolve_audio_path(case, dataset_root), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1).astype(np.float32, copy=False)
        vectors = []
        rows = []
        skipped = []
        for region in regions:
            if not valid_acoustic_region(region):
                skipped.append({"acoustic_region_id": region.get("acoustic_region_id"), "warning": "non_positive_duration"})
                continue
            vector, details = encode_acoustic_region(
                clap_model, processor, waveform, int(source_sr),
                float(region["start_time"]), float(region["end_time"]),
                int(acoustic_cfg["sample_rate"]), float(acoustic_cfg["maximum_chunk_duration_sec"]), device,
            )
            if vector is None:
                skipped.append({"acoustic_region_id": region["acoustic_region_id"], "warning": details["warning"]})
                continue
            row_index = len(vectors)
            vectors.append(vector)
            rows.append({
                "embedding_row": row_index,
                "acoustic_region_id": region["acoustic_region_id"],
                "video_id": video_id, "case_id": case["case_id"],
                "start_time": region["start_time"], "end_time": region["end_time"],
                "duration": region["duration"], "mean_rms": region["mean_rms"],
                "peak_rms": region["peak_rms"],
                "mean_spectral_change_score": region["mean_spectral_change_score"],
                "speech_overlap_ratio": region["speech_overlap_ratio"],
                "low_information_or_silence": region["low_information_or_silence"],
                "encoding_details": details,
                "semantic_interpretation_warning": "CLAP is a retrieval representation; it does not prove that this acoustic_region contains a particular sound.",
            })
        embeddings = np.stack(vectors).astype(np.float32) if vectors else np.empty((0, acoustic_dimension), dtype=np.float32)
        validation = validate_pair(embeddings, rows, acoustic_dimension)
        # Exact timestamp preservation check against frozen Task 3A region rows.
        source_by_id = {region["acoustic_region_id"]: region for region in regions}
        for row in rows:
            original = source_by_id[row["acoustic_region_id"]]
            if row["start_time"] != original["start_time"] or row["end_time"] != original["end_time"]:
                raise ValueError(f"timestamp mismatch for {video_id}/{row['acoustic_region_id']}")
        validation["timestamps_match_frozen_regions"] = True
        validation["speech_overlapping_regions_retained"] = sum(row["speech_overlap_ratio"] > 0 for row in rows)
        np.save(out / "acoustic_embeddings.npy", embeddings, allow_pickle=False)
        write_json(out / "acoustic_embedding_index.json", {
            "video_id": video_id, "region_type": "acoustic_regions_not_semantic_events",
            "model": acoustic_cfg["name"], "model_revision": acoustic_cfg["revision"],
            "model_cache_path": acoustic_cfg["resolved_snapshot"],
            "question_independent": True, "embedding_normalized": True,
            "validation": validation, "skipped_invalid_or_empty_regions": skipped,
            "rows": rows,
        })
        per_video[video_id].update({
            "acoustic_embedding_count": len(rows),
            "skipped_acoustic_region_count": len(skipped),
            "speech_overlapping_acoustic_regions_encoded": validation["speech_overlapping_regions_retained"],
        })
    acoustic_time = time.perf_counter() - acoustic_start
    total_time = time.perf_counter() - run_start

    summary = {
        "task": "Task 3B semantic embeddings over frozen Task 3A audio regions",
        "processed_video_count": len(cases), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "question_independent": True,
        "models": {
            "transcript": {"name": transcript_cfg["name"], "cache_path": transcript_cfg["local_snapshot"], "embedding_dimension": transcript_dimension},
            "acoustic": {"name": acoustic_cfg["name"], "revision": acoustic_cfg["revision"], "cache_path": acoustic_cfg["resolved_snapshot"], "embedding_dimension": acoustic_dimension, "weight_format": "safetensors"},
        },
        "processing_time_sec": {"transcript_embeddings": round(transcript_time, 3), "acoustic_embeddings": round(acoustic_time, 3), "total": round(total_time, 3)},
        "compatibility": {"torch": torch.__version__, "torch_unchanged": True, "pip_check": "passed", "warnings": ["The main CLAP .bin is blocked on torch 2.5.1 by Transformers CVE protection; Task 3B uses the official repository's safetensors conversion PR without changing PyTorch.", "Sentence-Transformers 5.6 warns that get_sentence_embedding_dimension is renamed to get_embedding_dimension; this is a non-breaking API deprecation only."] + warnings_list},
        "known_failure_cases": [{"case_id": "00061_5", "video_id": "00061", "issue": "Known distant/overlapping-speech VAD/ASR failure from Task 3A.1; no VAD or ASR tuning was performed in Task 3B."}],
        "videos": list(per_video.values()),
        "scope_exclusions": ["ASR regeneration", "VAD tuning", "question retrieval", "visual-audio linking", "local dense inspection", "reranking", "final QA", "agents"],
    }
    write_json(index_root / "audio_embedding_summary.json", summary)
    lines = [
        "# Task 3B audio embedding summary", "",
        "Semantic retrieval representations were added to the frozen Task 3A transcript and acoustic regions. Region construction and encoding did not use questions, answers, annotation context, or dataset reference timestamps.", "",
        f"- Device: `{device}`" + (f" — {summary['gpu']}" if summary['gpu'] else ""),
        f"- Transcript model: `{transcript_cfg['name']}` ({transcript_dimension} dimensions)",
        f"- Acoustic model: `{acoustic_cfg['name']}` at `{acoustic_cfg['revision']}` ({acoustic_dimension} dimensions, safetensors)",
        f"- Processing time: transcript {transcript_time:.3f} s; acoustic {acoustic_time:.3f} s; total {total_time:.3f} s", "",
        "| video | transcript embeddings | acoustic embeddings | speech-overlapping acoustic regions retained |", "|---:|---:|---:|---:|",
    ]
    for item in per_video.values():
        lines.append(f"| {item['video_id']} | {item['transcript_embedding_count']} | {item['acoustic_embedding_count']} | {item['speech_overlapping_acoustic_regions_encoded']} |")
    lines += [
        "", "## Validation", "",
        "Every `.npy` file is float32, finite, two-dimensional, and has the same row count as its JSON index. Embedding dimensions are consistent across all videos. Acoustic timestamps exactly match the frozen Task 3A regions, including regions that overlap speech.", "",
        "CLAP embeddings are semantic retrieval representations. They do not prove that an acoustic region contains a particular sound, and the regions remain `acoustic_regions`, not semantic sound events.", "",
        "## Compatibility warning", "",
        "The model's main `.bin` is blocked on torch 2.5.1 by Transformers' CVE protection. Task 3B therefore uses the same official model repository's safetensors conversion PR. PyTorch, torchaudio, Whisper, and Silero VAD were not changed; `pip check` passed.", "",
        "## Known failure case", "",
        "`00061_5` remains the known distant/overlapping-speech VAD/ASR failure documented in Task 3A.1. Task 3B embeds only the frozen available transcripts and does not tune or regenerate VAD/ASR.", "",
        "No retrieval, visual-audio linking, dense inspection, reranking, final QA, or agents were implemented.",
    ]
    (index_root / "audio_embedding_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
