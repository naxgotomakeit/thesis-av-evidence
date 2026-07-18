from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audio.index import write_json  # noqa: E402
from src.retrieval.three_channel import (  # noqa: E402
    evaluate_ranked_results,
    normalize_rows,
    rank_index,
    safe_retrieval_case,
    validate_query_dimension,
)

REFERENCE_LABEL = "Dataset-provided QA reference interval\n(not used for retrieval; not a gold event boundary)"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_safe_cases(path: Path) -> list[dict[str, str]]:
    # Every non-allowlisted field is discarded before any model is loaded.
    return [safe_retrieval_case(case) for case in load_json(path)]


def encode_visual_queries(cases: list[dict[str, str]], cfg: dict[str, Any], device: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    # OpenAI CLIP 1.0.1 imports pkg_resources only to access `packaging`.
    # Setuptools 83 removed pkg_resources, so provide that narrow API without
    # downgrading the environment or altering CLIP/model weights.
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        import packaging
        import types
        shim = types.ModuleType("pkg_resources")
        shim.packaging = packaging
        sys.modules["pkg_resources"] = shim
    import clip
    model, _ = clip.load(cfg["model_name"], device=device, download_root=str(Path(cfg["cache_path"]).parent))
    model.eval(); vectors = {}; timings = {}
    with torch.inference_mode():
        for case in cases:
            started = time.perf_counter()
            tokens = clip.tokenize([case["question"]]).to(device)
            vector = model.encode_text(tokens).float()
            vector = vector / vector.norm(dim=-1, keepdim=True)
            if device == "cuda": torch.cuda.synchronize()
            timings[case["case_id"]] = time.perf_counter() - started
            vectors[case["case_id"]] = vector.cpu().numpy()[0].astype(np.float32)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return vectors, timings


def encode_speech_queries(cases: list[dict[str, str]], cfg: dict[str, Any], device: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["cache_path"], device=device, local_files_only=True)
    vectors = {}; timings = {}
    for case in cases:
        started = time.perf_counter()
        vector = model.encode([case["question"]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        if device == "cuda": torch.cuda.synchronize()
        timings[case["case_id"]] = time.perf_counter() - started
        vectors[case["case_id"]] = np.asarray(vector, dtype=np.float32)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return vectors, timings


def encode_acoustic_queries(cases: list[dict[str, str]], cfg: dict[str, Any], device: str) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained(cfg["name"], revision=cfg["revision"], local_files_only=True)
    model = ClapModel.from_pretrained(cfg["name"], revision=cfg["revision"], use_safetensors=True, local_files_only=True).to(device).eval()
    vectors = {}; timings = {}
    with torch.inference_mode():
        for case in cases:
            started = time.perf_counter()
            inputs = processor(text=[case["question"]], return_tensors="pt", padding=True)
            output = model.get_text_features(**{key: value.to(device) for key, value in inputs.items()})
            vector = output.pooler_output if hasattr(output, "pooler_output") else output
            vector = vector.float(); vector = vector / vector.norm(dim=-1, keepdim=True)
            if device == "cuda": torch.cuda.synchronize()
            timings[case["case_id"]] = time.perf_counter() - started
            vectors[case["case_id"]] = vector.cpu().numpy()[0].astype(np.float32)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return vectors, timings


def visual_channel(case: dict[str, str], query: np.ndarray, top_k: int) -> tuple[list[dict[str, Any]], int]:
    folder = ROOT / "outputs/visual_index" / case["video_id"]
    embeddings = np.load(folder / "region_embeddings.npy")
    source = load_json(folder / "visual_state_regions.json")
    rows = source["visual_state_regions"]
    if not validate_query_dimension(query, embeddings): raise ValueError("visual query dimension mismatch")
    ranked = rank_index(query, embeddings, rows, top_k)
    results = []
    for item in ranked:
        row = item["metadata"]
        keyframe = folder / row["representative_keyframe_path"]
        results.append({
            "rank": item["rank"], "similarity_score": item["similarity_score"],
            "visual_region_id": row["region_id"], "start_time": row["start_time"],
            "end_time": row["end_time"], "duration": row["end_time"] - row["start_time"],
            "representative_keyframe_path": keyframe.relative_to(ROOT).as_posix(),
            "original_region_metadata": row,
        })
    return results, len(rows)


def speech_channel(case: dict[str, str], query: np.ndarray, top_k: int) -> tuple[list[dict[str, Any]], int]:
    folder = ROOT / "outputs/audio_index" / case["video_id"]
    embeddings = np.load(folder / "transcript_embeddings.npy")
    index = load_json(folder / "transcript_embedding_index.json")
    rows = index["rows"]
    if not validate_query_dimension(query, embeddings): raise ValueError("speech query dimension mismatch")
    ranked = rank_index(query, embeddings, rows, top_k)
    original_segments = {x["transcript_segment_id"]: x for x in load_json(folder / "transcripts.json")["segments"]}
    speech_regions = {x["speech_region_id"]: x for x in load_json(folder / "speech_regions.json")["regions"]}
    results = []
    for item in ranked:
        row = item["metadata"]; original = original_segments.get(row["transcript_segment_id"], {})
        results.append({
            "rank": item["rank"], "similarity_score": item["similarity_score"],
            "transcript_segment_id": row["transcript_segment_id"], "speech_region_id": row.get("speech_region_id"),
            "start_time": row["start_time"], "end_time": row["end_time"],
            "duration": row["end_time"] - row["start_time"], "transcript_text": row["text"],
            "asr_language": row.get("language"), "whisper_metadata": original.get("whisper_metadata"),
            "speech_region_source_metadata": speech_regions.get(row.get("speech_region_id")),
            "original_transcript_index_metadata": row,
        })
    return results, len(rows)


def acoustic_channel(case: dict[str, str], query: np.ndarray, top_k: int) -> tuple[list[dict[str, Any]], int]:
    folder = ROOT / "outputs/audio_index" / case["video_id"]
    embeddings = np.load(folder / "acoustic_embeddings.npy")
    index = load_json(folder / "acoustic_embedding_index.json")
    rows = index["rows"]
    if not validate_query_dimension(query, embeddings): raise ValueError("acoustic query dimension mismatch")
    ranked = rank_index(query, embeddings, rows, top_k)
    results = []
    for item in ranked:
        row = item["metadata"]
        results.append({
            "rank": item["rank"], "similarity_score": item["similarity_score"],
            "acoustic_region_id": row["acoustic_region_id"], "start_time": row["start_time"],
            "end_time": row["end_time"], "duration": row["duration"],
            "mean_rms": row["mean_rms"], "peak_rms": row["peak_rms"],
            "mean_spectral_change_score": row["mean_spectral_change_score"],
            "speech_overlap_ratio": row["speech_overlap_ratio"],
            "low_information_or_silence": row["low_information_or_silence"],
            "original_acoustic_index_metadata": row,
        })
    return results, len(rows)


def channel_payload(case: dict[str, str], channel: str, model: dict[str, Any], results: list[dict[str, Any]], count: int, encoding_time: float, retrieval_time: float, diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"], "video_id": case["video_id"], "question": case["question"],
        "channel": channel, "query_encoder": model, "similarity_metric": "cosine_similarity",
        "query_and_index_embeddings_normalized": True, "indexed_item_count": count,
        "query_encoding_time_sec": round(encoding_time, 6), "retrieval_time_sec": round(retrieval_time, 6),
        "results": results, "weak_reference_diagnostic": diagnostic,
        "leakage_audit": {"retrieval_inputs": ["case_id", "video_id", "raw_question"], "ranking_completed_before_reference_load": True, "answer_used": False, "answer_options_used": False, "context_used": False, "reference_timestamp_used_for_ranking": False},
    }


def save_timeline(path: Path, video_id: str, all_regions: dict[str, list[dict[str, Any]]], ranked: dict[str, list[dict[str, Any]]], reference: tuple[float, float]) -> None:
    colors = {"visual": "#8b5cf6", "speech": "#2563eb", "acoustic": "#0f9d76"}
    y = {"visual": 3, "speech": 2, "acoustic": 1}
    duration = max([float(r["end_time"]) for rows in all_regions.values() for r in rows] + [reference[1]])
    fig, ax = plt.subplots(figsize=(17, 7))
    for channel, regions in all_regions.items():
        for region in regions:
            ax.broken_barh([(region["start_time"], region["end_time"]-region["start_time"])], (y[channel]-.27, .54), facecolors="#e5e7eb", edgecolors="#9ca3af", linewidth=.45)
    for channel, results in ranked.items():
        for result in results:
            ax.broken_barh([(result["start_time"], result["end_time"]-result["start_time"])], (y[channel]-.27, .54), facecolors=colors[channel], alpha=max(.35, 1-.18*(result["rank"]-1)))
            label = f"#{result['rank']} {result['similarity_score']:.3f}"
            if channel == "speech":
                text = result.get("transcript_text", "").replace("\n", " ")
                text = text.encode("ascii", errors="replace").decode("ascii")
                label += " " + (text[:30] + ("…" if len(text)>30 else ""))
            ax.text((result["start_time"]+result["end_time"])/2, y[channel], label, ha="center", va="center", fontsize=7, color="white" if channel != "speech" or result["duration"] > 2 else "black", rotation=90 if result["duration"] < 2 else 0)
    ax.axvspan(reference[0], reference[1], color="#ec4899", alpha=.18, label=REFERENCE_LABEL)
    # Thumbnails are attached above the visual track without changing ranking.
    for result in ranked["visual"]:
        keyframe = ROOT / result["representative_keyframe_path"]
        if keyframe.is_file():
            image = np.asarray(Image.open(keyframe).convert("RGB"))
            box = AnnotationBbox(OffsetImage(image, zoom=.10), ((result["start_time"]+result["end_time"])/2, 3.75), frameon=True, pad=.15)
            ax.add_artist(box)
    ax.set(xlim=(0, duration), ylim=(.45, 4.25), yticks=[1,2,3], yticklabels=["Acoustic", "Speech", "Visual"], xlabel="Time (s)", title=f"Three-channel independent Top-3 retrieval — {video_id}")
    ax.legend(loc="upper right", fontsize=8); ax.grid(axis="x", alpha=.15)
    fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def result_table(results: list[dict[str, Any]], channel: str) -> list[str]:
    lines = ["| rank | score | interval | result |", "|---:|---:|---|---|"]
    for r in results:
        identifier = r.get("visual_region_id") or r.get("transcript_segment_id") or r.get("acoustic_region_id")
        if channel == "speech": identifier += f": {r['transcript_text']}"
        lines.append(f"| {r['rank']} | {r['similarity_score']:.4f} | {r['start_time']:.3f}–{r['end_time']:.3f} | {identifier} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/retrieval_mvp.yaml")
    args = parser.parse_args(); cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    cases_path = ROOT / cfg["frozen_inputs"]["cases"]
    cases = load_safe_cases(cases_path)
    device = "cuda" if cfg["device"] == "cuda" and torch.cuda.is_available() else "cpu"
    output_root = ROOT / cfg["output_path"]; output_root.mkdir(parents=True, exist_ok=True)

    visual_queries, visual_times = encode_visual_queries(cases, cfg["encoders"]["visual"], device)
    speech_queries, speech_times = encode_speech_queries(cases, cfg["encoders"]["speech"], device)
    acoustic_queries, acoustic_times = encode_acoustic_queries(cases, cfg["encoders"]["acoustic"], device)

    # Ranking phase: no reference timestamp object exists in this phase.
    ranked_cases: dict[str, dict[str, Any]] = {}
    for case in cases:
        started = time.perf_counter()
        v, vc = visual_channel(case, visual_queries[case["case_id"]], int(cfg["top_k"]["visual"]))
        s, sc = speech_channel(case, speech_queries[case["case_id"]], int(cfg["top_k"]["speech"]))
        a, ac = acoustic_channel(case, acoustic_queries[case["case_id"]], int(cfg["top_k"]["acoustic"]))
        ranked_cases[case["case_id"]] = {"case": case, "visual": v, "speech": s, "acoustic": a, "counts": {"visual": vc, "speech": sc, "acoustic": ac}, "retrieval_time": time.perf_counter()-started}

    # Diagnostic phase begins only now, after every channel/case ranking is frozen in memory.
    reference_rows = {x["case_id"]: x for x in load_json(cases_path)}
    case_summaries = []
    for case_id, bundle in ranked_cases.items():
        case = bundle["case"]; source_case = reference_rows[case_id]
        ref_start, ref_end = source_case.get("provided_timestamp_start"), source_case.get("provided_timestamp_end")
        warnings_list = []
        if ref_start is None or ref_end is None:
            warnings_list.append("dataset_reference_interval_unavailable")
            diagnostics = {channel: {"top1_overlap": False, "top3_overlap": False, "minimum_temporal_distance_sec": None, "closest_retrieved_rank": None, "closest_retrieved_similarity_score": None} for channel in ("visual","speech","acoustic")}
            reference = None
        else:
            reference = (float(ref_start), float(ref_end))
            diagnostics = {channel: evaluate_ranked_results(bundle[channel], *reference) for channel in ("visual","speech","acoustic")}
        union_top1 = any(diagnostics[x]["top1_overlap"] for x in diagnostics)
        union_top3 = any(diagnostics[x]["top3_overlap"] for x in diagnostics)
        if len(bundle["visual"]) < cfg["top_k"]["visual"]: warnings_list.append("visual_index_has_fewer_than_top_k_regions")
        if len(bundle["speech"]) < cfg["top_k"]["speech"]: warnings_list.append("speech_index_has_fewer_than_top_k_segments")
        if len(bundle["acoustic"]) < cfg["top_k"]["acoustic"]: warnings_list.append("acoustic_index_has_fewer_than_top_k_regions")
        if any(not str(result.get("transcript_text", "")).isascii() for result in bundle["speech"]):
            warnings_list.append("timeline_non_ascii_transcript_label_replaced_for_default_font_compatibility_json_text_preserved")
        known = None
        if case_id == "00061_5":
            known = {"distant_or_overlapping_speech_difficulty": True, "target_phrase_missing_from_frozen_asr_index": True, "suspected_dataset_speaker_attribution_inconsistency": True, "purpose_for_inclusion": "test whether visual or acoustic retrieval can compensate for the weak speech channel", "original_annotation_modified": False, "retrieval_tuned_for_case": False}
            warnings_list.append("known_00061_distant_overlapping_speech_and_suspected_speaker_annotation_inconsistency")
        out = output_root / case_id; out.mkdir(parents=True, exist_ok=True)
        model_cfg = cfg["encoders"]
        payloads = {
            "visual": channel_payload(case,"visual",model_cfg["visual"],bundle["visual"],bundle["counts"]["visual"],visual_times[case_id],bundle["retrieval_time"],diagnostics["visual"]),
            "speech": channel_payload(case,"speech",model_cfg["speech"],bundle["speech"],bundle["counts"]["speech"],speech_times[case_id],bundle["retrieval_time"],diagnostics["speech"]),
            "acoustic": channel_payload(case,"acoustic",model_cfg["acoustic"],bundle["acoustic"],bundle["counts"]["acoustic"],acoustic_times[case_id],bundle["retrieval_time"],diagnostics["acoustic"]),
        }
        for channel, payload in payloads.items(): write_json(out / f"{channel}_retrieval.json", payload)
        combined = {"case_id":case_id,"video_id":case["video_id"],"question":case["question"],"channels_not_fused":True,"visual_results":bundle["visual"],"speech_results":bundle["speech"],"acoustic_results":bundle["acoustic"],"weak_reference_interval": {"start":ref_start,"end":ref_end,"label":REFERENCE_LABEL,"loaded_after_ranking":True},"diagnostics":diagnostics,"union_diagnostics":{"union_top1_overlap":union_top1,"union_top3_overlap":union_top3},"query_encoding_time_sec":{"visual":visual_times[case_id],"speech":speech_times[case_id],"acoustic":acoustic_times[case_id]},"retrieval_time_sec":bundle["retrieval_time"],"known_case_note":known,"warnings":warnings_list}
        write_json(out / "three_channel_retrieval.json", combined)
        if reference:
            visual_all = load_json(ROOT/"outputs/visual_index"/case["video_id"]/"visual_state_regions.json")["visual_state_regions"]
            speech_all = load_json(ROOT/"outputs/audio_index"/case["video_id"]/"transcript_embedding_index.json")["rows"]
            acoustic_all = load_json(ROOT/"outputs/audio_index"/case["video_id"]/"acoustic_embedding_index.json")["rows"]
            save_timeline(out/"retrieval_timeline.png",case["video_id"],{"visual":visual_all,"speech":speech_all,"acoustic":acoustic_all},{"visual":bundle["visual"],"speech":bundle["speech"],"acoustic":bundle["acoustic"]},reference)
        lines = [f"# Retrieval report — {case_id}","",f"**Question:** {case['question']}","","Ranking used only the raw question and the case/video IDs needed to locate frozen indexes. The weak reference interval was loaded only after ranking.",""]
        for channel in ("visual","speech","acoustic"):
            d=diagnostics[channel]; lines += [f"## {channel.title()} Top-3",""]+result_table(bundle[channel],channel)+["",f"Top-1 overlap: `{d['top1_overlap']}`; Top-3 overlap: `{d['top3_overlap']}`; closest distance: `{d['minimum_temporal_distance_sec']}` s; closest rank: `{d['closest_retrieved_rank']}`.",""]
        lines += ["## Union diagnostic","",f"- Union Top-1 overlap: `{union_top1}`",f"- Union Top-3 overlap: `{union_top3}`","", "## Timing","",f"- Query encoding: visual {visual_times[case_id]:.6f}s, speech {speech_times[case_id]:.6f}s, acoustic {acoustic_times[case_id]:.6f}s",f"- Retrieval: {bundle['retrieval_time']:.6f}s","", "## Warnings","" ] + ([f"- {x}" for x in warnings_list] or ["- None."])
        if known: lines += ["","## Known 00061 limitation","","The target phrase was not recovered by the frozen ASR index. This case has distant/overlapping speech and suspected speaker-attribution inconsistency in the dataset. It remains included to test whether visual or acoustic retrieval can compensate for the weak speech channel. The original annotation was not modified, and retrieval was not tuned for this case."]
        (out/"retrieval_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        case_summaries.append({"case_id":case_id,"video_id":case["video_id"],"diagnostics":diagnostics,"union_top1_overlap":union_top1,"union_top3_overlap":union_top3,"indexed_counts":bundle["counts"],"query_encoding_time_sec":{"visual":visual_times[case_id],"speech":speech_times[case_id],"acoustic":acoustic_times[case_id]},"retrieval_time_sec":bundle["retrieval_time"],"warnings":warnings_list})

    channels=("visual","speech","acoustic"); n=len(case_summaries)
    aggregate={f"{ch}_top1_overlap_count":sum(x["diagnostics"][ch]["top1_overlap"] for x in case_summaries) for ch in channels}
    aggregate.update({f"{ch}_top3_overlap_count":sum(x["diagnostics"][ch]["top3_overlap"] for x in case_summaries) for ch in channels})
    aggregate.update({"denominator":n,"union_top1_overlap_count":sum(x["union_top1_overlap"] for x in case_summaries),"union_top3_overlap_count":sum(x["union_top3_overlap"] for x in case_summaries)})
    averages={"query_encoding_time_sec":{ch:float(np.mean([x["query_encoding_time_sec"][ch] for x in case_summaries])) for ch in channels},"retrieval_time_per_case_sec":float(np.mean([x["retrieval_time_sec"] for x in case_summaries])),"indexed_regions_per_case":{ch:float(np.mean([x["indexed_counts"][ch] for x in case_summaries])) for ch in channels}}
    summary={"task":"Task 4 independent question-conditioned three-channel candidate retrieval","case_count":n,"device":device,"no_cross_modal_score_fusion":True,"no_leakage_design":{"retrieval_allowlist":["case_id","video_id","raw_question"],"reference_loaded_after_all_rankings":True},"compatibility_warnings":["OpenAI CLIP 1.0.1 expects pkg_resources.packaging, removed by setuptools 83; a narrow runtime shim exposes the installed packaging module without downgrading the environment."],"aggregate":aggregate,"averages":averages,"cases":case_summaries}
    write_json(output_root/"retrieval_summary.json",summary)
    lines=["# Task 4 retrieval summary","","Independent Top-3 candidate retrieval was run in CLIP visual, Sentence-T5 speech, and CLAP acoustic spaces. Scores were normalized and ranked only within their own channel; no cross-modal fusion was performed.","","| diagnostic | overlap / 6 |","|---|---:|"]
    for ch in channels: lines += [f"| {ch} Top-1 | {aggregate[f'{ch}_top1_overlap_count']} / {n} |",f"| {ch} Top-3 | {aggregate[f'{ch}_top3_overlap_count']} / {n} |"]
    lines += [f"| multimodal union Top-1 | {aggregate['union_top1_overlap_count']} / {n} |",f"| multimodal union Top-3 | {aggregate['union_top3_overlap_count']} / {n} |","","The dataset interval is a weak diagnostic reference, not a precise gold event boundary. No-overlap is not definitive failure because regions may be coarse and evidence may occur nearby.","","## Average timing and search size","",f"- Query encoding: visual {averages['query_encoding_time_sec']['visual']:.6f}s; speech {averages['query_encoding_time_sec']['speech']:.6f}s; acoustic {averages['query_encoding_time_sec']['acoustic']:.6f}s",f"- Retrieval per case: {averages['retrieval_time_per_case_sec']:.6f}s",f"- Indexed items searched per case: visual {averages['indexed_regions_per_case']['visual']:.2f}; speech {averages['indexed_regions_per_case']['speech']:.2f}; acoustic {averages['indexed_regions_per_case']['acoustic']:.2f}","","`00061_5` remains included as the known distant/overlapping-speech case with suspected speaker-attribution inconsistency. The original annotation and frozen indexes were not modified, and no case-specific tuning was performed.","","No candidate fusion, temporal linking, dense inspection, reranking, final QA, or agents were implemented."]
    (output_root/"retrieval_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
