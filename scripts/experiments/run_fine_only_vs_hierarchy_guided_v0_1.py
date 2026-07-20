"""Run the isolated zero-API Fine-only vs reusable hierarchy retrieval study."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.fine_only_vs_hierarchy_guided.retrieval import (  # noqa: E402
    add_fine_only_proxy_metrics, fine_only_retrieval, hierarchy_guided_retrieval, normalize,
)
from src.experiments.fine_only_vs_hierarchy_guided.reporting import render  # noqa: E402

CONFIG_PATH = ROOT / "config/experiments/fine_only_vs_hierarchy_guided_v0_1.json"
OUT = ROOT / "outputs/experiments/fine_only_vs_hierarchy_guided_v0_1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def encode_clip_queries(questions: list[str], device: str) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import clip
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            raise
        import packaging
        import packaging.version  # expose packaging.version for legacy openai-clip
        compatibility = types.ModuleType("pkg_resources")
        compatibility.packaging = packaging
        sys.modules["pkg_resources"] = compatibility
        import clip
    started = time.perf_counter()
    model, _ = clip.load("ViT-B/32", device=device, download_root="C:/Users/72977/.cache/clip")
    model.eval()
    load_sec = time.perf_counter() - started
    encode_started = time.perf_counter()
    with torch.inference_mode():
        tokens = clip.tokenize(questions, truncate=True).to(device)
        features = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
    if device == "cuda":
        torch.cuda.synchronize()
    return features.cpu().numpy().astype(np.float32), {
        "model": "OpenAI CLIP ViT-B/32", "device": device,
        "model_load_sec": load_sec, "query_encode_sec": time.perf_counter() - encode_started,
        "query_count": len(questions), "external_api_calls": 0,
    }


def build_clip_prototypes(
    hierarchy: dict[str, Any], frame_embeddings: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray], list[dict[str, Any]]]:
    frames = normalize(frame_embeddings)
    nodes = {row["node_id"]: row for row in hierarchy["nodes"]}
    fine_ids = hierarchy["cuts"]["fine"]["node_ids"]
    fine_embeddings: dict[str, np.ndarray] = {}
    for identifier in fine_ids:
        node = nodes[identifier]
        indices = slice(int(node["frame_index_start"]), int(node["frame_index_end"]) + 1)
        fine_embeddings[identifier] = normalize(frames[indices].mean(axis=0))
    prototypes: dict[str, dict[str, np.ndarray]] = {}
    output_rows = []
    merge_by_parent = {row["parent_id"]: row for row in hierarchy["accepted_merge_order"]}
    for node in hierarchy["nodes"]:
        indices = slice(int(node["frame_index_start"]), int(node["frame_index_end"]) + 1)
        centroid = normalize(frames[indices].mean(axis=0))
        descendants = node["leaf_ids"]
        matrix = np.stack([fine_embeddings[identifier] for identifier in descendants])
        medoid_index = int(np.argmax(matrix @ centroid))
        medoid_id = descendants[medoid_index]
        medoid = fine_embeddings[medoid_id]
        prototypes[node["node_id"]] = {"centroid": centroid, "medoid": medoid}
        output_rows.append({
            "video_id": hierarchy["video_id"], "node_id": node["node_id"],
            "node_type": node["node_type"], "start": node["start"], "end": node["end"],
            "duration": node["duration"], "child_ids": node["child_ids"], "parent_id": node["parent_id"],
            "leaf_ids": node["leaf_ids"], "leftmost_leaf_index": node["leftmost_leaf_index"],
            "rightmost_leaf_index": node["rightmost_leaf_index"], "internal_variability": node["internal_variability"],
            "internal_boundary_ids": node["internal_boundary_ids"], "left_boundary": node["left_boundary"],
            "right_boundary": node["right_boundary"], "representative_frames": node["representative_frames"],
            "source_pooled_dinov2_sha256": node["pooled_dinov2_sha256"],
            "clip_centroid": centroid.tolist(), "clip_centroid_sha256": array_hash(centroid),
            "clip_medoid": medoid.tolist(), "clip_medoid_sha256": array_hash(medoid),
            "clip_medoid_descendant_fine_id": medoid_id,
            "merge_record": merge_by_parent.get(node["node_id"]),
        })
    return prototypes, fine_embeddings, output_rows


def benchmark(function, repeats: int) -> dict[str, float]:
    values = []
    for _ in range(repeats):
        values.append(float(function()["retrieval_sec"]))
    return {
        "repeats": repeats, "mean_sec": statistics.fmean(values), "median_sec": statistics.median(values),
        "p95_sec": float(np.quantile(values, 0.95)), "min_sec": min(values), "max_sec": max(values),
    }


def aggregate(rows: list[dict[str, Any]], primary_beam: int, budget: int) -> dict[str, Any]:
    fine = [row["fine_only"] for row in rows]
    guided = [row["hierarchy_guided"] for row in rows]
    mean_a = statistics.fmean(row["fine_nodes_scored"] for row in fine)
    mean_b_fine = statistics.fmean(row["fine_nodes_scored"] for row in guided)
    mean_b_nodes = statistics.fmean(row["total_node_score_operations"] for row in guided)
    mean_b_vectors = statistics.fmean(row["total_vector_comparisons"] for row in guided)
    result = {
        "case_count": len(rows), "primary_beam_width": primary_beam, "final_evidence_budget": budget,
        "fine_only": {
            "mean_fine_nodes_scored": mean_a,
            "mean_total_node_score_operations": statistics.fmean(row["total_node_score_operations"] for row in fine),
            "mean_total_vector_comparisons": statistics.fmean(row["total_vector_comparisons"] for row in fine),
            "mean_final_duration_sec": statistics.fmean(row["activated_temporal_duration"] for row in fine),
            "mean_retrieval_latency_sec": statistics.fmean(row["benchmark"]["mean_sec"] for row in fine),
        },
        "hierarchy_guided_primary": {
            "mean_operating_view_nodes_scored": statistics.fmean(row["operating_view_nodes_scored"] for row in guided),
            "mean_internal_parent_nodes_scored": statistics.fmean(row["parent_nodes_scored"] for row in guided),
            "mean_parent_prototype_comparisons": statistics.fmean(row["parent_prototype_comparisons"] for row in guided),
            "mean_fine_nodes_scored": mean_b_fine,
            "mean_total_node_score_operations": mean_b_nodes,
            "mean_total_vector_comparisons": mean_b_vectors,
            "fine_score_reduction": 1.0 - mean_b_fine / mean_a,
            "total_node_score_reduction": 1.0 - mean_b_nodes / mean_a,
            "total_vector_comparison_reduction": 1.0 - mean_b_vectors / mean_a,
            "mean_fine_activation_fraction": statistics.fmean(row["fine_activation_fraction"] for row in guided),
            "mean_unique_branches_visited": statistics.fmean(row["unique_branches_visited"] for row in guided),
            "mean_final_duration_sec": statistics.fmean(row["activated_temporal_duration"] for row in guided),
            "mean_retrieval_latency_sec": statistics.fmean(row["benchmark"]["mean_sec"] for row in guided),
            "fine_only_top1_reach_rate": statistics.fmean(float(row["diagnostic_proxy"]["fine_only_top1_reached"]) for row in guided),
            "fine_only_top3_reach_rate": statistics.fmean(row["diagnostic_proxy"]["fine_only_top3_reached_rate"] for row in guided),
            "final_selected_id_overlap_rate": statistics.fmean(row["diagnostic_proxy"]["final_selected_id_overlap_rate"] for row in guided),
            "cases_pruning_fine_only_top1": sum(not row["diagnostic_proxy"]["fine_only_top1_reached"] for row in guided),
        },
    }
    sensitivity = {}
    for beam in (1, 2, 3):
        values = [row["sensitivity"][str(beam)] for row in rows]
        sensitivity[str(beam)] = {
            "mean_fine_nodes_scored": statistics.fmean(row["fine_nodes_scored"] for row in values),
            "mean_total_node_score_operations": statistics.fmean(row["total_node_score_operations"] for row in values),
            "mean_total_vector_comparisons": statistics.fmean(row["total_vector_comparisons"] for row in values),
            "fine_only_top1_reach_rate": statistics.fmean(float(row["diagnostic_proxy"]["fine_only_top1_reached"]) for row in values),
            "fine_only_top3_reach_rate": statistics.fmean(row["diagnostic_proxy"]["fine_only_top3_reached_rate"] for row in values),
            "final_selected_id_overlap_rate": statistics.fmean(row["diagnostic_proxy"]["final_selected_id_overlap_rate"] for row in values),
        }
    result["beam_sensitivity"] = sensitivity
    result["failure_category_counts"] = dict(Counter(category for row in rows for category in row["structural_failure_categories"]))
    return result


def main() -> None:
    config = load_json(CONFIG_PATH)
    paths = {key: ROOT / config[key] for key in [
        "source_manifest", "question_source", "fine_source", "hierarchy_source", "hierarchy_config_source",
        "safe_merge_source", "frame_grid_source", "old_hybrid_reference", "taxonomy_source",
    ]}
    for key, path in paths.items():
        if not path.exists() and key not in {"old_hybrid_reference", "taxonomy_source"}:
            raise FileNotFoundError(f"Missing frozen source {key}: {path}")
    manifest = load_json(paths["source_manifest"])
    fine_rows = load_jsonl(paths["fine_source"])
    hierarchy_rows = load_jsonl(paths["hierarchy_source"])
    question_rows = load_jsonl(paths["question_source"])
    ids = [row["video_id"] for row in manifest["videos"]]
    if len(ids) != 10 or ids != [row["video_id"] for row in fine_rows] or ids != [row["video_id"] for row in hierarchy_rows]:
        raise RuntimeError("Frozen 10-video Fine/hierarchy lineage is ambiguous or mismatched")
    questions = {row["video_id"]: row for row in question_rows}
    if set(ids) != set(questions):
        raise RuntimeError("Question source does not exactly cover the frozen videos")
    if any(row.get("answer_options_used") or row.get("gold_used") or row.get("qa_correctness_used") for row in question_rows):
        raise RuntimeError("Forbidden question-source leakage flag")
    taxonomy = {}
    if paths["taxonomy_source"].exists():
        taxonomy = {row["question_id"]: row for row in load_jsonl(paths["taxonomy_source"]) if row.get("question_id") in set(ids)}
    old_reference = {row["video_id"]: row for row in load_jsonl(paths["old_hybrid_reference"])} if paths["old_hybrid_reference"].exists() else {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_embeddings, clip_runtime = encode_clip_queries([questions[identifier]["question"] for identifier in ids], device)
    retrieval_cfg = config["retrieval"]
    primary_beam, budget, repeats = int(retrieval_cfg["primary_beam_width"]), int(retrieval_cfg["final_fine_evidence_count"]), int(retrieval_cfg["benchmark_repeats"])
    all_nodes, all_edges, all_merges, results, per_metrics = [], [], [], [], []
    report_hierarchies = {}
    offline_per_video = []
    fine_source_by_id = {row["video_id"]: row for row in fine_rows}
    for case_index, (manifest_row, hierarchy) in enumerate(zip(manifest["videos"], hierarchy_rows)):
        video_id = manifest_row["video_id"]
        fine_source = fine_source_by_id[video_id]
        if hierarchy["fine_leaf_ids"] != [row["segment_id"] for row in fine_source["segments"]]:
            raise RuntimeError(f"Fine identity drift in hierarchy {video_id}")
        started = time.perf_counter()
        frame_path = ROOT / config["clip_frame_root"] / video_id / "frame_embeddings.npy"
        frame_embeddings = np.load(frame_path)
        prototypes, fine_embeddings, node_rows = build_clip_prototypes(hierarchy, frame_embeddings)
        prototype_sec = time.perf_counter() - started
        augmented_by_id = {row["node_id"]: row for row in node_rows}
        report_hierarchy = {**hierarchy, "nodes": node_rows}
        report_hierarchies[video_id] = report_hierarchy
        all_nodes.extend(node_rows)
        all_edges.extend({"video_id": video_id, "parent_id": row["node_id"], "child_id": child, "child_order": order} for row in node_rows for order, child in enumerate(row["child_ids"]))
        all_merges.extend({"video_id": video_id, **row} for row in hierarchy["accepted_merge_order"])
        nodes = augmented_by_id
        fine_ids = hierarchy["cuts"]["fine"]["node_ids"]
        medium_ids = hierarchy["cuts"]["medium"]["node_ids"]
        coarse_ids = hierarchy["cuts"]["coarse"]["node_ids"]
        query = query_embeddings[case_index]
        fine_call = lambda: fine_only_retrieval(query=query, fine_ids=fine_ids, fine_embeddings=fine_embeddings, nodes=nodes, final_budget=budget)
        fine_result = fine_call()
        fine_result["benchmark"] = benchmark(fine_call, repeats)
        guided_call = lambda beam=primary_beam: hierarchy_guided_retrieval(
            query=query, fine_ids=fine_ids, medium_ids=medium_ids, coarse_ids=coarse_ids,
            fine_embeddings=fine_embeddings, parent_prototypes=prototypes, nodes=nodes,
            beam_width=beam, final_budget=budget,
            centroid_weight=float(retrieval_cfg["centroid_weight"]), medoid_weight=float(retrieval_cfg["medoid_weight"]),
        )
        guided = add_fine_only_proxy_metrics(fine_result, guided_call())
        guided["benchmark"] = benchmark(guided_call, repeats)
        sensitivity = {}
        for beam in retrieval_cfg["beam_sensitivity"]:
            sensitivity[str(beam)] = add_fine_only_proxy_metrics(fine_result, guided_call(int(beam)))
        previous = questions[video_id]
        previous_ids = [row["segment_id"] for row in previous["ranking"]]
        current_ids = [row["node_id"] for row in fine_result["ranking"]]
        max_score_delta = max(abs(a["score"] - b["score"]) for a, b in zip(previous["ranking"], fine_result["ranking"]))
        if previous_ids != current_ids or max_score_delta > 1e-6:
            raise RuntimeError(f"Frozen CLIP Fine ranking mismatch for {video_id}: {max_score_delta}")
        categories = []
        if guided["fine_nodes_scored"] < fine_result["fine_nodes_scored"] and guided["diagnostic_proxy"]["fine_only_top1_reached"]:
            categories.append("A_hierarchy_prunes_fine_while_top1_reachable")
        if not guided["diagnostic_proxy"]["fine_only_top1_reached"]:
            categories.append("B_hierarchy_prunes_fine_only_top1_proxy")
        if guided["diagnostic_proxy"]["fine_only_top3_reached_rate"] < 1.0:
            categories.append("B_hierarchy_prunes_fine_only_top3_proxy")
        if guided["diagnostic_proxy"]["fine_only_top3_reached_rate"] < 1.0 and any(nodes[row["node_id"]]["duration"] >= 90 for row in guided["selected_coarse"]):
            categories.append("C_large_parent_abstraction_risk")
        if len(fine_result["ranking"]) >= 5 and fine_result["ranking"][0]["score"] - fine_result["ranking"][4]["score"] < 0.01:
            categories.append("D_fine_score_competition_redundancy_proxy")
        scope = taxonomy.get(video_id, {}).get("scope")
        if scope in {"global", "multi_event"} and guided["diagnostic_proxy"]["fine_only_top3_reached_rate"] < 1.0:
            categories.append("E_broad_or_multi_event_coverage_risk")
        explanation = (
            "Automatic categories compare hierarchy output with Fine-only Top-3 as a diagnostic proxy; "
            "they do not establish ground-truth evidence relevance."
        )
        result = {
            "case_id": questions[video_id]["case_id"], "video_id": video_id,
            "question": questions[video_id]["question"], "group": manifest_row["group"],
            "frozen_scope": scope, "scope_provenance": config["taxonomy_source"] if scope else None,
            "answer_options_used": False, "gold_used": False, "qa_correctness_used": False,
            "fine_only": fine_result, "hierarchy_guided": guided, "sensitivity": sensitivity,
            "fine_ranking_regression": {"match": True, "max_score_delta": max_score_delta, "reference": config["question_source"]},
            "structural_failure_categories": categories, "structural_explanation": explanation,
        }
        results.append(result)
        per_metrics.append({
            "video_id": video_id, "group": manifest_row["group"], "frozen_scope": scope,
            "fine_nodes": len(fine_ids), "medium_reference_nodes": len(medium_ids), "coarse_reference_nodes": len(coarse_ids),
            "fine_only_fine_scored": fine_result["fine_nodes_scored"],
            "guided_parent_cut_nodes_scored": guided["operating_view_nodes_scored"],
            "guided_parent_prototype_comparisons": guided["parent_prototype_comparisons"],
            "guided_fine_scored": guided["fine_nodes_scored"], "guided_total_node_scores": guided["total_node_score_operations"],
            "guided_total_vector_comparisons": guided["total_vector_comparisons"],
            "fine_score_reduction": 1-guided["fine_nodes_scored"]/fine_result["fine_nodes_scored"],
            "total_node_score_reduction": 1-guided["total_node_score_operations"]/fine_result["total_node_score_operations"],
            "top1_proxy_reached": guided["diagnostic_proxy"]["fine_only_top1_reached"],
            "top3_proxy_reach_rate": guided["diagnostic_proxy"]["fine_only_top3_reached_rate"],
            "final_id_overlap_rate": guided["diagnostic_proxy"]["final_selected_id_overlap_rate"],
            "fine_only_final_ids": [row["node_id"] for row in fine_result["selected_final"]],
            "guided_final_ids": [row["node_id"] for row in guided["selected_final"]],
            "fine_only_final_duration": fine_result["activated_temporal_duration"],
            "guided_final_duration": guided["activated_temporal_duration"],
            "structural_failure_categories": categories, "structural_explanation": explanation,
        })
        offline_per_video.append({"video_id": video_id, "clip_frame_embedding_load_and_parent_prototype_sec": prototype_sec, "clip_frame_embedding_sha256": sha256(frame_path)})
    agg = aggregate(results, primary_beam, budget)
    hashes = {key: sha256(path) for key, path in paths.items() if path.exists()}
    branch, head, status = git("branch", "--show-current"), git("rev-parse", "HEAD"), git("status", "--short")
    canonical_diff = git("diff", "--name-only", "--", "src/canonical_pipeline", "config/canonical_pipeline.json", "CURRENT_BASELINE.md")
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUT/"hierarchy_nodes.jsonl", all_nodes)
    write_jsonl(OUT/"hierarchy_edges.jsonl", all_edges)
    write_jsonl(OUT/"merge_trace.jsonl", all_merges)
    write_jsonl(OUT/"retrieval_results.jsonl", results)
    write_json(OUT/"per_video_metrics.json", per_metrics)
    write_json(OUT/"aggregate_metrics.json", agg)
    frozen = {**config, "source_hashes": hashes, "git_branch": branch, "git_head": head, "working_tree_status": status, "canonical_tracked_diff": bool(canonical_diff), "device": device, "experiment_timestamp_utc": __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    write_json(OUT/"frozen_config.json", frozen)
    run_manifest = {"experiment_id": config["experiment_id"], "video_ids": ids, "case_ids": [questions[i]["case_id"] for i in ids], "question_text_hashes": {i: hashlib.sha256(questions[i]["question"].encode()).hexdigest() for i in ids}, "fine_leaf_count": sum(len(row["fine_leaf_ids"]) for row in hierarchy_rows), "fine_preservation_rate": 1.0, "hierarchy_source_reused": True, "answer_options_used": False, "gold_used": False, "qa_correctness_used": False, "external_api_calls": 0}
    write_json(OUT/"run_manifest.json", run_manifest)
    runtime = {"offline_reusable": {"hierarchy_construction_sec_current_run": 0.0, "hierarchy_reused_from": config["hierarchy_source"], "per_video_parent_representation": offline_per_video, "mean_parent_representation_sec": statistics.fmean(row["clip_frame_embedding_load_and_parent_prototype_sec"] for row in offline_per_video)}, "shared_query_encoder": clip_runtime, "online_per_question": {"fine_only_mean_sec": agg["fine_only"]["mean_retrieval_latency_sec"], "hierarchy_guided_mean_sec": agg["hierarchy_guided_primary"]["mean_retrieval_latency_sec"], "benchmark_repeats": repeats}, "api_calls": 0}
    write_json(OUT/"runtime_metrics.json", runtime)
    (OUT/"assets").mkdir(exist_ok=True)
    (OUT/"assets/README.md").write_text("No thumbnails duplicated. HTML references the frozen outputs/visual_index frame assets and validates them.\n", encoding="utf-8")
    render(output=OUT/"comparison.html", aggregate=agg, results=results, hierarchies=report_hierarchies, metrics=per_metrics, old_reference=old_reference)
    serialized = sum((OUT/name).stat().st_size for name in ["hierarchy_nodes.jsonl","hierarchy_edges.jsonl","merge_trace.jsonl"])
    runtime["offline_reusable"]["hierarchy_serialized_size_bytes"] = serialized
    write_json(OUT/"runtime_metrics.json", runtime)
    (OUT/"README.md").write_text(f"""# Fine-only vs hierarchy-guided retrieval v0.1

Zero-API comparison over the exact frozen 10 videos and 159 immutable CoMET-style Event leaves.

- Hierarchy: exact reused Boundary-aware Safe Merge full trees.
- Reference operating cuts: ~50% and ~25%; not natural semantic levels.
- Retrieval: question-only OpenAI CLIP ViT-B/32.
- Primary beam: {primary_beam}; final Fine budget: {budget} for both methods.
- Fine-only Top-3 comparisons are diagnostic proxies, not evidence ground truth.
- Gold/options/QA correctness/API calls: 0.

Open `comparison.html` for timelines, branch traces, thumbnails, and blank manual-review controls.
""", encoding="utf-8")
    print(json.dumps({"cases": len(results), "fine_reduction": agg["hierarchy_guided_primary"]["fine_score_reduction"], "total_node_reduction": agg["hierarchy_guided_primary"]["total_node_score_reduction"], "top1_reach": agg["hierarchy_guided_primary"]["fine_only_top1_reach_rate"], "api_calls": 0, "output": OUT.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
