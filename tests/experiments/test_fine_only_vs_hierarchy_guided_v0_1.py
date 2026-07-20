from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np

from src.experiments.fine_only_vs_hierarchy_guided.retrieval import (
    add_fine_only_proxy_metrics, fine_only_retrieval, hierarchy_guided_retrieval,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/fine_only_vs_hierarchy_guided_v0_1"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def fixture():
    nodes = {}
    for index in range(6):
        identifier = f"f{index}"
        nodes[identifier] = {"node_id": identifier, "node_type": "fine_leaf", "start": 10.0*index, "end": 10.0*(index+1), "duration": 10.0, "leaf_ids": [identifier]}
    nodes.update({
        "m0": {"node_id":"m0","node_type":"internal","start":0.,"end":20.,"duration":20.,"leaf_ids":["f0","f1"]},
        "m1": {"node_id":"m1","node_type":"internal","start":20.,"end":40.,"duration":20.,"leaf_ids":["f2","f3"]},
        "m2": {"node_id":"m2","node_type":"internal","start":40.,"end":60.,"duration":20.,"leaf_ids":["f4","f5"]},
        "c0": {"node_id":"c0","node_type":"internal","start":0.,"end":40.,"duration":40.,"leaf_ids":["f0","f1","f2","f3"]},
        "c1": {"node_id":"c1","node_type":"internal","start":40.,"end":60.,"duration":20.,"leaf_ids":["f4","f5"]},
    })
    fine = {f"f{i}": np.asarray([1.,0.]) if i < 4 else np.asarray([0.,1.]) for i in range(6)}
    proto = {identifier:{"centroid":(np.asarray([1.,0.]) if identifier in {"m0","m1","c0"} else np.asarray([0.,1.])),"medoid":(np.asarray([1.,0.]) if identifier in {"m0","m1","c0"} else np.asarray([0.,1.]))} for identifier in ["m0","m1","m2","c0","c1"]}
    return nodes, fine, proto


def test_exact_frozen_video_and_fine_hierarchy_reuse() -> None:
    manifest=json.loads((ROOT/"data/manifests/coarse_segmentation_3way_10.json").read_text(encoding="utf-8"))
    fine=load_jsonl(ROOT/"outputs/experiments/coarse_segmentation_3way_v0_1/comet_segments.jsonl")
    hier=load_jsonl(ROOT/"outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl")
    ids=[row["video_id"] for row in manifest["videos"]]
    assert len(ids)==10==len(set(ids))
    assert ids==[row["video_id"] for row in fine]==[row["video_id"] for row in hier]
    for f,h in zip(fine,hier):
        assert h["fine_leaf_ids"]==[row["segment_id"] for row in f["segments"]]
        assert h["invariants"]["fine_leaf_preservation_rate"]==1.0
        assert h["invariants"]["reversible_to_original_fine_leaves"] is True


def test_hierarchy_lineage_contiguous_and_adjacency_only() -> None:
    for hierarchy in load_jsonl(ROOT/"outputs/experiments/fine_to_coarse_hierarchy_v0_1/safe_merge_hierarchies.jsonl"):
        nodes={row["node_id"]:row for row in hierarchy["nodes"]}
        for node in hierarchy["nodes"]:
            if node["parent_id"] is not None:
                assert node["node_id"] in nodes[node["parent_id"]]["child_ids"]
            if node["child_ids"]:
                left,right=[nodes[x] for x in node["child_ids"]]
                assert left["end"]==right["start"]
                assert node["start"]==left["start"] and node["end"]==right["end"]
                assert left["rightmost_leaf_index"]+1==right["leftmost_leaf_index"]
        assert set(hierarchy["cuts"]["fine"]["node_ids"])==set(hierarchy["fine_leaf_ids"])


def test_fine_only_scores_every_fine_once_and_budget_matches() -> None:
    nodes,fine,_=fixture(); ids=list(fine)
    result=fine_only_retrieval(query=np.asarray([1.,0.]),fine_ids=ids,fine_embeddings=fine,nodes=nodes,final_budget=3)
    assert result["fine_nodes_scored"]==len(ids)
    assert len(result["ranking"])==len(ids)==len({row["node_id"] for row in result["ranking"]})
    assert len(result["selected_final"])==3


def test_guided_scores_only_fine_inside_selected_medium_branches() -> None:
    nodes,fine,proto=fixture(); ids=list(fine)
    result=hierarchy_guided_retrieval(query=np.asarray([0.,1.]),fine_ids=ids,medium_ids=["m0","m1","m2"],coarse_ids=["c0","c1"],fine_embeddings=fine,parent_prototypes=proto,nodes=nodes,beam_width=1,final_budget=1,centroid_weight=.5,medoid_weight=.5)
    assert [row["node_id"] for row in result["selected_coarse"]]==["c1"]
    assert [row["node_id"] for row in result["selected_medium"]]==["m2"]
    assert result["visited_fine_ids"]==["f4","f5"]
    assert result["fine_nodes_scored"]==2
    assert set(result["pruned_fine_ids"])=={"f0","f1","f2","f3"}


def test_parent_scoring_does_not_receive_descendant_fine_embeddings() -> None:
    source=(ROOT/"src/experiments/fine_only_vs_hierarchy_guided/retrieval.py").read_text(encoding="utf-8")
    function=source[source.index("def score_parent_nodes"):source.index("def fine_only_retrieval")]
    assert "fine_embeddings" not in function
    assert "leaf_ids" not in function
    assert "parent_prototypes" in function


def test_deterministic_navigation_and_proxy() -> None:
    nodes,fine,proto=fixture(); kwargs=dict(query=np.asarray([1.,0.]),fine_ids=list(fine),medium_ids=["m0","m1","m2"],coarse_ids=["c0","c1"],fine_embeddings=fine,parent_prototypes=proto,nodes=nodes,beam_width=2,final_budget=3,centroid_weight=.5,medoid_weight=.5)
    a=hierarchy_guided_retrieval(**kwargs); b=hierarchy_guided_retrieval(**kwargs)
    for row in (a,b): row.pop("retrieval_sec")
    assert a==b
    reference=fine_only_retrieval(query=np.asarray([1.,0.]),fine_ids=list(fine),fine_embeddings=fine,nodes=nodes,final_budget=3)
    assert add_fine_only_proxy_metrics(reference,a)["diagnostic_proxy"]["fine_only_top1_reached"] is True


def test_zero_api_no_leakage_and_no_canonical_diff() -> None:
    config=json.loads((ROOT/"config/experiments/fine_only_vs_hierarchy_guided_v0_1.json").read_text(encoding="utf-8"))
    assert config["external_api_calls"]==0 and config["canonical_pipeline_modified"] is False
    questions=load_jsonl(ROOT/config["question_source"])
    assert all(not row["answer_options_used"] and not row["gold_used"] and not row["qa_correctness_used"] for row in questions)
    diff=subprocess.run(["git","diff","--name-only","--","src/canonical_pipeline","config/canonical_pipeline.json","CURRENT_BASELINE.md"],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    assert diff==""


def test_outputs_and_html_assets_when_present() -> None:
    if not (OUT/"comparison.html").exists(): return
    required=["comparison.html","per_video_metrics.json","aggregate_metrics.json","runtime_metrics.json","frozen_config.json","run_manifest.json","README.md","hierarchy_nodes.jsonl","hierarchy_edges.jsonl","retrieval_results.jsonl","merge_trace.jsonl"]
    assert all((OUT/name).exists() for name in required)
    html=(OUT/"comparison.html").read_text(encoding="utf-8")
    manifest=json.loads((ROOT/"data/manifests/coarse_segmentation_3way_10.json").read_text(encoding="utf-8"))
    assert sum(f'id="case-{row["video_id"].replace("-","_")}"' in html for row in manifest["videos"])==10
    nodes=load_jsonl(OUT/"hierarchy_nodes.jsonl")
    for node in nodes:
        for frame in node["representative_frames"]:
            assert (ROOT/frame["frame_path"]).exists()
    run=json.loads((OUT/"run_manifest.json").read_text(encoding="utf-8"))
    assert run["fine_preservation_rate"]==1.0 and run["external_api_calls"]==0
