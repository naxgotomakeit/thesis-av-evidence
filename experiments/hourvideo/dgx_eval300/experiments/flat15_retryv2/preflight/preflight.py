from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
UID_FILE = Path("/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt")
PARQUET = Path("/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_dev_v1.0_videoseal_dgx.parquet")
INDEX_ROOT = Path("/home/naxucl/data/HourVideo/videoseal_original/indexes/semantic")
PARENT_CONFIG = Path("/home/naxucl/data/HourVideo/videoseal_original/runs_dgx_eval300_v1/experiment_config.json")
FORMAL_TEMPLATE = Path("/home/naxucl/data/HourVideo/videoseal_original/videoseal_flat15_eval300_formal_v1_<UTC>")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_env(path: Path) -> dict[str, str]:
    out = {}
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            k, v = s.split("=", 1)
            out[k] = v
    return out


uids_file_order = [x.strip() for x in UID_FILE.read_text().splitlines() if x.strip()]
uid_set = set(uids_file_order)
assert len(uids_file_order) == len(uid_set) == 300

# Project only identity/asset columns. The ground_truth field is never read.
t = pq.read_table(PARQUET, columns=["extra_info.qa_uid", "extra_info.video_id", "extra_info.VIDEO_PATH", "extra_info.VISUAL_INDEX_DIR"])
cols = {n: t[n].to_pylist() for n in t.column_names}
rows = [dict(zip(t.column_names, values)) for values in zip(*(cols[n] for n in t.column_names)) if values[0] in uid_set]
assert len(rows) == 300
assert len({r["qa_uid"] for r in rows}) == 300
assert {r["qa_uid"] for r in rows} == uid_set
ordered = [r["qa_uid"] for r in rows]
videos = sorted({r["video_id"] for r in rows})
assert len(videos) == 12

asset_rows = []
for vid in videos:
    vr = [r for r in rows if r["video_id"] == vid]
    vpaths = sorted(set(r["VIDEO_PATH"] for r in vr))
    ipaths = sorted(set(r["VISUAL_INDEX_DIR"] for r in vr))
    assert len(vpaths) == len(ipaths) == 1
    vp, ip = Path(vpaths[0]), Path(ipaths[0])
    required = [ip / "semantic_captions.json", ip / "semantic_vectors.npy", ip / "semantic_norms.npy", ip / "semantic_doc_ids.txt", ip / "semantic_meta.json"]
    asset_rows.append({
        "video_id": vid,
        "questions": len(vr),
        "video_path": str(vp),
        "video_exists": vp.is_file(),
        "index_path": str(ip),
        "required": {str(p): {"exists": p.is_file(), "sha256": sha(p) if p.is_file() else None} for p in required},
    })
assert all(x["video_exists"] and all(x["required"].values()) for x in asset_rows)

env = parse_env(ROOT / "flat15.env")
parent = json.loads(PARENT_CONFIG.read_text())
penv = parent["parameters"]["runner_environment"]
assert env["VISUAL_RETRIEVE_TOPK"] == "15"
assert penv["VISUAL_RETRIEVE_TOPK"] == "30"
for key, value in env.items():
    if key in {"EXPERIMENT_NAME", "VISUAL_RETRIEVE_TOPK", "RETRIEVE_MIN_TIME_GAP_SEC"}:
        continue
    if key in penv:
        assert str(penv[key]) == value, (key, penv[key], value)
assert env["CONCURRENCY"] == "1"
assert env["SEMANTIC_RETRIEVE_MIX"] == "embed"
assert env["SEMANTIC_RETRIEVE_TOPK"] == "40"

active_30 = [k for k, v in env.items() if v == "30"]
assert active_30 == ["MLLM_RETRY_DELAY", "EMBED_RETRY_DELAY"]  # delays, not evidence budgets

runtime = ROOT / "runtime_overlay_v2"
parent_runtime = Path("/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/VideoSEAL_eval300_runtime_reference_20260820T174344Z/source")
diffs = []
for p in sorted(parent_runtime.rglob("*")):
    if p.is_file():
        rel = p.relative_to(parent_runtime)
        q = runtime / rel
        if not q.is_file() or sha(p) != sha(q):
            diffs.append(str(rel))
assert diffs == ["scripts/dgx/common.env", "scripts/dgx/smoke_one.sh", "videoseal/agents/tool_agent.py"]

(ROOT / "ordered_eval300_uids.txt").write_text("\n".join(ordered) + "\n")
(ROOT / "video_asset_inventory.json").write_text(json.dumps(asset_rows, indent=2) + "\n")
result = {
    "status": "PASS",
    "gold_loaded": False,
    "api_calls": 0,
    "model_calls": 0,
    "uid_count": len(ordered),
    "uid_unique": len(set(ordered)),
    "uid_set_sha256": hashlib.sha256(("\n".join(sorted(uid_set)) + "\n").encode()).hexdigest(),
    "ordered_uid_sha256": sha(ROOT / "ordered_eval300_uids.txt"),
    "frozen_uid_file_sha256": sha(UID_FILE),
    "video_count": len(videos),
    "videos": videos,
    "all_assets_available": True,
    "flat_index_asset_closure_sha256": hashlib.sha256(json.dumps(asset_rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    "flat30_budget": 30,
    "flat15_budget": 15,
    "budget_semantics": "retrieved Flat caption-segment candidates supplied to Summarizer per visual_retrieve call",
    "runtime_diff_from_frozen_flat30": diffs,
    "runtime_diff_reason": "deployment-only CODE_ROOT overlay pin, Flat-15-only VISUAL_RETRIEVE_TOPK=15, plus approved full-video fallback endpoint correctness patch",
    "formal_output_namespace_template": str(FORMAL_TEMPLATE),
    "formal_output_exists": FORMAL_TEMPLATE.exists(),
}
(ROOT / "preflight_report.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
