"""Offline-only V7.4 contract checks; never loads an encoder or calls a provider."""
from __future__ import annotations
import copy, hashlib, json, os, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SUPPORT = ROOT / "runtime_support"
INDEX = pathlib.Path(os.getenv("INDEX_ROOT") or ROOT / "work_index").expanduser().resolve()
OUT = ROOT / "outputs"
sys.path[:0] = [str(ROOT / "runnable_runtime"), str(SUPPORT), str(SUPPORT / "src")]
from experiments.hourvideo_v7_4_variant_c_budgets_v1.retriever import PROFILES, load_case, rank_case

def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def scrub(value):
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in {"latency_sec", "retrieval_latency_sec"}}
    if isinstance(value, list): return [scrub(v) for v in value]
    return value
def sec(value):
    h,m,s = [int(x) for x in str(value).split(":")]
    return h*3600+m*60+s

def main():
    cases = sorted(p for p in INDEX.joinpath("cases").iterdir() if p.is_dir())
    q = np.linspace(-1, 1, 768, dtype=np.float32); q /= np.linalg.norm(q)
    rows=[]; failures=[]; signatures=set(); totals={"medium":0,"coarse":0,"fine":0}
    for case_dir in cases:
        case = load_case(case_dir)
        totals["medium"] += len(case["hierarchy"]["medium_nodes"]); totals["fine"] += len(case["hierarchy"]["fine_nodes"])
        totals["coarse"] += len(case["navigation"]["coarse_regions"])
        case_row={"video_id":case_dir.name,"profiles":{}}
        for name, profile in PROFILES.items():
            spans, meta = rank_case(case, query="person performing an action", query_embedding=q, profile=profile, min_time_gap_sec=15)
            sig=tuple(sorted(spans[0].keys())) if spans else ("start_time","end_time","caption")
            signatures.add(sig)
            valid=all(0 <= sec(x["start_time"]) < sec(x["end_time"]) <= case["hierarchy"]["duration_sec"] for x in spans)
            budget=len(spans) <= profile.output_top_k
            if not (valid and budget): failures.append({"video_id":case_dir.name,"profile":name,"valid":valid,"budget":budget})
            if name == "h30":
                if not (meta["all_coarse_expanded"] and meta["all_medium_expanded"] and not meta["coarse_gate_applied"] and not meta["medium_gate_applied"]):
                    failures.append({"video_id":case_dir.name,"profile":name,"reason":"gate_flags"})
            else:
                expected=min(profile.coarse_top_k, len(case["navigation"]["coarse_regions"]))
                expected_medium=min(profile.medium_top_k, meta["counts"]["medium_eligible"])
                if meta["counts"]["coarse_selected"] != expected or meta["counts"]["medium_selected"] != expected_medium:
                    failures.append({"video_id":case_dir.name,"profile":name,"reason":"gate_width"})
            for span in spans:
                if set(span) != {"start_time","end_time","caption"}: failures.append({"video_id":case_dir.name,"profile":name,"reason":"schema"})
            # Determinism (latency is scrubbed).
            spans2, meta2 = rank_case(case, query="person performing an action", query_embedding=q, profile=profile, min_time_gap_sec=15)
            if scrub(spans) != scrub(spans2) or scrub(meta) != scrub(meta2): failures.append({"video_id":case_dir.name,"profile":name,"reason":"nondeterministic"})
            case_row["profiles"][name]={"candidate_count":len(spans),"metadata":meta,"schema":list(sig)}
        # Portability: source_frame_path is not used by ranking.
        no_frames=copy.deepcopy(case)
        for node in no_frames["hierarchy"]["fine_nodes"]: node["source_frame_path"]=""
        a,_=rank_case(case, query="person performing an action", query_embedding=q, profile=PROFILES["h6"], min_time_gap_sec=15)
        b,_=rank_case(no_frames, query="person performing an action", query_embedding=q, profile=PROFILES["h6"], min_time_gap_sec=15)
        case_row["no_frame_ranking_equal"] = a == b
        if a != b: failures.append({"video_id":case_dir.name,"reason":"frame_path_affects_ranking"})
        rows.append(case_row)
    manifest=INDEX/"manifests/materialization_manifest.json"
    report={"status":"PASS" if not failures and len(cases)==12 and len(signatures)==1 else "FAIL","video_count":len(cases),"totals":totals,"contract_layers":{"tool_input":"visual_retrieve(query: string)","internal_raw_candidate_schema":["start_time","end_time","caption"],"planner_visible_response_schema":{"summary":"string"}},"internal_raw_candidate_schema_signatures":[list(x) for x in sorted(signatures)],"failures":failures,"cases":rows,"map_hash_manifest":sha256(manifest) if manifest.exists() else None,"api_called":False,"model_loaded":False,"qa_used":False}
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"offline_contract_v74.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"videos":len(cases),"failures":len(failures),"totals":totals}))
    return 0 if report["status"] == "PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
