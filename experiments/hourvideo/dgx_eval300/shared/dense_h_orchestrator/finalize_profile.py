#!/usr/bin/env python3
"""Create immutable, source-preserving first-pass and retry-augmented reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from state_tool import attempted, complete, read_json, uids, video_id


def pct(values: list[float], q: float) -> float | None:
    if not values: return None
    rows=sorted(values); pos=(len(rows)-1)*q; lo=int(pos); hi=min(lo+1,len(rows)-1); f=pos-lo
    return rows[lo]*(1-f)+rows[hi]*f


def collect(root: Path, uid: str) -> dict[str, Any]:
    ok, data = complete(root, uid); metric=data["metric"]; traj=data["trajectory"]; pred=data["prediction"]
    retrieves=[]; inspects=[]; planner=[]; parser_failures=0
    for step in traj.get("steps") or []:
        if not isinstance(step,dict): continue
        timing=step.get("timing") if isinstance(step.get("timing"),dict) else {}
        if timing.get("model_elapsed_sec") is not None: planner.append(float(timing["model_elapsed_sec"]))
        action=step.get("action") if isinstance(step.get("action"),dict) else {}
        obs=step.get("observation") if isinstance(step.get("observation"),dict) else {}
        meta=obs.get("metadata") if isinstance(obs.get("metadata"),dict) else {}
        if action.get("name")=="visual_retrieve": retrieves.append(meta)
        elif action.get("name")=="visual_inspect": inspects.append((obs,meta))
        if not action.get("name") and str(traj.get("note") or "").lower().find("invalid response")>=0: parser_failures += 1
    return {"uid":uid,"video_id":video_id(uid),"complete":ok,"status":metric.get("status"),"error_type":metric.get("error_type"),
            "pred":str(pred.get("pred") or "").upper(),"gt":str(pred.get("gt") or "").upper(),"elapsed_sec":metric.get("elapsed_sec"),
            "steps":metric.get("steps"),"planner_calls":metric.get("planner_calls"),"planner_latency_sec":planner,
            "retrieval_calls":len(retrieves),"retrieval_latency_sec":[x.get("retrieval_latency_sec") for x in retrieves if x.get("retrieval_latency_sec") is not None],
            "summarizer_latency_sec":[x.get("model_elapsed_sec") for x in retrieves if x.get("model_elapsed_sec") is not None],
            "raw_candidate_counts":[x.get("raw_candidate_count") for x in retrieves],"hierarchy_counts":[x.get("counts") for x in retrieves],
            "profile_flags":[{k:x.get(k) for k in ("all_coarse_expanded","all_medium_expanded","coarse_gate_applied","medium_gate_applied")} for x in retrieves],
            "provenance":[x.get("coarse_medium_fine_paths") for x in retrieves],"inspector_calls":len(inspects),
            "inspector_latency_sec":[m.get("model_elapsed_sec") for _,m in inspects if m.get("model_elapsed_sec") is not None],
            "inspector_frames":sum(int(m.get("sent_image_count") or m.get("image_count") or 0) for _,m in inspects),
            "fallback":any(o.get("forced") is True and o.get("mode")=="full_video" for o,_ in inspects),"parser_failure_count":parser_failures,
            "trajectory_path":data["trajectory_path"],"prediction_path":str(root/video_id(uid)/"preds"/f"{uid}.json")}


def summarize(rows: list[dict[str,Any]]) -> dict[str,Any]:
    comp=[x for x in rows if x["complete"]]; elapsed=[float(x["elapsed_sec"]) for x in comp if x.get("elapsed_sec") is not None]
    planner=[float(v) for x in rows for v in x["planner_latency_sec"] if v is not None]
    retrieval=[float(v) for x in rows for v in x["retrieval_latency_sec"] if v is not None]
    summarizer=[float(v) for x in rows for v in x["summarizer_latency_sec"] if v is not None]
    inspector=[float(v) for x in rows for v in x["inspector_latency_sec"] if v is not None]
    correct=sum(x["pred"]==x["gt"] and bool(x["gt"]) for x in comp)
    raw_with_pred=[x for x in rows if x["pred"] in set("ABCDE")]
    raw_correct=sum(x["pred"]==x["gt"] and bool(x["gt"]) for x in raw_with_pred)
    return {"attempted":sum(x["status"] in {"success","timeout","failed","incomplete"} for x in rows),"completed":len(comp),
            "timeout":sum(x["status"]=="timeout" or x["error_type"]=="timeout" for x in rows),"error":sum(x["status"]=="failed" for x in rows),
            "raw_accuracy":raw_correct/len(raw_with_pred) if raw_with_pred else None,"strict_completed_only_accuracy":correct/len(comp) if comp else None,
            "retrieval_calls":sum(x["retrieval_calls"] for x in rows),"raw_candidate_count_distribution":sorted(n for x in rows for n in x["raw_candidate_counts"] if isinstance(n,int)),
            "direct_answer_count":sum(x["complete"] and x["inspector_calls"]==0 for x in rows),"direct_answer_ratio":sum(x["complete"] and x["inspector_calls"]==0 for x in rows)/len(comp) if comp else None,
            "inspector_calls":sum(x["inspector_calls"] for x in rows),"inspector_frames":sum(x["inspector_frames"] for x in rows),
            "fallback_count":sum(bool(x["fallback"]) for x in rows),"fallback_ratio":sum(bool(x["fallback"]) for x in rows)/len(rows) if rows else None,
            "parser_failures":sum(x["parser_failure_count"] for x in rows),
            "latency_sec":{
                name:{"count":len(vals),"mean":statistics.mean(vals) if vals else None,"median":statistics.median(vals) if vals else None,"p90":pct(vals,.9),"p95":pct(vals,.95)}
                for name,vals in (("e2e",elapsed),("planner",planner),("retrieval",retrieval),("summarizer",summarizer),("inspector",inspector))
            }}


def write_immutable(path: Path, payload: Any) -> None:
    text=json.dumps(payload,indent=2,sort_keys=True)+"\n"
    if path.exists() and path.read_text(encoding="utf-8") != text: raise SystemExit(f"immutable file drift: {path}")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text,encoding="utf-8")


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--profile",required=True); ap.add_argument("--uids",required=True); ap.add_argument("--first",required=True); ap.add_argument("--retry",required=True); ap.add_argument("--flat",required=True); ap.add_argument("--profile-root",required=True); ns=ap.parse_args()
    ids=uids(Path(ns.uids)); first=[collect(Path(ns.first),u) for u in ids]; retry=[collect(Path(ns.retry),u) for u in ids]
    merged=[]; recovered=[]
    for a,b in zip(first,retry):
        selected=b if attempted(Path(ns.retry), a["uid"]) else a
        merged.append(selected)
        if not a["complete"] and b["complete"]: recovered.append(a["uid"])
    flat={x["uid"]:x for x in (collect(Path(ns.flat),u) for u in ids)}
    paired=[x for x in merged if x["complete"] and flat[x["uid"]]["complete"]]
    paired_h=sum(x["pred"]==x["gt"] and bool(x["gt"]) for x in paired); paired_f=sum(flat[x["uid"]]["pred"]==flat[x["uid"]]["gt"] and bool(flat[x["uid"]]["gt"]) for x in paired)
    retry_summary=summarize(retry)
    report={"profile":ns.profile,"first_pass":summarize(first),"retry_attempted":retry_summary["attempted"],
            "retry_timeout":retry_summary["timeout"],"retry_error":retry_summary["error"],"retry_recovered":len(recovered),"retry_augmented":summarize(merged),
            "remaining_incomplete_uids":[x["uid"] for x in merged if not x["complete"]],"recovered_uids":recovered,
            "paired_with_flat":{"intersection_completed":len(paired),"hierarchical_accuracy":paired_h/len(paired) if paired else None,"flat_accuracy":paired_f/len(paired) if paired else None}}
    root=Path(ns.profile_root); write_immutable(root/"status"/"final_report.json",report)
    write_immutable(root/"merged"/"per_question_manifest.json",merged)
    digest=hashlib.sha256((root/"merged"/"per_question_manifest.json").read_bytes()).hexdigest()
    write_immutable(root/"merged"/"immutable_manifest.json",{"profile":ns.profile,"records":len(merged),"sha256":digest})
    print(json.dumps(report,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
