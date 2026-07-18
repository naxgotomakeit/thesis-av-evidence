from __future__ import annotations

import copy, html, json, os, sys, time
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.retrieval.task5b import canonical_visual_frames,package_micro_frames  # noqa:E402

BASE=ROOT/"outputs/planner_guided_retrieval"
OUT=BASE/"v1_1"

def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def save(path:Path,obj:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def declared_fps(index:dict[str,Any],source_index_path:Path)->float|None:
    for key in ("fps","sampling_fps","frame_sampling_fps"):
        value=index.get(key)
        if isinstance(value,(int,float)) and value>0:return float(value)
    video_id=index.get("video_id")
    coarse=ROOT/"outputs/visual_index"/str(video_id)/"visual_state_regions.json"
    if coarse.is_file():
        metadata=load(coarse)
        for value in (metadata.get("fps"),metadata.get("sampling_fps"),(metadata.get("config") or {}).get("fps")):
            if isinstance(value,(int,float)) and value>0:return float(value)
    return None

def corrected_micro(candidate:dict,index_cache:dict[str,tuple[dict,dict[str,dict],float|None]])->dict:
    source_path=str(candidate["source_index"])
    if source_path not in index_cache:
        path=ROOT/source_path;index=load(path);lookup={x["microclip_id"]:x for x in index["microclips"]};index_cache[source_path]=(index,lookup,declared_fps(index,path))
    index,lookup,fps=index_cache[source_path]
    source=lookup.get(candidate["microclip_id"])
    if source is None:
        result=copy.deepcopy(candidate);result["source_frame_paths"]=list(candidate.get("frame_paths",[]));result["source_frame_timestamps"]=[];result["selected_frame_paths"]=[];result["selected_frame_timestamps"]=[];result["frame_paths"]=[];result.setdefault("warnings",[]).append("unable_to_resolve_micro_frame_timestamps");return result
    result=package_micro_frames(candidate,source,fps)
    if index.get("video_id") is not None:result["video_id"]=str(index["video_id"])
    return result

def protected_signature(record:dict)->dict:
    return {"task5a_plan_used":record["task5a_plan_used"],"anchor_resolution":record["anchor_resolution"],"executed_modalities":record["executed_modalities"],"skipped_modalities":record["skipped_modalities"],"candidate_ids":[x["candidate_id"] for x in record["all_candidates"]],"selected_candidate_ids":[x["candidate_id"] for x in record["selected_candidates"]],"candidate_intervals":[(x["candidate_id"],x["start_time"],x["end_time"]) for x in record["all_candidates"]],"linked_windows":record["linked_windows"],"budget":record["budget"],"selected_duration_sec":record["selected_duration_sec"],"routing_trace":record["routing_trace"]}

def correct_record(base:dict)->dict:
    result=copy.deepcopy(base);cache={};corrected={}
    for candidate in base["all_candidates"]:
        if candidate.get("candidate_type")=="micro_window":corrected[candidate["candidate_id"]]=corrected_micro(candidate,cache)
    def rewrite(items:list[dict])->list[dict]:return [copy.deepcopy(corrected.get(x["candidate_id"],x)) for x in items]
    result["all_candidates"]=rewrite(base["all_candidates"]);result["selected_candidates"]=rewrite(base["selected_candidates"])
    result["local_visual_refinement"]["micro_windows"]=rewrite(base["local_visual_refinement"]["micro_windows"])
    selected_micro=[x for x in result["selected_candidates"] if x.get("candidate_type")=="micro_window"]
    dense=list(result["local_visual_refinement"].get("dense_frames",[]));video_id=next((x.get("video_id") for x in selected_micro if x.get("video_id")),None);canonical=canonical_visual_frames(selected_micro,dense,video_id=video_id)
    accounting={"source_micro_frame_count":sum(len(x.get("source_frame_paths",[])) for x in selected_micro),"selected_micro_frame_count":sum(len(x.get("selected_frame_paths",[])) for x in selected_micro),"dense_frame_count":len(dense),"unique_selected_visual_frame_count":len(canonical),"old_task5b_selected_visual_frame_count":base["selected_visual_frame_count"],"timestamp_normalization":"video_id + timestamp rounded to nearest millisecond","deduplication_policy":"dense frame preferred as canonical representation; all provenance retained"}
    result["visual_evidence_accounting"]=accounting;result["selected_visual_evidence_frames"]=canonical;result["local_visual_refinement"]["selected_visual_evidence_frames"]=copy.deepcopy(canonical);result["selected_visual_frame_count"]=len(canonical);result["task5b_correction_version"]="v1.1";result["correction_scope"]="visual frame packaging and deduplicated efficiency accounting only"
    assert protected_signature(result)==protected_signature(base),"v1.1 correction changed protected retrieval behavior"
    return result

def asset(path:str)->str:return Path(os.path.relpath(ROOT/path,OUT)).as_posix()
def pre(value:Any)->str:return f"<pre>{html.escape(json.dumps(value,ensure_ascii=False,indent=2))}</pre>"
def make_html(records:list[dict],summary:dict)->str:
    cases=[]
    for r in records:
        local=r["local_visual_refinement"];micro=[]
        for x in local["micro_windows"]:
            micro.append(f'''<section><h4>{x["microclip_id"]}</h4><p>source interval: {x["source_start_time"]:.3f}–{x["source_end_time"]:.3f}s<br>selected interval: {x["start_time"]:.3f}–{x["end_time"]:.3f}s</p><h5>source_frame_paths / source_frame_timestamps</h5>{pre({"paths":x["source_frame_paths"],"timestamps":x["source_frame_timestamps"]})}<h5>selected_frame_paths / selected_frame_timestamps（半开区间）</h5>{pre({"paths":x["selected_frame_paths"],"timestamps":x["selected_frame_timestamps"]})}</section>''')
        images=[]
        for frame in r["selected_visual_evidence_frames"]:images.append(f'<figure><img src="{html.escape(asset(frame["canonical_frame_path"]))}"><figcaption>{frame["timestamp"]:.3f}s · {frame["provenance"]}</figcaption></figure>')
        cases.append(f'''<article><h2>{r["case_id"]}</h2><h3>原始问题</h3><p>{html.escape(r["question"])}</p><h3>修正范围</h3><p>仅修正视觉帧打包与去重效率统计；路由、区间、候选排序、连接、预算和 reference evaluation 均保持不变。</p><h3>源 microclip 与已选局部区间</h3>{''.join(micro) or '<p>本案例未选择 micro-window。</p>'}<h3>新提取的 dense frames</h3>{pre(local["dense_frames"])}<h3>去重后的 canonical visual evidence</h3><div class=gallery>{''.join(images)}</div>{pre(r["selected_visual_evidence_frames"])}<h3>修正前后帧计数</h3>{pre(r["visual_evidence_accounting"])}<h3>警告</h3>{pre(sorted({w for x in local["micro_windows"] for w in x.get("warnings",[])}))}<h3>完整 Task 5B v1.1 JSON</h3>{pre(r)}</article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 5B v1.1 视觉证据打包修正</title><style>body{{font:15px system-ui;margin:2rem;max-width:1500px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:3rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:450px;overflow:auto}}.gallery{{display:flex;gap:.5rem;overflow:auto}}figure{{margin:0}}img{{height:150px}}</style><h1>Task 5B v1.1：视觉证据打包与效率统计修正</h1><h2>实验概览</h2><p>frame_paths 现在是 selected_frame_paths 的兼容别名。帧筛选采用 selected_start ≤ timestamp &lt; selected_end。Dense 与 micro 帧按 video_id + 毫秒时间戳去重，并优先使用 dense frame。</p>{pre(summary["visual_frame_accounting"])}{''.join(cases)}'''

def main()->int:
    started=time.perf_counter();base_candidates=BASE/"task5b_candidates.jsonl";base_summary_path=BASE/"task5b_summary.json"
    if not base_candidates.is_file() or not base_summary_path.is_file():raise SystemExit("Task 5B base outputs are missing")
    base_records=[json.loads(x) for x in base_candidates.read_text(encoding="utf-8").splitlines() if x.strip()];records=[correct_record(x) for x in base_records];OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/"task5b_candidates.jsonl").open("w",encoding="utf-8") as handle:
        for record in records:handle.write(json.dumps(record,ensure_ascii=False)+"\n")
    summary=copy.deepcopy(load(base_summary_path));old_count=summary["aggregate"]["selected_visual_frame_count"];new_count=sum(x["selected_visual_frame_count"] for x in records)
    summary["task"]="Task 5B v1.1 visual evidence packaging correction";summary["task5b_correction_version"]="v1.1";summary["source_task5b_outputs"]={"candidates":str(base_candidates.relative_to(ROOT)).replace("\\","/"),"summary":str(base_summary_path.relative_to(ROOT)).replace("\\","/")};summary["aggregate"]["selected_visual_frame_count"]=new_count
    for record in records:
        comparison=summary["task4_comparison"][record["case_id"]]["task5b"];comparison["visual_frame_or_clip_count"]=record["selected_visual_frame_count"]+record["selected_visual_clip_count"]
    summary["visual_frame_accounting"]={"old_selected_visual_frame_count":old_count,"corrected_selected_visual_frame_count":new_count,"per_case":{x["case_id"]:x["visual_evidence_accounting"] for x in records},"half_open_interval_semantics":True,"dense_frame_preferred_on_duplicate_timestamp":True};summary["correction_runtime"]={"task5a_llm_api_calls":0,"task5b_llm_api_calls":0,"latency_sec":time.perf_counter()-started};summary["unchanged_behavior"]={"task5a_plans":True,"anchor_resolution":True,"modality_routing":True,"search_intervals":True,"candidate_ids_and_intervals":True,"linked_windows":True,"evidence_budget":True,"posthoc_reference_evaluation":True}
    save(OUT/"task5b_summary.json",summary)
    lines=["# Task 5B v1.1 summary","","Generic visual evidence packaging/accounting correction only; no retrieval or model call was rerun.","",f"- Old/corrected selected visual frame count: {old_count} / {new_count}","- Half-open frame filtering: `selected_start <= timestamp < selected_end`","- Dense frames are canonical when timestamps duplicate micro-window frames.","- Task 5A/Task 5B LLM calls in correction: 0 / 0","- Routing, intervals, candidate IDs/times, links, budgets, and post-hoc reference results unchanged.","","## Per case","", "| case | source micro | selected micro | dense | unique canonical | old |", "|---|---:|---:|---:|---:|---:|"]
    for r in records:
        a=r["visual_evidence_accounting"];lines.append(f'| {r["case_id"]} | {a["source_micro_frame_count"]} | {a["selected_micro_frame_count"]} | {a["dense_frame_count"]} | {a["unique_selected_visual_frame_count"]} | {a["old_task5b_selected_visual_frame_count"]} |')
    (OUT/"task5b_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");(OUT/"task5b_human_review.html").write_text(make_html(records,summary),encoding="utf-8")
    print(json.dumps(summary["visual_frame_accounting"],ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
