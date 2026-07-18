from __future__ import annotations

import argparse, gc, hashlib, html, json, math, os, sys, time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch, yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
from src.visual.micro_clips import build_micro_clips, normalize_rows, temporal_overlap  # noqa: E402


def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def save(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
def need(path: Path) -> Path:
    if not path.exists(): raise FileNotFoundError(str(path.resolve()))
    return path
def rel(path: Path) -> str: return path.relative_to(ROOT).as_posix()
def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def ranges_valid(rows: list[dict], n: int, window: int, stride: int) -> list[str]:
    warnings=[]
    covered=set()
    for i,row in enumerate(rows):
        if row["microclip_id"] != f"microclip_{i:04d}" or row["embedding_row_index"] != i: warnings.append("metadata_row_misalignment")
        covered.update(row["frame_indices"])
        if i and row["start_time"]-rows[i-1]["start_time"] != stride: warnings.append("incorrect_stride")
        if row["duration"] > window: warnings.append("window_exceeds_configured_duration")
    if covered != set(range(n)): warnings.append("incomplete_video_coverage")
    if len({r["microclip_id"] for r in rows}) != len(rows): warnings.append("duplicate_microclip_ids")
    return sorted(set(warnings))


def encode(cases: list[dict], cfg: dict, device: str) -> tuple[dict[str,np.ndarray],dict[str,float]]:
    try: import pkg_resources  # noqa
    except ModuleNotFoundError:
        import packaging,types
        shim=types.ModuleType("pkg_resources"); shim.packaging=packaging; sys.modules["pkg_resources"]=shim
    import clip
    model,_=clip.load(cfg["clip_model"],device=device,download_root=cfg["clip_cache"]); model.eval()
    vectors={}; timings={}
    with torch.inference_mode():
        for case in cases:
            t=time.perf_counter(); tokens=clip.tokenize([case["question"]]).to(device)
            v=model.encode_text(tokens).float(); v/=v.norm(dim=-1,keepdim=True)
            if device=="cuda": torch.cuda.synchronize()
            timings[case["case_id"]]=time.perf_counter()-t; vectors[case["case_id"]]=v.cpu().numpy()[0].astype(np.float32)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return vectors,timings


def rank(q: np.ndarray, emb: np.ndarray, rows: list[dict], k: int) -> tuple[list[dict],float]:
    t=time.perf_counter(); scores=normalize_rows(emb) @ (q/max(float(np.linalg.norm(q)),1e-12))
    order=np.argsort(-scores,kind="stable")[:k]
    out=[]
    for rank_no,i in enumerate(order,1): out.append({**rows[int(i)],"rank":rank_no,"similarity_score":float(scores[int(i)])})
    return out,time.perf_counter()-t


def distance(a:float,b:float,c:float,d:float)->float: return max(0.0,c-b,a-d)
def union_seconds(rows:list[dict])->float:
    intervals=sorted((float(x["start_time"]),float(x["end_time"])) for x in rows); total=0.; end=-math.inf
    for a,b in intervals:
        if a>end: total+=b-a
        elif b>end: total+=b-end
        end=max(end,b)
    return total
def diagnostic(rows:list[dict],start:float,end:float)->dict:
    ref_duration=max(end-start,1e-12); details=[]
    for r in rows:
        overlap=max(0.,min(float(r["end_time"]),end)-max(float(r["start_time"]),start)); duration=float(r["end_time"])-float(r["start_time"])
        details.append({"rank":r["rank"],"overlaps_weak_reference":overlap>0,"overlap_duration_sec":overlap,
            "candidate_duration_sec":duration,"temporal_distance_sec":distance(float(r["start_time"]),float(r["end_time"]),start,end),
            "broad_overlap_risk":overlap>0 and duration>=max(8.0,3*ref_duration)})
    durations=[x["candidate_duration_sec"] for x in details]
    return {"top1_overlap":bool(details and details[0]["overlaps_weak_reference"]),"top3_overlap":any(x["overlaps_weak_reference"] for x in details),
        "minimum_temporal_distance_sec":min((x["temporal_distance_sec"] for x in details),default=None),
        "closest_retrieved_rank":min(details,key=lambda x:(x["temporal_distance_sec"],x["rank"]))["rank"] if details else None,
        "mean_top3_candidate_duration_sec":float(np.mean(durations)) if durations else 0.,"maximum_top3_candidate_duration_sec":max(durations,default=0.),
        "total_unique_visual_seconds":union_seconds(rows),"estimated_dense_inspection_frames_1fps":len(set(i for r in rows for i in r.get("frame_indices",range(math.floor(r["start_time"]),math.ceil(r["end_time"]))))),"results":details}


def timeline(path:Path,duration:float,ref:tuple[float,float],coarse:list[dict],micro:list[dict])->None:
    w,h,m=1400,260,65; im=Image.new("RGB",(w,h),"white"); d=ImageDraw.Draw(im); scale=lambda x:m+int(x/max(duration,1)*(w-m-25))
    d.rectangle((scale(ref[0]),25,scale(ref[1]),220),fill="#fde2ef"); d.text((m,5),"Weak QA reference (diagnostic only; not used for ranking)",fill="black")
    for label,y,rows,color in (("Coarse",70,coarse,"#805ad5"),("Micro",155,micro,"#087f5b")):
        d.text((5,y+10),label,fill="black")
        for r in rows:
            d.rectangle((scale(r["start_time"]),y,scale(r["end_time"]),y+42),fill=color,outline="black"); d.text((scale(r["start_time"])+2,y+12),f'#{r["rank"]}',fill="white")
    d.line((m,225,w-25,225),fill="black"); d.text((m,230),"0s",fill="black"); d.text((w-80,230),f"{duration:.1f}s",fill="black"); im.save(path)


def index_visuals(out:Path,rows:list[dict],duration:float)->None:
    reps=out/"representative_frames"; reps.mkdir(exist_ok=True)
    for r in rows:
        source=ROOT/r["representative_frame_path"]; target=reps/f'{r["microclip_id"]}_t{r["representative_frame_timestamp"]:07.3f}.jpg'
        if not target.exists(): target.write_bytes(source.read_bytes())
        r["representative_frame_path"]=rel(target)
    cols,tw,th,lh=4,240,150,34; sheet=Image.new("RGB",(cols*tw,max(1,math.ceil(len(rows)/cols))*(th+lh)),"white"); draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(rows):
        x,y=(i%cols)*tw,(i//cols)*(th+lh); img=Image.open(ROOT/r["representative_frame_path"]).convert("RGB"); sheet.paste(ImageOps.fit(img,(tw,th)),(x,y)); draw.text((x+3,y+th+3),f'{r["microclip_id"]} {r["start_time"]:.0f}-{r["end_time"]:.0f}s m={r["motion_magnitude"]:.3f}',fill="black")
    sheet.save(out/"microclip_contact_sheet.jpg",quality=90)
    timeline(out/"microclip_timeline.png",duration,(0,0),[],[{**r,"rank":i+1} for i,r in enumerate(rows)])


def cards(report:Path,title:str,results:list[dict],diag:dict,micro:bool)->str:
    detail={x["rank"]:x for x in diag["results"]}; parts=[]
    for r in results:
        x=detail[r["rank"]]; imgs=r.get("frame_paths",[r.get("representative_keyframe_path")]) if micro else [r["representative_keyframe_path"]]
        strip="".join(f'<img src="{html.escape(Path(os.path.relpath(ROOT/p,report.parent)).as_posix())}">' for p in imgs)
        motion=f' · motion {r["motion_magnitude"]:.4f} (observation only)' if micro else ""
        parts.append(f'<article data-key="{html.escape(title)}-{r["rank"]}"><div class=strip>{strip}</div><p><b>#{r["rank"]} {r["start_time"]:.1f}–{r["end_time"]:.1f}s</b> · duration {x["candidate_duration_sec"]:.1f}s · score {r["similarity_score"]:.4f}<br>distance {x["temporal_distance_sec"]:.1f}s · overlap {x["overlap_duration_sec"]:.1f}s · broad risk {x["broad_overlap_risk"]}{motion}</p><select><option value="">Choose judgment…</option>{"".join(f"<option>{v}</option>" for v in ["Semantic hit","Partial hit","Temporal-only hit","Broad-region accidental hit","Miss","Unsure"])}</select><textarea placeholder="Optional notes"></textarea></article>')
    return f'<section><h3>{title}</h3>{"".join(parts)}</section>'


def review_html(path:Path,cases:list[dict],agg:dict)->None:
    blocks=[]
    for c in cases:
        timeline_src=Path(os.path.relpath(ROOT/c["comparison_timeline_path"],path.parent)).as_posix()
        blocks.append(f'<div class=case><h2>{c["case_id"]}</h2><p><b>Question:</b> {html.escape(c["question"])}</p><p class=warn>Weak reference {c["weak_reference_interval"]["start"]:.1f}–{c["weak_reference_interval"]["end"]:.1f}s; loaded only after rankings and not a precise gold boundary.</p><img class=timeline src="{timeline_src}"><div class=columns>{cards(path,"Coarse Top 3",c["coarse_results"],c["coarse_diagnostic"],False)}{cards(path,"Micro-clip Top 3",c["microclip_results"],c["microclip_diagnostic"],True)}</div><p>Unique candidate seconds: coarse {c["coarse_diagnostic"]["total_unique_visual_seconds"]:.1f}; micro {c["microclip_diagnostic"]["total_unique_visual_seconds"]:.1f}.</p></div>')
    css="body{font:14px system-ui;margin:25px;color:#183153}.warn{background:#fff3bf;padding:8px}.case{border-top:4px solid #829ab1;margin-top:30px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}article{border:1px solid #bcccdc;margin:8px 0;padding:8px;border-radius:7px}.strip{display:flex;overflow:auto}.strip img{height:105px;min-width:0;object-fit:cover}.timeline{width:100%;max-width:1400px}select,textarea{width:100%;margin-top:5px}textarea{height:42px}@media(max-width:800px){.columns{grid-template-columns:1fr}}"
    js="const K='task21-judgments';let s=JSON.parse(localStorage.getItem(K)||'{}');document.querySelectorAll('article').forEach(a=>{let k=a.dataset.key+'-'+a.closest('.case').querySelector('h2').textContent,q=a.querySelector('select'),n=a.querySelector('textarea');q.value=s[k]?.judgment||'';n.value=s[k]?.notes||'';[q,n].forEach(x=>x.oninput=()=>{s[k]={judgment:q.value,notes:n.value};localStorage.setItem(K,JSON.stringify(s))})});function exp(){let b=new Blob([JSON.stringify(s,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='task21_judgments.json';a.click()}"
    path.write_text(f'<!doctype html><meta charset=utf-8><title>Task 2.1 review</title><style>{css}</style><h1>Coarse vs visual micro-clip human review</h1><p>Temporal diagnostics do not establish semantic correctness. Human review is required.</p><button onclick=exp()>Export judgments JSON</button><p>Coarse Top-1/Top-3: {agg["coarse_top1_overlap_count"]}/{agg["coarse_top3_overlap_count"]}; micro: {agg["micro_top1_overlap_count"]}/{agg["micro_top3_overlap_count"]}.</p>{"".join(blocks)}<script>{js}</script>',encoding="utf-8")


def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/visual_microclips.yaml"); args=ap.parse_args(); cfg=yaml.safe_load(need(ROOT/args.config).read_text())
    # Required outputs are new; frozen trees are hashed before and checked after.
    frozen=[ROOT/"outputs/visual_index",ROOT/"outputs/audio_index",ROOT/"outputs/retrieval"]
    frozen_files=[p for d in frozen for p in d.rglob("*") if p.is_file()]; frozen_state={str(p): (p.stat().st_size,p.stat().st_mtime_ns) for p in frozen_files}
    cases=load(need(ROOT/cfg["cases"])); index_root=ROOT/"outputs/visual_micro_index"; retrieval_root=ROOT/"outputs/visual_micro_retrieval"; index_root.mkdir(exist_ok=True); retrieval_root.mkdir(exist_ok=True)
    sources={}; dimensions=set(); index_summaries=[]
    for case in cases:
        folder=need(ROOT/"outputs/visual_index"/case["video_id"]); frames_dir=need(folder/"frames_1fps"); ep=need(folder/"frame_embeddings.npy"); jp=need(folder/"visual_state_regions.json")
        frames=sorted(frames_dir.glob("frame_*.jpg"));
        if not frames: raise FileNotFoundError(str((frames_dir/"frame_*.jpg").resolve()))
        emb=np.load(ep); dimensions.add(emb.shape[1]);
        if len(frames)!=len(emb): raise ValueError(f"{folder.resolve()}: {len(frames)} frames != {len(emb)} embeddings")
        sources[case["video_id"]]=(folder,frames,emb,load(jp))
    if len(dimensions)!=1: raise ValueError(f"inconsistent embedding dimensions: {dimensions}")
    index_total_start=time.perf_counter()
    for case in cases:
        t=time.perf_counter(); folder,frames,emb,_=sources[case["video_id"]]; out=index_root/case["video_id"]; out.mkdir(exist_ok=True)
        rows,pooled=build_micro_clips(emb,frames,float(case["video_duration"]),int(cfg["window_sec"]),int(cfg["stride_sec"]))
        for i,r in enumerate(rows):
            r["microclip_id"]=r.pop("micro_clip_id").replace("micro_clip_", "microclip_"); r["embedding_row_index"]=r.pop("embedding_index"); r["frame_paths"]=[rel(Path(p)) for p in r["frame_paths"]]; r["representative_frame_path"]=rel(Path(r["representative_frame_path"]))
        warnings=ranges_valid(rows,len(frames),int(cfg["window_sec"]),int(cfg["stride_sec"]))
        if pooled.dtype!=np.float32: warnings.append("embeddings_not_float32")
        if not np.isfinite(pooled).all(): warnings.append("embeddings_contain_nan_or_inf")
        index_visuals(out,rows,float(case["video_duration"])); np.save(out/"microclip_embeddings.npy",pooled)
        elapsed=time.perf_counter()-t; payload={"video_id":case["video_id"],"construction_inputs":["frozen_frame_embeddings","frozen_1fps_frames","video_duration"],"excluded_inputs":["question","answer","answer_options","annotation_context","dataset_reference_interval","human_review_notes"],"window_sec":cfg["window_sec"],"stride_sec":cfg["stride_sec"],"embedding_model":"OpenAI CLIP ViT-B/32","embedding_dtype":str(pooled.dtype),"embedding_dimension":pooled.shape[1],"indexing_time_sec":elapsed,"warnings":warnings,"microclips":rows}; save(out/"microclip_index.json",payload)
        (out/"microclip_index_report.md").write_text(f'# Micro-clip index — {case["video_id"]}\n\n- Micro-clips: {len(rows)}\n- Frames: {len(frames)}\n- Embedding shape: `{pooled.shape}` / `{pooled.dtype}`\n- Complete coverage: `{not warnings}`\n- Indexing time: {elapsed:.6f}s\n- Warnings: {", ".join(warnings) or "None"}\n',encoding="utf-8")
        index_summaries.append({"video_id":case["video_id"],"microclip_count":len(rows),"frame_count":len(frames),"indexing_time_sec":elapsed,"disk_usage_bytes":sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),"warnings":warnings})
    indexing_total=time.perf_counter()-index_total_start
    index_summary={"task":"Task 2.1 fine visual micro-clip index","frozen_assets_only":True,"microclip_construction_question_independent":True,"total_indexing_time_sec":indexing_total,"videos":index_summaries}; save(index_root/"visual_micro_index_summary.json",index_summary)
    (index_root/"visual_micro_index_summary.md").write_text("# Visual micro-index summary\n\n"+"\n".join(f'- {x["video_id"]}: {x["microclip_count"]} clips, {x["indexing_time_sec"]:.4f}s, warnings: {x["warnings"] or "none"}' for x in index_summaries)+f"\n\nTotal indexing time: {indexing_total:.6f}s\n",encoding="utf-8")
    device="cuda" if cfg["device"]=="cuda" and torch.cuda.is_available() else "cpu"; queries,qtime=encode(cases,cfg,device); ranked=[]
    for case in cases:
        case_out=retrieval_root/case["case_id"]; case_out.mkdir(exist_ok=True); idx=load(index_root/case["video_id"]/"microclip_index.json"); emb=np.load(index_root/case["video_id"]/"microclip_embeddings.npy")
        micro,rtime=rank(queries[case["case_id"]],emb,idx["microclips"],int(cfg["top_k"]))
        task4=load(need(ROOT/"outputs/retrieval"/case["case_id"]/"visual_retrieval.json")); coarse=task4["results"][:3]
        # Restore exact coarse frame index lists from frozen metadata for inspection accounting.
        cmap={r["region_id"]:r for r in sources[case["video_id"]][3]["visual_state_regions"]}
        for r in coarse:
            meta=cmap[r["visual_region_id"]]; r["region_id"]=r["visual_region_id"]; r["frame_indices"]=meta["frame_indices"]
        ranked.append({"case_id":case["case_id"],"video_id":case["video_id"],"question":case["question"],"coarse_results":coarse,"microclip_results":micro,"query_encoding_time_sec":qtime[case["case_id"]],"microclip_ranking_time_sec":rtime})
        save(case_out/"microclip_visual_retrieval.json",{"case_id":case["case_id"],"raw_question":case["question"],"query_encoder":"OpenAI CLIP ViT-B/32","timestamp_parsed_from_question":False,"reference_loaded_before_ranking":False,"motion_used_for_ranking":False,"results":micro,"query_encoding_time_sec":qtime[case["case_id"]],"ranking_time_sec":rtime})
    refs={c["case_id"]:c for c in cases}
    for r in ranked:
        case=refs[r["case_id"]]; a,b=float(case["provided_timestamp_start"]),float(case["provided_timestamp_end"]); r["weak_reference_interval"]={"start":a,"end":b,"not_precise_gold_boundary":True,"used_for_ranking":False}; r["coarse_diagnostic"]=diagnostic(r["coarse_results"],a,b); r["microclip_diagnostic"]=diagnostic(r["microclip_results"],a,b)
        out=retrieval_root/r["case_id"]; timeline(out/"comparison_timeline.png",float(case["video_duration"]),(a,b),r["coarse_results"],r["microclip_results"]); r["comparison_timeline_path"]=rel(out/"comparison_timeline.png"); save(out/"coarse_vs_microclip_comparison.json",r)
        (out/"comparison_report.md").write_text(f'# Coarse vs micro-clip — {r["case_id"]}\n\nThe weak reference ({a:.1f}–{b:.1f}s) was loaded after ranking and is not a precise gold boundary. Temporal overlap is not semantic correctness.\n\n- Coarse Top-1 / Top-3 overlap: {r["coarse_diagnostic"]["top1_overlap"]} / {r["coarse_diagnostic"]["top3_overlap"]}\n- Micro Top-1 / Top-3 overlap: {r["microclip_diagnostic"]["top1_overlap"]} / {r["microclip_diagnostic"]["top3_overlap"]}\n- Unique Top-3 seconds: coarse {r["coarse_diagnostic"]["total_unique_visual_seconds"]:.2f}; micro {r["microclip_diagnostic"]["total_unique_visual_seconds"]:.2f}\n',encoding="utf-8")
    agg={"case_count":len(ranked),"coarse_top1_overlap_count":sum(r["coarse_diagnostic"]["top1_overlap"] for r in ranked),"coarse_top3_overlap_count":sum(r["coarse_diagnostic"]["top3_overlap"] for r in ranked),"micro_top1_overlap_count":sum(r["microclip_diagnostic"]["top1_overlap"] for r in ranked),"micro_top3_overlap_count":sum(r["microclip_diagnostic"]["top3_overlap"] for r in ranked)}
    for method,key in (("coarse","coarse_diagnostic"),("microclip","microclip_diagnostic")):
        agg[f"{method}_average_candidate_duration_sec"]=float(np.mean([d["candidate_duration_sec"] for r in ranked for d in r[key]["results"]])); agg[f"{method}_average_unique_top3_visual_seconds"]=float(np.mean([r[key]["total_unique_visual_seconds"] for r in ranked])); agg[f"{method}_estimated_dense_inspection_frames"]=sum(r[key]["estimated_dense_inspection_frames_1fps"] for r in ranked)
    warnings=["weak_reference_is_not_a_precise_gold_boundary","temporal_overlap_does_not_establish_semantic_correctness"]
    for cid in ("00002_7","00018_1","00006_3"): warnings.append(f"{cid}:known_excessively_broad_frozen_coarse_regions_review_broad_overlap_risk")
    summary={"task":"Task 2.1 coarse versus visual micro-clip comparison","aggregate":agg,"indexing_time_sec":indexing_total,"average_query_encoding_time_sec":float(np.mean(list(qtime.values()))),"average_microclip_ranking_time_sec":float(np.mean([r["microclip_ranking_time_sec"] for r in ranked])),"index_disk_usage_bytes":sum(p.stat().st_size for p in index_root.rglob("*") if p.is_file()),"retrieval_disk_usage_bytes_before_summary":sum(p.stat().st_size for p in retrieval_root.rglob("*") if p.is_file()),"warnings":warnings,"cases":ranked}; save(retrieval_root/"comparison_summary.json",summary)
    reduction=100*(1-agg["microclip_average_candidate_duration_sec"]/agg["coarse_average_candidate_duration_sec"])
    (retrieval_root/"comparison_summary.md").write_text(f'# Task 2.1 comparison summary\n\nTemporal overlap against the weak dataset interval is diagnostic only; semantic correctness requires human review.\n\n- Coarse Top-1 / Top-3 overlap: {agg["coarse_top1_overlap_count"]}/{len(ranked)}; {agg["coarse_top3_overlap_count"]}/{len(ranked)}\n- Micro Top-1 / Top-3 overlap: {agg["micro_top1_overlap_count"]}/{len(ranked)}; {agg["micro_top3_overlap_count"]}/{len(ranked)}\n- Mean candidate duration: coarse {agg["coarse_average_candidate_duration_sec"]:.3f}s; micro {agg["microclip_average_candidate_duration_sec"]:.3f}s ({reduction:.1f}% reduction)\n- Average unique Top-3 seconds: coarse {agg["coarse_average_unique_top3_visual_seconds"]:.3f}; micro {agg["microclip_average_unique_top3_visual_seconds"]:.3f}\n- Estimated dense-inspection frames: coarse {agg["coarse_estimated_dense_inspection_frames"]}; micro {agg["microclip_estimated_dense_inspection_frames"]}\n- Indexing time: {indexing_total:.6f}s\n- Warnings: {", ".join(warnings)}\n',encoding="utf-8")
    review_html(retrieval_root/"coarse_vs_microclip_human_review.html",ranked,agg)
    after={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for d in frozen for p in d.rglob("*") if p.is_file()}
    if after!=frozen_state: raise RuntimeError("frozen output integrity check failed: frozen files changed")
    print(json.dumps({"aggregate":agg,"indexing_time_sec":indexing_total,"warnings":warnings},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
