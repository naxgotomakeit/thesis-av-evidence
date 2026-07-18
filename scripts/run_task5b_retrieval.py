from __future__ import annotations

import argparse, copy, gc, html, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.retrieval.task5b import (apply_budget, clip_candidate, interval_distance, link_candidates, load_plans_immutable, match_quoted_phrase, modalities_to_execute, overlap_seconds, resolve_question_intervals, stable_id, union_duration)  # noqa:E402

OUT=ROOT/"outputs/planner_guided_retrieval"
PLAN_PATH=ROOT/"outputs/question_planner/v2/task5a_plans.jsonl"
MANIFEST_PATH=ROOT/"data/manifests/mvp_cases_6.json"
MEDIA_ROOT=Path("D:/ThesisData/EgoSound/data/Ego4d/videos")
FFMPEG=Path("C:/Users/72977/miniforge3/envs/roboenv2/Library/bin/ffmpeg.exe")

def load(path:Path)->Any: return json.loads(path.read_text(encoding="utf-8"))
def save(path:Path,obj:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def rel(path:Path)->str:
    try:return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:return path.as_posix()
def intervals_tuple(items:list[dict])->list[tuple[float,float]]: return [(float(x["start_sec"]),float(x["end_sec"])) for x in items]
def nearest_distance(row:dict,intervals:list[dict])->float: return min((interval_distance(float(row["start_time"]),float(row["end_time"]),float(x["start_sec"]),float(x["end_sec"])) for x in intervals),default=0.0)
def score_map(results:list[dict],id_key:str)->dict[str,dict]: return {str(x[id_key]):x for x in results}

def extract_dense_frames(video:Path,out:Path,start:float,end:float)->tuple[list[dict],str|None]:
    out.mkdir(parents=True,exist_ok=True); pattern=out/"dense_%03d.jpg"
    command=[str(FFMPEG),"-y","-hide_banner","-loglevel","error","-ss",f"{start:.3f}","-t",f"{max(0,end-start):.3f}","-i",str(video),"-vf","fps=2","-q:v","2","-start_number","0",str(pattern)]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=120,check=False)
        if result.returncode!=0:return [],"local_ffmpeg_failed: "+result.stderr.strip()[:300]
        frames=[]
        for i,path in enumerate(sorted(out.glob("dense_*.jpg"))): frames.append({"dense_frame_id":f"dense_{i:03d}","timestamp":round(start+i*.5,3),"frame_path":rel(path),"source_mp4_path":str(video)})
        return frames,None
    except Exception as exc:return [],f"local_video_read_error: {type(exc).__name__}: {str(exc)[:250]}"

def encode_acoustic_questions(plan_rows:list[dict])->tuple[dict[str,np.ndarray],dict[str,float],dict[str,Any]]:
    targets=[x for x in plan_rows if "acoustic" in x["plan"]["resolver_modalities"] or x["plan"]["primary_anchor_modality"]=="acoustic"]
    if not targets:return {},{}, {"model_load_time_sec":0.0,"device":"not_used","model":"not_used"}
    import torch
    from transformers import ClapModel,ClapProcessor
    sample=load(ROOT/"outputs/audio_index"/targets[0]["case_id"].split("_")[0]/"acoustic_embedding_index.json")
    name,revision=sample["model"],sample.get("model_revision");device="cuda" if torch.cuda.is_available() else "cpu";load_start=time.perf_counter()
    processor=ClapProcessor.from_pretrained(name,revision=revision,local_files_only=True);model=ClapModel.from_pretrained(name,revision=revision,use_safetensors=True,local_files_only=True).to(device).eval();load_time=time.perf_counter()-load_start
    vectors={};timings={}
    with torch.inference_mode():
        for row in targets:
            started=time.perf_counter();inputs=processor(text=[row["raw_question"]],return_tensors="pt",padding=True);output=model.get_text_features(**{k:v.to(device) for k,v in inputs.items()});vector=output.pooler_output if hasattr(output,"pooler_output") else output;vector=vector.float();vector/=vector.norm(dim=-1,keepdim=True)
            if device=="cuda":torch.cuda.synchronize()
            timings[row["case_id"]]=time.perf_counter()-started;vectors[row["case_id"]]=vector.cpu().numpy()[0].astype(np.float32)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return vectors,timings,{"model_load_time_sec":load_time,"device":device,"model":name,"revision":revision}

def build_case(row:dict,manifest_safe:dict,acoustic_query:np.ndarray|None=None,acoustic_query_time:float=0.0,*,fresh_score_maps:dict[str,dict[str,dict]]|None=None,allow_historical_score_files:bool=True,source_video_path:Path|None=None,local_visual_output_root:Path|None=None)->dict:
    started=time.perf_counter(); case_id=row["case_id"]; question=row["raw_question"]; plan=copy.deepcopy(row["plan"]); cues=copy.deepcopy(row["deterministic_cues"]); video_id=manifest_safe["video_id"]; duration=float(manifest_safe["video_duration"])
    visual_dir=ROOT/"outputs/visual_index"/video_id; micro_dir=ROOT/"outputs/visual_micro_index"/video_id; audio_dir=ROOT/"outputs/audio_index"/video_id; baseline_dir=ROOT/"outputs/retrieval"/case_id
    video_source=source_video_path or MEDIA_ROOT/f"{video_id}.mp4"
    files={"task5a_v2":rel(PLAN_PATH),"coarse_visual_index":rel(visual_dir/"visual_state_regions.json"),"coarse_visual_embeddings":rel(visual_dir/"region_embeddings.npy"),"speech_transcripts":rel(audio_dir/"transcripts.json"),"speech_embedding_index":rel(audio_dir/"transcript_embedding_index.json"),"acoustic_index":rel(audio_dir/"acoustic_embedding_index.json"),"acoustic_embeddings":rel(audio_dir/"acoustic_embeddings.npy"),"visual_micro_index":rel(micro_dir/"microclip_index.json"),"task4_visual_scores":rel(baseline_dir/"visual_retrieval.json"),"task4_speech_scores":rel(baseline_dir/"speech_retrieval.json"),"task4_acoustic_scores":rel(baseline_dir/"acoustic_retrieval.json"),"source_mp4":str(video_source)}
    fresh_score_maps=fresh_score_maps or {}
    score_keys={"task4_visual_scores","task4_speech_scores","task4_acoustic_scores"}
    missing=[path for key,value in files.items() if key not in {"task5a_v2","source_mp4"} and not (key in score_keys and not allow_historical_score_files) and not (ROOT/value).is_file()]
    warnings=["missing_input: "+x for x in missing]
    anchor=resolve_question_intervals(question,cues,plan["temporal_relation"],duration); trace=list(anchor.pop("routing_trace")); methods=[]
    if anchor["raw_question_intervals"]: methods.append("explicit_question_time_hard_constraint")
    if anchor["deterministic_interpretations"]: methods.append("relative_start_first_5_seconds")
    transcripts=load(audio_dir/"transcripts.json").get("segments",[]) if (audio_dir/"transcripts.json").is_file() else []
    hard=intervals_tuple(anchor["raw_question_intervals"]); phrase_matches=[]
    for quote in cues.get("quoted_phrases",[]): phrase_matches.extend(match_quoted_phrase(quote["text"],transcripts,hard[0] if hard else None))
    if cues.get("quoted_phrases") and not phrase_matches and not hard and plan["primary_anchor_modality"]=="speech":
        for quote in cues["quoted_phrases"]: phrase_matches.extend(match_quoted_phrase(quote["text"],transcripts,None))
        trace.append({"event":"speech_anchor_fallback","reason":"no phrase match in constrained scope; plan requires speech anchor","expanded_outside_hard_interval":False})
    if phrase_matches:
        methods.append("quoted_speech_transcript_match")
        if not anchor["final_search_intervals"]:
            best=phrase_matches[0]; before,after=(.5,2.0) if anchor["executed_relation"] in {"after","sequence"} else (1.0,1.0)
            anchor["expanded_intervals"].append({"original_start_sec":best["start_time"],"original_end_sec":best["end_time"],"expand_before_sec":before,"expand_after_sec":after,"reason":"quoted speech anchor context","start_sec":max(0,best["start_time"]-before),"end_sec":min(duration,best["end_time"]+after)})
            anchor["final_search_intervals"]=[{"start_sec":max(0,best["start_time"]-before),"end_sec":min(duration,best["end_time"]+after),"source":"matched_transcript_anchor"}]
    if not anchor["final_search_intervals"]:
        anchor["final_search_intervals"]=[{"start_sec":0.0,"end_sec":duration,"source":"global_fallback_no_stronger_anchor"}]; methods.append("global_fallback")
        warnings.append("no_stronger_anchor_global_interval_used")
    search=anchor["final_search_intervals"]
    executed,skipped=modalities_to_execute(plan); reasons={m:("required by Task 5A resolver_modalities" if m in plan["resolver_modalities"] else "recorded fallback") for m in executed}
    trace.extend([{"event":"modality_executed","modality":m,"reason":reasons[m]} for m in executed]); trace.extend([{"event":"modality_skipped",**x} for x in skipped])
    runtime={"task5b_llm_api_calls":0,"visual_index_calls":0,"speech_index_calls":0,"acoustic_index_calls":0,"local_video_reads":0,"latency_sec":0.0,"branch_latency_sec":{"visual":0.0,"speech":0.0,"acoustic":0.0},"branch_local_model_calls":{"visual":0,"speech":0,"acoustic":0},"local_visual_refinement_sec":None,"temporal_linking_sec":0.0,"budget_selection_sec":0.0,"new_embedding_computations":0,"new_embedding_computation_time_sec":0.0,"reused_embedding_files":[],"reused_retrieval_score_files":[],"reused_index_metadata_files":[]}
    speech=[]; acoustic=[]; coarse=[]; micro=[]; dense=[]; acoustic_anchors=[]; visual_anchors=[]
    if "speech" in executed:
        t=time.perf_counter(); runtime["speech_index_calls"]+=1; speech_score={}
        if "speech" in fresh_score_maps: speech_score=fresh_score_maps["speech"]
        elif allow_historical_score_files and (baseline_dir/"speech_retrieval.json").is_file(): speech_score=score_map(load(baseline_dir/"speech_retrieval.json")["results"],"transcript_segment_id");runtime["reused_retrieval_score_files"].append(files["task4_speech_scores"])
        pool=[]
        if plan["answer_requirement"]["operation"]=="count_occurrences":
            pool=phrase_matches
        elif plan["answer_requirement"]["operation"]=="measure_delay":
            triggers=phrase_matches
            for match in triggers:
                base=next((x for x in transcripts if x["transcript_segment_id"]==match["transcript_segment_id"]),None)
                if base: pool.append({**base,"phrase_match":match,"sequence_role":"trigger","candidate_delay_sec":0.0})
                for segment in transcripts:
                    delay=float(segment["start_time"])-match["end_time"]
                    if 0<=delay<=2.0: pool.append({**segment,"sequence_role":"plausible_response","candidate_delay_sec":round(delay,3),"trigger_segment_id":match["transcript_segment_id"]})
        else:
            for segment in transcripts:
                clipped=clip_candidate(segment,search)
                if clipped: pool.append(clipped)
            for match in phrase_matches:
                if not any((x.get("transcript_segment_id")==match["transcript_segment_id"]) for x in pool): pool.append(next(x for x in transcripts if x["transcript_segment_id"]==match["transcript_segment_id"]))
        seen=set()
        for item in pool:
            sid=item.get("transcript_segment_id");
            if sid in seen and plan["answer_requirement"]["operation"]!="measure_delay":continue
            seen.add(sid); match=next((x for x in phrase_matches if x["transcript_segment_id"]==sid),None); score=speech_score.get(sid,{}).get("similarity_score")
            speech.append({"candidate_id":stable_id("speech",video_id,sid,item.get("sequence_role","segment")),"candidate_type":"speech_segment","modality":"speech","transcript_segment_id":sid,"transcript_text":item.get("text",item.get("matched_transcript_text","")),"start_time":float(item["start_time"]),"end_time":float(item["end_time"]),"similarity_score":score,"phrase_match_score":match.get("match_score") if match else None,"phrase_match_method":match.get("match_method") if match else None,"temporal_relation_to_anchor":anchor["executed_relation"],"anchor_distance_sec":nearest_distance(item,search),"hard_timestamp_consistent":bool(clip_candidate(item,search)),"sequence_role":item.get("sequence_role"),"candidate_delay_sec":item.get("candidate_delay_sec"),"speaker_attribution_warning":"Whisper transcript index does not establish speaker identity.","source_index":files["speech_transcripts"],"warnings":[]})
        runtime["branch_latency_sec"]["speech"]=time.perf_counter()-t
        if plan["answer_requirement"]["operation"]=="count_occurrences" and not speech: warnings.append("no_plausible_ASR_phrase_occurrence_in_requested_interval")
    if "acoustic" in executed:
        t=time.perf_counter(); runtime["acoustic_index_calls"]+=1; rows=load(audio_dir/"acoustic_embedding_index.json").get("rows",[]) if (audio_dir/"acoustic_embedding_index.json").is_file() else []; amap={}
        if "acoustic" in fresh_score_maps:
            amap=fresh_score_maps["acoustic"]
        elif acoustic_query is not None:
            matrix=np.load(audio_dir/"acoustic_embeddings.npy");matrix=matrix/np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-12);scores=matrix@acoustic_query;amap={source["acoustic_region_id"]:{"similarity_score":float(scores[i])} for i,source in enumerate(rows)};runtime["new_embedding_computations"]+=1;runtime["new_embedding_computation_time_sec"]+=acoustic_query_time;runtime["branch_local_model_calls"]["acoustic"]+=1;runtime["reused_embedding_files"].append(files["acoustic_embeddings"])
        elif allow_historical_score_files and (baseline_dir/"acoustic_retrieval.json").is_file(): amap=score_map(load(baseline_dir/"acoustic_retrieval.json")["results"],"acoustic_region_id");runtime["reused_retrieval_score_files"].append(files["task4_acoustic_scores"])
        runtime["reused_index_metadata_files"].append(files["acoustic_index"])
        for source in rows:
            item=clip_candidate(source,search)
            if not item:continue
            aid=source["acoustic_region_id"]; score=amap.get(aid,{}).get("similarity_score"); broad=float(source["end_time"])-float(source["start_time"])>12
            acoustic.append({"candidate_id":stable_id("acoustic",video_id,aid),"candidate_type":"acoustic_region","modality":"acoustic","acoustic_region_id":aid,"start_time":item["start_time"],"end_time":item["end_time"],"source_start_time":source["start_time"],"source_end_time":source["end_time"],"similarity_score":score,"temporal_overlap_sec":item["hard_interval_overlap_sec"],"anchor_distance_sec":0.0,"hard_timestamp_consistent":True,"local_audio_clip_reference":None,"source_index":files["acoustic_index"],"warnings":(["broad_source_acoustic_region_clipped_to_local_search_interval"] if broad else [])+(["low_information_or_silence"] if source.get("low_information_or_silence") else [])})
        runtime["branch_latency_sec"]["acoustic"]=time.perf_counter()-t
    # Acoustic/visual anchor retrieval is only needed when no stronger question-time or speech anchor exists.
    if plan["primary_anchor_modality"]=="acoustic" and not hard and not phrase_matches: acoustic_anchors=copy.deepcopy(acoustic[:3]); methods.append("acoustic_anchor_retrieval")
    local_required=bool(plan["requires_local_visual_inspection"])
    local={"executed":False,"reason":"not required by Task 5A and no fallback triggered","coarse_regions_used":[],"micro_windows":[],"dense_frames":[],"original_video_duration_read_sec":0.0}
    if "visual" in executed:
        t=time.perf_counter(); runtime["visual_index_calls"]+=1; vdata=load(visual_dir/"visual_state_regions.json"); regions=vdata["visual_state_regions"]
        for source in regions:
            item=clip_candidate(source,search)
            if not item:continue
            visual_score=fresh_score_maps.get("visual",{}).get(str(source["region_id"]),{})
            coarse.append({"candidate_id":stable_id("coarse",video_id,source["region_id"]),"candidate_type":"coarse_visual_region","modality":"visual","region_id":source["region_id"],"start_time":item["start_time"],"end_time":item["end_time"],"source_start_time":source["start_time"],"source_end_time":source["end_time"],"representative_keyframe_path":rel(visual_dir/source["representative_keyframe_path"]),"similarity_score":visual_score.get("similarity_score"),"query_rank":visual_score.get("rank"),"anchor_distance_sec":0.0,"hard_timestamp_consistent":True,"source_index":files["coarse_visual_index"],"warnings":["coarse_region_is_localization_context_not_complete_evidence"]})
        if not coarse:
            nearest=min(regions,key=lambda x:nearest_distance(x,search)); coarse.append({"candidate_id":stable_id("coarse",video_id,nearest["region_id"]),"candidate_type":"coarse_visual_region","modality":"visual","region_id":nearest["region_id"],"start_time":nearest["start_time"],"end_time":nearest["end_time"],"source_start_time":nearest["start_time"],"source_end_time":nearest["end_time"],"representative_keyframe_path":rel(visual_dir/nearest["representative_keyframe_path"]),"anchor_distance_sec":nearest_distance(nearest,search),"hard_timestamp_consistent":False,"source_index":files["coarse_visual_index"],"warnings":["nearest_coarse_region_used_no_overlap"]})
        if plan["primary_anchor_modality"]=="visual" or (not hard and not phrase_matches and plan["visual_route"] in {"global_visual_fallback","coarse_event_retrieval"}): visual_anchors=copy.deepcopy(sorted(coarse,key=lambda x:(x.get("query_rank") if x.get("query_rank") is not None else 10**9,x["start_time"],x["region_id"]))[:3]); methods.append("coarse_visual_anchor_retrieval")
        if local_required:
            refinement_started=time.perf_counter()
            local.update({"executed":True,"reason":"Task 5A requires_local_visual_inspection=true; coarse-to-fine route executed","coarse_regions_used":[{"region_id":x["region_id"],"keyframe":x["representative_keyframe_path"]} for x in coarse]})
            mdata=load(micro_dir/"microclip_index.json"); runtime["reused_index_metadata_files"].append(files["visual_micro_index"]); candidates=[]
            for source in mdata["microclips"]:
                item=clip_candidate(source,search)
                if item:candidates.append((nearest_distance(source,search),source,item))
            candidates.sort(key=lambda x:(x[0],x[1]["start_time"],x[1]["microclip_id"])); candidates=candidates[:2]
            for _,source,item in candidates:
                provenance=[x["region_id"] for x in coarse if overlap_seconds(x["source_start_time"],x["source_end_time"],source["start_time"],source["end_time"])>0]
                micro.append({"candidate_id":stable_id("micro",video_id,source["microclip_id"]),"candidate_type":"micro_window","modality":"visual","microclip_id":source["microclip_id"],"start_time":item["start_time"],"end_time":item["end_time"],"source_start_time":source["start_time"],"source_end_time":source["end_time"],"frame_paths":source["frame_paths"],"representative_frame_path":source["representative_frame_path"],"motion_magnitude":source["motion_magnitude"],"coarse_region_provenance":provenance,"anchor_distance_sec":0.0,"hard_timestamp_consistent":True,"source_index":files["visual_micro_index"],"warnings":["micro_window_is_detailed_candidate_not_event_label"]})
            local["micro_windows"]=copy.deepcopy(micro)
            local_start=min(x["start_sec"] for x in search); local_end=max(x["end_sec"] for x in search); video=video_source
            if video.is_file() and FFMPEG.is_file():
                dense,error=extract_dense_frames(video,(local_visual_output_root or OUT/"local_visual")/case_id,local_start,local_end); runtime["local_video_reads"]+=1; local["original_video_duration_read_sec"]=local_end-local_start
                if error:warnings.append(error)
            else:warnings.append("source_mp4_or_ffmpeg_missing_local_dense_refinement_failed")
            local["dense_frames"]=dense
            runtime["local_visual_refinement_sec"]=time.perf_counter()-refinement_started
        runtime["branch_latency_sec"]["visual"]=time.perf_counter()-t
    all_candidates=speech+acoustic+micro
    linking_started=time.perf_counter()
    linked=link_candidates(all_candidates,anchor["executed_relation"])
    runtime["temporal_linking_sec"]=time.perf_counter()-linking_started
    budget_started=time.perf_counter()
    selected,budget=apply_budget(all_candidates,plan["answer_requirement"]["operation"])
    runtime["budget_selection_sec"]=time.perf_counter()-budget_started
    selected_duration=union_duration(selected); selected_frames=len({p for x in selected if x["modality"]=="visual" for p in x.get("frame_paths",[])})+len(dense if any(x["modality"]=="visual" for x in selected) else [])
    selected_clips=sum(x.get("candidate_type")=="micro_window" for x in selected)
    anchor.update({"methods":methods,"transcript_matches":phrase_matches,"acoustic_anchor_candidates":acoustic_anchors,"visual_anchor_candidates":visual_anchors,"confidence":1.0 if anchor["raw_question_intervals"] else (.85 if anchor["deterministic_interpretations"] or phrase_matches else .4),"warnings":[]})
    runtime["latency_sec"]=time.perf_counter()-started
    record={"case_id":case_id,"question":question,"task5a_plan_used":plan,"input_files":files,"anchor_resolution":anchor,"executed_modalities":[{"modality":m,"reason":reasons[m]} for m in executed],"skipped_modalities":skipped,"speech_candidates":speech,"acoustic_candidates":acoustic,"coarse_visual_candidates":coarse,"local_visual_refinement":local,"linked_windows":linked,"all_candidates":all_candidates,"selected_candidates":selected,"budget":budget,"selected_duration_sec":selected_duration,"selected_visual_frame_count":selected_frames,"selected_visual_clip_count":selected_clips,"routing_trace":trace,"warnings":warnings,"runtime":runtime,"future_final_qa_cost":{"final_model_api_calls":None,"final_input_tokens":None,"final_output_tokens":None,"final_visual_frames":None,"final_latency_sec":None,"final_estimated_cost":None,"label":"future Task value; not executed"}}
    record["codex_diagnostic"]=diagnose(record)
    return record

def diagnose(r:dict)->dict:
    required=set(r["task5a_plan_used"]["resolver_modalities"]); executed={x["modality"] for x in r["executed_modalities"]}; hard=bool(r["anchor_resolution"]["raw_question_intervals"]); hard_ok=all(x.get("hard_timestamp_consistent",True) for x in r["selected_candidates"]); local_req=r["task5a_plan_used"]["requires_local_visual_inspection"]; local_ok=r["local_visual_refinement"]["executed"] if local_req else True; op=r["task5a_plan_used"]["answer_requirement"]["operation"]
    exception=op in {"count_occurrences","measure_delay"}; budget_ok=exception or r["selected_duration_sec"]<=12.000001
    problems=[]
    if not required<=executed:problems.append("required modality branch missing")
    if hard and not hard_ok:problems.append("selected evidence violates hard question-time constraint")
    if local_req and not local_ok:problems.append("required local visual refinement failed")
    if any(x.startswith("missing_input") or "failed" in x for x in r["warnings"]):problems.append("an index/media branch warning may reduce evidence")
    if not budget_ok:problems.append("selected evidence exceeds operation budget")
    status="complete" if not problems else ("failed" if not required<=executed else "partial")
    return {"execution_status":status,"routing_followed":required<=executed,"required_modalities_executed":required<=executed,"hard_temporal_constraints_respected":hard_ok,"local_visual_refinement_status":"executed" if local_req and local_ok else ("failed" if local_req else "not_required"),"budget_status":"operation_exception" if exception else ("within_budget" if budget_ok else "exceeded"),"possible_problem":"; ".join(problems) or "none","likely_functional_impact":"none" if not problems else ("blocking" if status=="failed" else "may_reduce_recall"),"recommended_human_check":"Review transcript ambiguity, coarse-to-micro provenance, broad regions, and whether selected candidates are semantically sufficient; do not infer an answer from temporal overlap alone."}

def offline_cost()->dict:
    visual=load(ROOT/"outputs/visual_index/visual_index_summary.json"); audio=load(ROOT/"outputs/audio_index/audio_index_summary.json"); emb=load(ROOT/"outputs/audio_index/audio_embedding_summary.json"); micro=load(ROOT/"outputs/visual_micro_index/visual_micro_index_summary.json")
    frames=sum(x["extracted_frames"] for x in visual["videos"]); acoustic=sum(x["acoustic_region_count"] for x in audio["cases"]); speech=sum(x["speech_region_count"] for x in audio["cases"])
    total_known=float(audio.get("processing_time_sec",0))+float(emb["processing_time_sec"]["total"])+float(micro["total_indexing_time_sec"])
    return {"one_time_reusable_cost":True,"frame_sampling_time":"not recorded","clip_encoding_time":"not recorded","vad_time":"not recorded separately","whisper_transcription_time":"not recorded separately","transcript_embedding_time_sec":emb["processing_time_sec"]["transcript_embeddings"],"acoustic_segmentation_time":"not recorded separately","clap_encoding_time_sec":emb["processing_time_sec"]["acoustic_embeddings"],"audio_vad_asr_segmentation_total_sec":audio["processing_time_sec"],"visual_micro_index_total_sec":micro["total_indexing_time_sec"],"total_known_indexing_latency_sec":total_known,"device":{"visual":visual["config"].get("device"),"audio":audio.get("device"),"gpu":audio.get("gpu")},"counts":{"videos":6,"frames":frames,"speech_regions":speech,"acoustic_regions":acoustic},"questions_per_video_known":1,"amortized_known_indexing_cost_per_question_sec":total_known/6}

def task4_comparison(record:dict,evaluation:dict)->dict:
    path=ROOT/"outputs/retrieval"/record["case_id"]/"three_channel_retrieval.json"; base=load(path); results=base["visual_results"]+base["speech_results"]+base["acoustic_results"];planner=next(x for x in load_task5a_records() if x["case_id"]==record["case_id"]);planner_calls=len(planner["attempts"]);planner_tokens=planner["api_usage"]["input_tokens"]+planner["api_usage"]["output_tokens"]
    return {"task4":{"branches":{"visual":True,"speech":True,"acoustic":True},"modality_branch_count":3,"candidate_count":len(results),"selected_duration_sec":union_duration(results),"visual_frame_or_clip_count":len(base["visual_results"]),"retrieval_latency_sec":base.get("retrieval_time_sec"),"reference_hit":base.get("union_diagnostics",{}).get("union_top3_overlap"),"planner_api_calls":0,"total_api_calls":0,"total_tokens":0,"estimated_api_cost":None},"task5b":{"branches":{m:any(x["modality"]==m for x in record["executed_modalities"]) for m in ("visual","speech","acoustic")},"modality_branch_count":len(record["executed_modalities"]),"candidate_count":len(record["selected_candidates"]),"selected_duration_sec":record["selected_duration_sec"],"visual_frame_or_clip_count":record["selected_visual_frame_count"]+record["selected_visual_clip_count"],"retrieval_latency_sec":record["runtime"]["latency_sec"],"reference_hit":evaluation["reference_interval_hit"],"planner_api_calls":planner_calls,"task5a_planner_latency_sec":planner["api_usage"]["latency_sec"],"task5b_llm_api_calls":0,"total_api_calls":planner_calls,"total_tokens":planner_tokens,"estimated_api_cost":None},"interpretation":"Task 4 retrieves all three channels without planner overhead. Task 5B pays the frozen Task 5A planner cost, makes zero Task 5B LLM calls, and skips branches not requested by the plan. Efficiency gain is only supported when measured branches/evidence decrease."}

def load_task5a_records()->list[dict]:return [json.loads(x) for x in PLAN_PATH.read_text(encoding="utf-8").splitlines() if x.strip()]

def evaluate(record:dict,reference:tuple[float,float],duration:float)->dict:
    selected=record["selected_candidates"]; overlaps=[overlap_seconds(x["start_time"],x["end_time"],*reference) for x in selected]; distances=[interval_distance(x["start_time"],x["end_time"],*reference) for x in selected]
    hit=any(x>0 for x in overlaps); broad=any((float(x.get("source_end_time",x["end_time"]))-float(x.get("source_start_time",x["start_time"])))>max(12,3*(reference[1]-reference[0])) and ov>0 for x,ov in zip(selected,overlaps))
    return {"reference_role":"post-hoc weak reference loaded only after retrieval file was saved; not a precise gold boundary","hard_anchor_resolution_success":bool(record["anchor_resolution"]["raw_question_intervals"] or record["anchor_resolution"]["transcript_matches"] or record["anchor_resolution"]["deterministic_interpretations"]),"reference_interval_hit":hit,"maximum_temporal_overlap_sec":max(overlaps,default=0.0),"minimum_distance_to_reference_sec":min(distances,default=None),"selected_evidence_recall":1.0 if hit else 0.0,"selected_candidate_count":len(selected),"broad_region_warning":broad,"evidence_sufficiency_warning":"temporal hit does not prove semantic sufficiency" if hit else "no selected evidence overlaps weak reference; human review required","routing_efficiency":{"modalities_executed":[x["modality"] for x in record["executed_modalities"]],"modalities_skipped":[x["modality"] for x in record["skipped_modalities"]],"visual_index_calls":record["runtime"]["visual_index_calls"],"speech_index_calls":record["runtime"]["speech_index_calls"],"acoustic_index_calls":record["runtime"]["acoustic_index_calls"],"local_original_video_reads":record["runtime"]["local_video_reads"],"selected_video_duration_sec":record["selected_duration_sec"],"selected_visual_frames":record["selected_visual_frame_count"],"selected_visual_clips":record["selected_visual_clip_count"],"percentage_original_video_selected":100*record["selected_duration_sec"]/duration,"task5b_retrieval_latency_sec":record["runtime"]["latency_sec"]}}

def chinese_html(records:list[dict],summary:dict)->str:
    def pre(x):return f"<pre>{html.escape(json.dumps(x,ensure_ascii=False,indent=2))}</pre>"
    def asset(path:str)->str:return Path(os.path.relpath(ROOT/path,OUT)).as_posix()
    cards=[]
    for r in records:
        ev=summary["posthoc_evaluation"][r["case_id"]]; comp=summary["task4_comparison"][r["case_id"]]; visuals=[]
        for x in r["coarse_visual_candidates"]: visuals.append(f'<figure><img src="{html.escape(asset(x["representative_keyframe_path"]))}"><figcaption>{x["region_id"]}: {x["start_time"]:.1f}-{x["end_time"]:.1f}s</figcaption></figure>')
        for x in r["local_visual_refinement"]["dense_frames"]: visuals.append(f'<figure><img src="{html.escape(asset(x["frame_path"]))}"><figcaption>{x["timestamp"]:.1f}s</figcaption></figure>')
        cards.append(f'''<article><h2>{r["case_id"]}</h2><h3>原始问题</h3><p>{html.escape(r["question"])}</p><h3>Task 5A 路由计划</h3>{pre(r["task5a_plan_used"])}<h3>锚点解析</h3>{pre(r["anchor_resolution"])}<h3>实际执行的模态</h3>{pre(r["executed_modalities"])}<h3>跳过的模态</h3>{pre(r["skipped_modalities"])}<h3>Speech 候选证据</h3>{pre(r["speech_candidates"])}<h3>Acoustic 候选证据</h3>{pre(r["acoustic_candidates"])}<h3>Coarse visual 定位</h3><p>Coarse region 用于全局廉价定位；micro-window 和局部密集帧用于回到原视频后的精细取证。</p><div class=gallery>{''.join(visuals)}</div>{pre(r["coarse_visual_candidates"])}<h3>局部视觉精查</h3>{pre(r["local_visual_refinement"])}<h3>跨模态候选连接</h3>{pre(r["linked_windows"])}<h3>全部候选证据</h3>{pre(r["all_candidates"])}<h3>预算筛选后的证据</h3>{pre(r["selected_candidates"])}<h3>被预算移除的候选</h3>{pre(r["budget"]["removed_candidates"])}<h3>效率统计</h3>{pre(r["runtime"])}<h3>与 Task 4 baseline 的比较</h3>{pre(comp)}<h3>潜在问题</h3>{pre({"warnings":r["warnings"],"evaluation":ev,"codex_diagnostic":r["codex_diagnostic"]})}<h3>人工检查建议</h3><p>请核对锚点、转录歧义、coarse-to-micro 来源链和候选语义充分性。时间重叠本身不代表证据正确。</p><textarea placeholder="人工备注"></textarea></article>''')
    return f'''<!doctype html><meta charset=utf-8><title>Task 5B 规划引导的多模态候选检索</title><style>body{{font:15px system-ui;margin:2rem;max-width:1500px;color:#183153}}article{{border-top:4px solid #627d98;margin-top:3rem}}pre{{white-space:pre-wrap;word-break:break-word;background:#f2f5f7;padding:1rem;max-height:500px;overflow:auto}}.gallery{{display:flex;gap:.5rem;overflow:auto}}figure{{margin:0}}img{{height:150px}}textarea{{width:100%;height:80px}}</style><h1>Task 5B：规划引导的多模态候选检索</h1><h2>实验概览</h2><p>本页仅展示候选证据，不生成最终答案。Reference 仅在检索保存完成后用于弱诊断。</p>{pre(summary["aggregate"])}{''.join(cards)}'''

def main()->int:
    ap=argparse.ArgumentParser();ap.parse_args()
    if not PLAN_PATH.is_file():raise SystemExit(f"Missing Task 5A v2 input: {PLAN_PATH}")
    OUT.mkdir(parents=True,exist_ok=True); plans=load_plans_immutable(load_task5a_records());online_started=time.perf_counter();acoustic_queries,acoustic_query_times,acoustic_model_runtime=encode_acoustic_questions(plans)
    # Retrieval allowlist: only video identity/duration are read before post-hoc evaluation.
    manifest_rows=load(MANIFEST_PATH); safe={x["case_id"]:{"video_id":x["video_id"],"video_duration":x["video_duration"]} for x in manifest_rows}
    records=[build_case(row,safe[row["case_id"]],acoustic_queries.get(row["case_id"]),acoustic_query_times.get(row["case_id"],0.0)) for row in plans];end_to_end_online_latency=time.perf_counter()-online_started
    with (OUT/"task5b_candidates.jsonl").open("w",encoding="utf-8") as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False)+"\n")
    # Post-hoc stage begins only after frozen retrieval candidates are saved.
    references={x["case_id"]:(float(x["provided_timestamp_start"]),float(x["provided_timestamp_end"]),float(x["video_duration"])) for x in manifest_rows}
    evaluations={r["case_id"]:evaluate(r,references[r["case_id"]][:2],references[r["case_id"]][2]) for r in records}
    comparisons={r["case_id"]:task4_comparison(r,evaluations[r["case_id"]]) for r in records}
    task5a={r["case_id"]:{"planner_api_calls":len(r["attempts"]),"retries":int(r["retry_required"]),"input_tokens":r["api_usage"]["input_tokens"],"output_tokens":r["api_usage"]["output_tokens"],"latency_sec":r["api_usage"]["latency_sec"],"model_identifier":r["model_identifier"],"prompt_caching":False,"cache_write_tokens":None,"cache_read_tokens":None,"estimated_api_cost":None,"cost_explanation":"No reliable pricing configuration was supplied; live prices were not fetched or assumed."} for r in plans}
    executed={m:sum(any(x["modality"]==m for x in r["executed_modalities"]) for r in records) for m in ("visual","speech","acoustic")}; skipped={m:6-executed[m] for m in executed}
    aggregate={"case_count":6,"execution_status_counts":{s:sum(r["codex_diagnostic"]["execution_status"]==s for r in records) for s in ("complete","partial","failed")},"executed_modality_case_counts":executed,"skipped_modality_case_counts":skipped,"anchor_resolution_success_count":sum(e["hard_anchor_resolution_success"] for e in evaluations.values()),"local_visual_refinement_case_count":sum(r["local_visual_refinement"]["executed"] for r in records),"selected_duration_sec":sum(r["selected_duration_sec"] for r in records),"selected_visual_frame_count":sum(r["selected_visual_frame_count"] for r in records),"selected_visual_clip_count":sum(r["selected_visual_clip_count"] for r in records),"task5b_llm_api_calls":0,"task5b_case_execution_latency_sec":sum(r["runtime"]["latency_sec"] for r in records),"task5b_end_to_end_online_latency_sec":end_to_end_online_latency,"reference_hit_count":sum(e["reference_interval_hit"] for e in evaluations.values()),"task4_total_branch_count":18,"task5b_total_branch_count":sum(len(r["executed_modalities"]) for r in records),"task4_total_candidate_count":sum(c["task4"]["candidate_count"] for c in comparisons.values()),"task5b_total_selected_candidate_count":sum(len(r["selected_candidates"]) for r in records)}
    summary={"task":"Task 5B Planner-Guided Multimodal Candidate Retrieval","retrieval_saved_before_reference_load":True,"no_ground_truth_or_reference_used_for_retrieval":True,"no_llm_or_vlm_calls":True,"aggregate":aggregate,"posthoc_evaluation":evaluations,"task4_comparison":comparisons,"offline_indexing_cost":offline_cost(),"task5a_online_planner_cost":{"per_case":task5a,"aggregate":{"api_calls":sum(x["planner_api_calls"] for x in task5a.values()),"retries":sum(x["retries"] for x in task5a.values()),"input_tokens":sum(x["input_tokens"] for x in task5a.values()),"output_tokens":sum(x["output_tokens"] for x in task5a.values()),"latency_sec":sum(x["latency_sec"] for x in task5a.values()),"estimated_api_cost":None}},"task5b_online_retrieval_cost":{"llm_api_calls":0,"local_index_calls":sum(r["runtime"]["visual_index_calls"]+r["runtime"]["speech_index_calls"]+r["runtime"]["acoustic_index_calls"] for r in records),"new_embedding_computations":sum(r["runtime"]["new_embedding_computations"] for r in records),"new_embedding_computation_time_sec":sum(r["runtime"]["new_embedding_computation_time_sec"] for r in records),"local_acoustic_model_runtime":acoustic_model_runtime,"reused_embeddings":sorted({x for r in records for x in r["runtime"]["reused_embedding_files"]}),"reused_retrieval_score_files":sorted({x for r in records for x in r["runtime"]["reused_retrieval_score_files"]}),"reused_index_metadata_files":sorted({x for r in records for x in r["runtime"]["reused_index_metadata_files"]}),"local_video_reads":sum(r["runtime"]["local_video_reads"] for r in records),"case_execution_latency_sec":aggregate["task5b_case_execution_latency_sec"],"end_to_end_online_latency_sec":aggregate["task5b_end_to_end_online_latency_sec"]},"future_final_qa_cost":{"final_model_api_calls":None,"final_input_tokens":None,"final_output_tokens":None,"final_visual_frames":None,"final_latency_sec":None,"final_estimated_cost":None,"label":"future Task values; not executed"},"warnings":sorted({w for r in records for w in r["warnings"]})}
    save(OUT/"task5b_summary.json",summary)
    md=f'''# Task 5B summary\n\nCandidate retrieval only; no final QA, answer generation, LLM/VLM call, training, or relation-aware reranking.\n\n- Cases: 6\n- Execution complete/partial/failed: {aggregate["execution_status_counts"]}\n- Executed modality case counts: {executed}; skipped: {skipped}\n- Anchor resolution success: {aggregate["anchor_resolution_success_count"]}/6\n- Local visual refinement: {aggregate["local_visual_refinement_case_count"]}/6\n- Selected duration: {aggregate["selected_duration_sec"]:.3f}s\n- Selected visual frames/clips: {aggregate["selected_visual_frame_count"]}/{aggregate["selected_visual_clip_count"]}\n- Task 5A calls/tokens/latency: {summary["task5a_online_planner_cost"]["aggregate"]["api_calls"]} calls, {summary["task5a_online_planner_cost"]["aggregate"]["input_tokens"]}/{summary["task5a_online_planner_cost"]["aggregate"]["output_tokens"]} tokens, {summary["task5a_online_planner_cost"]["aggregate"]["latency_sec"]:.3f}s\n- Task 5B LLM calls: 0\n- Task 5B case execution / end-to-end online latency: {aggregate["task5b_case_execution_latency_sec"]:.3f}s / {aggregate["task5b_end_to_end_online_latency_sec"]:.3f}s\n- Task 4 vs Task 5B branches: {aggregate["task4_total_branch_count"]} vs {aggregate["task5b_total_branch_count"]}\n- Weak-reference hits: {aggregate["reference_hit_count"]}/6 (not proof of semantic quality)\n- Estimated API cost: null; no reliable pricing configuration supplied.\n- Warnings: {summary["warnings"] or "none"}\n''';(OUT/"task5b_summary.md").write_text(md,encoding="utf-8")
    (OUT/"task5b_human_review.html").write_text(chinese_html(records,summary),encoding="utf-8")
    print(json.dumps(aggregate,ensure_ascii=False,indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
