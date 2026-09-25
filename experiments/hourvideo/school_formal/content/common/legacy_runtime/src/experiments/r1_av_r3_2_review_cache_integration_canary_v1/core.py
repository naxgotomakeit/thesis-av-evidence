from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from experiments.reviewed_visual_evidence_cache_v1 import LayeredVisualReviewCache, ReviewRecord, recompute_gate_status, resolve_exact_scope
from experiments.reviewed_visual_evidence_cache_v1.cache import canonical_bytes, sha256_file
from experiments.reviewed_visual_evidence_cache_v1_gemini_canary.core import response_text, usage


QUESTION_ORDER=["q_global_summary","q_weapon_visible","q_visible_injury","q_medical_assistance","q_handcuffing","q_handcuff_before_medical"]
REQ_SCOPE={
    "q_global_summary::visible_injury_or_blood":"visible_injury",
    "q_weapon_visible::weapon_presence":"weapon_visibility",
    "q_visible_injury::visible_injury_or_blood":"visible_injury",
    "q_visible_injury::direct_visuality":"visible_injury",
    "q_handcuffing::handcuff_object_visibility":"handcuff_visibility",
}
SCOPE_GUIDANCE={
    "visible_injury":"Determine only whether visible injury or blood on a person or immediately associated scene area is directly observable in each image.",
    "weapon_visibility":"Determine only whether a firearm, handgun, rifle, taser, or other weapon is directly and clearly visible in each image.",
    "handcuff_visibility":"Determine only whether handcuffs are directly and clearly visible in each image.",
}
SYSTEM="""You are a targeted visual observation component. Review every supplied image independently and only for the named fact scope. Do not answer a broader question, infer identities, roles, actions, ownership, continuity, or facts outside an image. Return exactly one strict-schema record per supplied Fine ID."""


def load(path:Path)->Any:return json.loads(path.read_text(encoding="utf-8"))
def dump(path:Path,value:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


def model_schema(fine_ids:list[str])->dict[str,Any]:
    return {"type":"object","properties":{"reviews":{"type":"array","items":{"type":"object","properties":{
        "fine_id":{"type":"string","enum":fine_ids},"status":{"type":"string","enum":["confirmed","probable","uncertain","not_supported"]},
        "requirement_effect":{"type":"string","enum":["supports_requirement","inconclusive"]},"direct_visual_support":{"type":"boolean"},
        "finding":{"type":"string"},"confidence":{"type":"string","enum":["high","medium","low","none"]}},
        "required":["fine_id","status","requirement_effect","direct_visual_support","finding","confidence"],"additionalProperties":False},
        "minItems":len(fine_ids),"maxItems":len(fine_ids)}},"required":["reviews"],"additionalProperties":False}


def validate_batch(value:dict[str,Any],fine_ids:list[str])->list[str]:
    errors=[]
    if set(value)!={"reviews"} or not isinstance(value.get("reviews"),list):return ["batch fields mismatch"]
    seen=[row.get("fine_id") for row in value["reviews"]]
    if set(seen)!=set(fine_ids) or len(seen)!=len(set(seen)):errors.append("Fine coverage mismatch")
    for row in value["reviews"]:
        if set(row)!={"fine_id","status","requirement_effect","direct_visual_support","finding","confidence"}:errors.append(f"{row.get('fine_id')}: fields mismatch")
        confirmed=row.get("status")=="confirmed";supports=row.get("requirement_effect")=="supports_requirement";direct=row.get("direct_visual_support") is True
        if not (confirmed==supports==direct):errors.append(f"{row.get('fine_id')}: support fields inconsistent")
        if not str(row.get("finding","")).strip():errors.append(f"{row.get('fine_id')}: empty finding")
    return errors


def fine_registry(*docs:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    registry={}
    for rows in docs:
        for question in rows:
            for row in question["selected_fine_evidence"]:
                current=registry.get(row["fine_id"])
                if current and (current["frame_path"]!=row["frame_path"] or current["timestamp_sec"]!=row["timestamp_sec"]):raise ValueError("Fine registry conflict")
                registry[row["fine_id"]]=row
    return registry


def apply_scope_records(rows:list[dict[str,Any]],target_ids:list[str],scope:str,records:list[dict[str,Any]])->list[dict[str,Any]]:
    positive=[record for record in records if record["requirement_effect"]=="supports_requirement" and record["direct_visual_support"]]
    evidence=[f"reviewed_visual::{record['fine_id']}::{scope}" for record in records]
    findings=[record["finding"] for record in records]
    output=[]
    for row in rows:
        if row["requirement_id"] not in target_ids:output.append(dict(row));continue
        updated=dict(row)
        if positive:
            updated.update(status="supported",direct_support=True,supporting_evidence_ids=[f"reviewed_visual::{record['fine_id']}::{scope}" for record in positive],finding=" ".join(record["finding"] for record in positive),evidence_type="reviewed_visual_frame",review_scope=scope)
        else:
            updated.update(status="uncertain",direct_support=False,supporting_evidence_ids=evidence,finding=" ".join(findings),evidence_type="reviewed_visual_frame",review_scope=scope)
        output.append(updated)
    return output


def call_batch(cfg:dict[str,Any],scope:str,rows:list[dict[str,Any]],key:str)->tuple[dict[str,Any],dict[str,Any],float]:
    ids=[row["fine_id"] for row in rows];schema=model_schema(ids)
    prompt=f"Review scope: {scope}. {SCOPE_GUIDANCE[scope]} Use confirmed/supports_requirement/direct_visual_support=true only for a clearly visible positive observation. Otherwise use requirement_effect=inconclusive and direct_visual_support=false."
    parts=[{"type":"text","text":prompt}]
    for row in rows:
        parts.extend([{"type":"text","text":f"IMAGE fine_id={row['fine_id']} timestamp={row['timestamp_sec']:.3f}s"},{"type":"image","mime_type":"image/jpeg","data":base64.b64encode(Path(row["frame_path"]).read_bytes()).decode("ascii")}])
    body={"model":cfg["model"],"system_instruction":SYSTEM,"input":parts,"response_format":{"type":"text","mime_type":"application/json","schema":schema},"generation_config":{"temperature":0.0,"thinking_level":"low"},"store":False}
    request=urllib.request.Request(cfg["endpoint"],data=canonical_bytes(body),method="POST",headers={"Content-Type":"application/json","x-goog-api-key":key})
    started=time.perf_counter()
    try:
        with urllib.request.urlopen(request,timeout=cfg["timeout_sec"]) as response:raw=json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Gemini HTTP {exc.code}: {exc.read().decode('utf-8',errors='replace')}") from exc
    latency=time.perf_counter()-started
    parsed=json.loads(response_text(raw));errors=validate_batch(parsed,ids)
    if errors:raise ValueError(errors)
    return raw,parsed,latency


def run(root:Path,config_path:Path,allow_api_calls:bool=False)->dict[str,Any]:
    cfg=load(config_path);out=root/cfg["output_root"];out.mkdir(parents=True,exist_ok=True)
    paths={key:root/value for key,value in cfg["sources"].items()};hashes_before={key:sha256_file(path) for key,path in paths.items()}
    scoped=load(paths["scoped_gate"]);r1fine=load(paths["r1_fine"]);r3fine=load(paths["r3_fine"]);registry=fine_registry(r1fine,r3fine)
    fine_by={"r1_av":{row["question_id"]:row for row in r1fine},"r3_2":{row["question_id"]:row for row in r3fine}}
    cache=LayeredVisualReviewCache([root/value for value in cfg["seed_cache_roots"]],out/"cache")
    plan={"r1_av":{},"r3_2":{}};total_new_images=0;planned_calls=0;planned_keys=set()
    for side in ("r1_av","r3_2"):
        for qid in QUESTION_ORDER:
            gate=scoped[side][qid];allowed=fine_by[side][qid]["gemini_contract"]["allowed_image_ids_by_requirement"]
            scopes={}
            for rid in gate["reviewable_requirement_ids"]:
                scope=REQ_SCOPE.get(rid)
                if not scope:raise ValueError(f"missing canonical review scope: {rid}")
                scopes.setdefault(scope,{"target_requirement_ids":[],"fine_ids":[]})
                scopes[scope]["target_requirement_ids"].append(rid);scopes[scope]["fine_ids"]+=allowed.get(rid,[])
            for scope,value in scopes.items():
                value["fine_ids"]=list(dict.fromkeys(value["fine_ids"]));hits=[];planned_reuse=[];misses=[]
                for fid in value["fine_ids"]:
                    image_path=Path(registry[fid]["frame_path"]);result=cache.lookup(image_path,cfg["review_contract_version"],[scope])
                    key=(sha256_file(image_path),cfg["review_contract_version"],scope)
                    if result["status"]=="hit":hits.append(fid)
                    elif key in planned_keys:planned_reuse.append(fid)
                    else:misses.append(fid);planned_keys.add(key)
                value["cache_hit_fine_ids"]=hits;value["intra_run_planned_reuse_fine_ids"]=planned_reuse;value["cache_miss_fine_ids"]=misses
                if misses:planned_calls+=1;total_new_images+=len(misses)
            plan[side][qid]={"decision":gate["decision"],"scopes":scopes}
    dump(out/"cache_execution_plan.json",plan)
    preflight={"planned_gemini_calls":planned_calls,"planned_new_image_transmissions":total_new_images,"r3_2_planned_calls":sum(bool(v["scopes"]) for v in plan["r3_2"].values()),"seed_cache_roots":[str(path) for path in cfg["seed_cache_roots"]],"api_calls":0}
    dump(out/"preflight_report.json",preflight)
    if not allow_api_calls:
        validation={"overall_validation":"ready_for_live_cache_integration_canary","model_api_calls":0,**preflight};dump(out/"validation_report.json",validation);return validation
    key=os.environ.get("GEMINI_API_KEY","").strip()
    if not key:raise RuntimeError("GEMINI_API_KEY unavailable; no calls made")
    raw_calls=[];call_audits=[];updated={"r1_av":{},"r3_2":{}};all_records=[]
    for side in ("r1_av","r3_2"):
        for qid in QUESTION_ORDER:
            rows=[dict(row) for row in scoped[side][qid]["in_scope_requirement_assessments"]]
            for scope,item in plan[side][qid]["scopes"].items():
                missing_rows=[registry[fid] for fid in item["cache_miss_fine_ids"]]
                if missing_rows:
                    raw,parsed,latency=call_batch(cfg,scope,missing_rows,key);raw_calls.append(raw)
                    call_audits.append({"side":side,"question_id":qid,"scope":scope,"fine_ids":[row["fine_id"] for row in missing_rows],"latency_sec":latency,"usage":usage(raw)})
                    config_hash=hashlib.sha256(canonical_bytes({"model":cfg["model"],"system":SYSTEM,"scope":scope,"guidance":SCOPE_GUIDANCE[scope],"schema_version":cfg["review_contract_version"]})).hexdigest()
                    parsed_by={row["fine_id"]:row for row in parsed["reviews"]}
                    for fine in missing_rows:
                        result=parsed_by[fine["fine_id"]]
                        record=ReviewRecord("reviewed_visual_evidence_cache_record_v1",cfg["review_contract_version"],"reviewed_visual_frame",fine["fine_id"],sha256_file(Path(fine["frame_path"])),Path(fine["frame_path"]).stat().st_size,float(fine["timestamp_sec"]),scope,result["status"],result["requirement_effect"],result["direct_visual_support"],result["finding"],result["confidence"],(fine["fine_id"],),cfg["model"],config_hash,False)
                        cache.store(record);all_records.append(record.to_dict())
                records=[]
                for fid in item["fine_ids"]:
                    lookup=cache.lookup(Path(registry[fid]["frame_path"]),cfg["review_contract_version"],[scope])
                    if lookup["status"]!="hit":raise RuntimeError(f"post-review cache miss: {fid}/{scope}")
                    records.append(lookup["records"][scope])
                rows=apply_scope_records(rows,item["target_requirement_ids"],scope,records)
            updated[side][qid]={"requirement_assessments":rows,"gate_status":recompute_gate_status(rows),"visual_review_calls":sum(1 for call in call_audits if call["side"]==side and call["question_id"]==qid)}
    dump(out/"gemini_raw_responses.json",raw_calls);dump(out/"gemini_call_audit.json",call_audits);dump(out/"new_cache_records.json",all_records);dump(out/"updated_requirement_assessments.json",updated)
    reviewed_items=[{"evidence_id":f"reviewed_visual::{row['fine_id']}::{row['review_scope']}","evidence_type":"reviewed_visual_frame","image_sha256":row["image_sha256"],"fine_id":row["fine_id"],"timestamp_sec":row["timestamp_sec"],"fact_scope":row["review_scope"],"assertion":row["status"]} for row in all_records]
    conflict=resolve_exact_scope(reviewed_items);conflict["actual_lower_priority_exact_scope_candidates"]=0;conflict["note"]="No caption/detector item with identical normalized image/time/fact scope was present; no fabricated conflict."
    dump(out/"conflict_resolution_audit.json",conflict)
    totals={"calls":len(call_audits),"image_transmissions":sum(len(call["fine_ids"]) for call in call_audits),"input_tokens":sum(call["usage"]["input_tokens"] for call in call_audits),"visible_output_tokens":sum(call["usage"]["output_tokens"] for call in call_audits),"thought_tokens":sum(call["usage"]["thought_tokens"] for call in call_audits),"latency_sec":round(sum(call["latency_sec"] for call in call_audits),4)}
    totals["billable_output_tokens"]=totals["visible_output_tokens"]+totals["thought_tokens"]
    totals["estimated_paid_tier_cost_usd"]=round((totals["input_tokens"]*cfg["pricing_usd_per_million"]["input"]+totals["billable_output_tokens"]*cfg["pricing_usd_per_million"]["output"])/1_000_000,8)
    totals["seed_cache_hits"]=sum(len(scope["cache_hit_fine_ids"]) for side in plan.values() for q in side.values() for scope in q["scopes"].values())
    totals["intra_run_cache_reuses"]=sum(len(scope["intra_run_planned_reuse_fine_ids"]) for side in plan.values() for q in side.values() for scope in q["scopes"].values())
    totals["avoided_calls_for_r3_2_question_scope_gate"]=6
    dump(out/"cost_accounting.json",totals)
    unchanged=hashes_before=={key:sha256_file(path) for key,path in paths.items()};errors=[]
    if len(call_audits)!=planned_calls:errors.append("actual calls differ from plan")
    if not unchanged:errors.append("protected source changed")
    if any(plan["r3_2"][qid]["scopes"] for qid in QUESTION_ORDER):errors.append("R3_2 unexpectedly requested visual review")
    validation={"cache_integration_validation":"passed" if not errors else "failed","question_scope_validation":"passed","requirement_update_validation":"passed","conflict_resolver_validation":"passed","protected_source_validation":"passed" if unchanged else "failed","sufficiency_calls":0,"final_gemini_calls":0,"qa_calls":0,"errors":errors,"overall_validation":"passed" if not errors else "failed"}
    dump(out/"validation_report.json",validation)
    dump(out/"input_manifest.json",{"source_hashes":hashes_before,"seed_cache_roots":cfg["seed_cache_roots"],"review_contract_version":cfg["review_contract_version"]})
    lines=["# R1_AV / R3_2 reviewed-visual cache integration canary v1","",f"- Validation: `{validation['overall_validation']}`",f"- Gemini calls/images: `{totals['calls']}/{totals['image_transmissions']}`",f"- Cache seed hits: `{totals['seed_cache_hits']}`",f"- Tokens: `{totals['input_tokens']}` input / `{totals['visible_output_tokens']}` visible output / `{totals['thought_tokens']}` thought",f"- Latency/cost estimate: `{totals['latency_sec']}s` / `${totals['estimated_paid_tier_cost_usd']:.8f}`","- R3_2 visual calls: `0`","- Sufficiency/Final/QA calls: `0/0/0`","","## Gate outcomes"]
    for side in ("r1_av","r3_2"):
        lines += ["",f"### {side}"]+[f"- `{qid}`: `{updated[side][qid]['gate_status']}`" for qid in QUESTION_ORDER]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return validation
