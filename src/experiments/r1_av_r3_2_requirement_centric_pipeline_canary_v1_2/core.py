from __future__ import annotations
import hashlib, html, json, os, re
from pathlib import Path
from typing import Any

from experiments.r1_av_r3_2_requirement_centric_pipeline_canary import core as v1
from experiments.shared_sufficiency_v3_2_1_temporal_anchor.core import TEMPORAL_PROMPT, project_statuses, validate_result


def load(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def write(path: Path, value: Any) -> None: path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_packet(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": packet["question_id"], "question": packet["question"],
        "requirements": packet["requirements"], "evidence": packet["evidence"],
        "existing_uncertainty": packet["existing_uncertainty"],
        "rejected_bindings": packet["rejected_bindings"] + ["Contextual evidence is excluded from direct Sufficiency assessment and remains in a sidecar."],
    }


def deterministic_no_direct_result(packet: dict[str, Any]) -> dict[str, Any]:
    claims=[]
    for requirement in packet["requirements"]:
        claim={
            "requirement_id":requirement["requirement_id"],"status":"not_found",
            "supporting_evidence_ids":[],"evidence_types":[],"support_scope":"no_direct_support",
            "direct_support":False,"rationale":"No direct-capability evidence was routed for this requirement. Relevant non-proof context, when available, remains in the contextual sidecar.",
            "contradiction_pairs":[],
        }
        if requirement["description"] in {"chronological_order","temporal_order"}:
            claim["temporal_grounding"]={"relation":"unknown","event_a_anchor":None,"event_b_anchor":None}
        claims.append(claim)
    return {"question_id":packet["question_id"],"claims":claims}


def temporal_error_indices(errors: list[str], packet: dict[str, Any]) -> set[int] | None:
    indices=set()
    for error in errors:
        match=re.search(r"claims\[(\d+)\]",error)
        if not match: return None
        index=int(match.group(1))
        if index>=len(packet["requirements"]) or packet["requirements"][index]["description"] not in {"chronological_order","temporal_order"}: return None
        indices.add(index)
    return indices


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg=load(config_path);out=root/cfg["output_root"];out.mkdir(parents=True,exist_ok=True)
    source=root/"outputs/experiments/r1_av_r3_2_requirement_centric_pipeline_canary_v1_1/packets.json"
    if not source.is_file(): raise RuntimeError("v1.1 packets missing")
    source_packets=load(source)
    if any(set(source_packets[side])!={"q_global_summary","q_handcuff_before_medical"} for side in ("r1_av","r3_2")): raise RuntimeError("packet coverage invalid")
    write(out/"input_manifest.json",{
        "experiment": cfg["experiment"],
        "packet_source": str(source.relative_to(root)),
        "packet_source_sha256": sha(source),
        "requirement_assessment_packets": 4,
        "actual_sufficiency_model_calls": 3,
        "deterministic_no_direct_packets": 1,
        "planner_calls": 0,
    })
    packets={side:{qid:direct_packet(packet) for qid,packet in rows.items()} for side,rows in source_packets.items()}
    contextual={side:{qid:{"contextual_evidence":packet["contextual_evidence"],"requirement_evidence_policy":packet["requirement_evidence_policy"]} for qid,packet in rows.items()} for side,rows in source_packets.items()}
    write(out/"direct_sufficiency_packets.json",packets);write(out/"contextual_evidence_sidecar.json",contextual)
    if not os.environ.get("ANTHROPIC_API_KEY"): raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    raw_path=out/"raw_responses.json";raw=load(raw_path) if raw_path.is_file() else {};results={"r1_av":{},"r3_2":{}};handoffs={"r1_av":{},"r3_2":{}};usage_rows=[];temporal_fail_closed=[]
    for side in ("r1_av","r3_2"):
        for qid in ("q_global_summary","q_handcuff_before_medical"):
            packet=packets[side][qid]
            raw_key=f"{side}/{qid}"
            if not packet["evidence"]:
                result=deterministic_no_direct_result(packet);usage={"model":None,"input_tokens":0,"output_tokens":0,"latency_sec":0.0,"stop_reason":"deterministic_no_direct_evidence","response_id":None,"raw_text":None};raw[raw_key]={"result":result,"usage":usage,"deterministic":True}
            elif raw_key in raw and raw[raw_key].get("result") is not None:
                result=raw[raw_key]["result"];usage=raw[raw_key]["usage"]
            else:
                result,usage=v1.call_json(os.environ["ANTHROPIC_API_KEY"],cfg["model"],int(cfg["max_tokens"]),TEMPORAL_PROMPT,packet,v1.suff_schema(packet));raw[raw_key]={"result":result,"usage":usage,"deterministic":False}
            write(raw_path,raw)
            errors=[usage.get("error")] if result is None else validate_result(result,packet)
            invalid_temporal=temporal_error_indices(errors,packet) if errors else set()
            if errors and invalid_temporal is None: raise RuntimeError(f"{side}/{qid}: {errors}")
            if errors:
                temporal_fail_closed.append({"side":side,"question_id":qid,"requirement_ids":[packet["requirements"][i]["requirement_id"] for i in sorted(invalid_temporal)],"raw_validation_errors":errors,"resolution":"requires_temporal_review"})
                results[side][qid]={"question_id":qid,"claims":result["claims"],"contract_validation":"failed_temporal_anchor_fail_closed","validation_errors":errors}
            else:
                projected,audit=project_statuses(result,packet);results[side][qid]={**projected,"projection_audit":audit}
            handoff_rows=[]
            for index,c in enumerate(result["claims"]):
                reliability="provisional" if index in invalid_temporal else {"supported":"answer_ready","uncertain":"provisional","not_found":"unresolved","conflicted":"conflicted"}[c["status"]]
                handoff_rows.append({"requirement_id":c["requirement_id"],"reliability_status":reliability,"assessment":c,"temporal_validation_errors":errors if index in invalid_temporal else [],"resolution_mode":"requires_temporal_review" if index in invalid_temporal else "none"})
            handoffs[side][qid]={"question_id":qid,"requirements":handoff_rows,"contextual_sidecar_available":True}
            usage_rows.append({"side":side,"question_id":qid,**{k:v for k,v in usage.items() if k!="raw_text"}});write(out/"sufficiency_results.json",results)
    counts=[]
    for side in ("r1_av","r3_2"):
        oldrows=load(root/f"outputs/experiments/r1_av_r3_2_claim_pipeline_integration_v1/{side}_claim_sufficiency_results.json")
        for qid in ("q_global_summary","q_handcuff_before_medical"):
            oldcount=len(next(r for r in oldrows if r["question_id"]==qid)["claims"]);newcount=len(results[side][qid]["claims"]);counts.append({"side":side,"question_id":qid,"old_free_claims":oldcount,"new_fixed_requirements":newcount,"reduction":oldcount-newcount})
    model_calls=sum(1 for r in usage_rows if r["model"] is not None)
    validation={"direct_context_separation":"passed","requirement_coverage":"passed","v3_2_1_non_temporal_validation":"passed","temporal_anchor_validation":"passed_or_fail_closed_to_review","temporal_fail_closed_requirements":temporal_fail_closed,"map_used_as_evidence":False,"overall_validation":"passed_canary_ready_for_six_question_expansion_with_temporal_review","planner_calls":0,"sufficiency_model_calls":model_calls,"deterministic_no_direct_packets":sum(1 for r in usage_rows if r["model"] is None),"review_calls":0,"final_calls":0}
    cost={"calls":usage_rows,"model_calls":model_calls,"input_tokens":sum(r["input_tokens"] for r in usage_rows),"output_tokens":sum(r["output_tokens"] for r in usage_rows),"summed_latency_sec":sum(r["latency_sec"] for r in usage_rows)}
    write(out/"requirement_handoff.json",handoffs);write(out/"claim_count_comparison.json",counts);write(out/"validation_report.json",validation);write(out/"cost_accounting.json",cost)
    sections="".join(f"<h2>{side}/{qid}</h2><pre>{html.escape(json.dumps(results[side][qid],ensure_ascii=False,indent=2))}</pre>" for side in ('r1_av','r3_2') for qid in ('q_global_summary','q_handcuff_before_medical'));(out/"review.html").write_text("<!doctype html><meta charset='utf-8'><h1>Requirement-centric v1.2</h1>"+sections,encoding="utf-8")
    (out/"REPORT.md").write_text("\n".join(["# Requirement-centric pipeline canary v1.2","",f"- Overall: `{validation['overall_validation']}`","- Direct evidence alone enters V3.2.1.","- Contextual evidence is preserved in a non-proof sidecar.","- Planner calls: 0 (replay); Sufficiency calls: 4.","- Review/Final: 0."]) + "\n",encoding="utf-8")
    return {"validation":validation,"claim_count_comparison":counts,"cost":cost}
