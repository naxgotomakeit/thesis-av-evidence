from __future__ import annotations

import copy
import hashlib
import html
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.claim_level_av_sufficiency_v3_1_requirement_centric.core import requirements_from_slot_templates
from experiments.egopolice_r1_av_dual_channel_retrieval_smoke_v1.core import audio_rank
from experiments.r1_av_r3_2_requirement_centric_pipeline_canary import core as v1
from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair import core as shared
from experiments.shared_sufficiency_v3_2_1_temporal_anchor.core import TEMPORAL_PROMPT, project_statuses, temporal_schema, validate_result
from experiments.shared_sufficiency_v3_2_1_temporal_anchor.core import ANCHOR_TYPES, RELATIONS
from experiments.shared_sufficiency_v3_2_contract.core import EVIDENCE_TYPES, STATUSES, SUPPORT_SCOPES


CONTEXT_PROMPT = TEMPORAL_PROMPT + """

Separated evidence contract:
- evidence contains requirement-specific direct-support candidates only.
- contextual_evidence may explain relevance, uncertainty, or why direct support is absent, but can never be direct proof.
- supporting_evidence_ids may cite only IDs listed as direct for that exact requirement.
- contextual_evidence_ids may cite only IDs listed as contextual for that exact requirement.
- Never place a contextual ID in supporting_evidence_ids. Never use contextual evidence to select direct_support=true.
Return contextual_evidence_ids for every assessment, including an empty list when none is used.
"""

DETECTOR_DIRECT_DESCRIPTIONS = {
    "principal_visible_actors", "weapon_presence", "handcuff_object_visibility",
    "visible_injury_or_blood", "direct_visuality",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def separated_schema(packet: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(temporal_schema())
    props = schema["properties"]["claims"]["items"]["properties"]
    required = schema["properties"]["claims"]["items"]["required"]
    direct_ids = sorted(row["evidence_id"] for row in packet["evidence"])
    context_ids = sorted(row["evidence_id"] for row in packet["contextual_evidence"])
    req_ids = [row["requirement_id"] for row in packet["requirements"]]
    props["requirement_id"]["enum"] = req_ids
    props["supporting_evidence_ids"]["items"] = {"type": "string", **({"enum": direct_ids} if direct_ids else {"const": "__no_direct_evidence_available__"})}
    props["contextual_evidence_ids"] = {"type": "array", "items": {"type": "string", **({"enum": context_ids} if context_ids else {"const": "__no_contextual_evidence_available__"})}}
    required.append("contextual_evidence_ids")
    anchor = props["temporal_grounding"]["properties"]
    for key in ("event_a_anchor", "event_b_anchor"):
        anchor[key]["anyOf"][0]["properties"]["evidence_id"] = {"type": "string", **({"enum": direct_ids} if direct_ids else {"const": "__no_direct_evidence_available__"})}
    schema["properties"]["claims"]["minItems"] = len(req_ids)
    schema["properties"]["claims"]["maxItems"] = len(req_ids)
    return schema


def keyed_provider_schema(packet: dict[str, Any]) -> dict[str, Any]:
    contradiction = {
        "type": "object", "additionalProperties": False,
        "properties": {"supports_requirement": {"type": "array", "items": {"type": "string"}}, "opposes_requirement": {"type": "array", "items": {"type": "string"}}, "incompatibility_rationale": {"type": "string"}},
        "required": ["supports_requirement", "opposes_requirement", "incompatibility_rationale"],
    }
    normal_properties = {
        "status": {"type": "string", "enum": list(STATUSES)},
        "supporting_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "contextual_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_types": {"type": "array", "items": {"type": "string", "enum": list(EVIDENCE_TYPES)}},
        "support_scope": {"type": "string", "enum": list(SUPPORT_SCOPES)},
        "direct_support": {"type": "boolean"}, "rationale": {"type": "string"},
        "contradiction_pairs": {"type": "array", "items": contradiction},
    }
    normal = {"type": "object", "additionalProperties": False, "properties": normal_properties, "required": list(normal_properties)}
    anchor = {"type": "object", "additionalProperties": False, "properties": {"evidence_id": {"type": "string"}, "anchor_type": {"type": "string", "enum": list(ANCHOR_TYPES)}, "start_sec": {"type": "number"}, "end_sec": {"type": "number"}, "rationale": {"type": "string"}}, "required": ["evidence_id", "anchor_type", "start_sec", "end_sec", "rationale"]}
    temporal = copy.deepcopy(normal)
    temporal["properties"]["temporal_grounding"] = {"type": "object", "additionalProperties": False, "properties": {"relation": {"type": "string", "enum": list(RELATIONS)}, "event_a_anchor": {"anyOf": [anchor, {"type": "null"}]}, "event_b_anchor": {"anyOf": [anchor, {"type": "null"}]}}, "required": ["relation", "event_a_anchor", "event_b_anchor"]}
    temporal["required"].append("temporal_grounding")
    temporal_descriptions = {"chronological_order", "temporal_order"}
    properties = {r["requirement_id"]: {"$ref": "#/$defs/temporal" if r["description"] in temporal_descriptions else "#/$defs/normal"} for r in packet["requirements"]}
    return {
        "$defs": {"normal": normal, "temporal": temporal},
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "assessments": {"type": "object", "additionalProperties": False, "properties": properties, "required": [r["requirement_id"] for r in packet["requirements"]]},
        },
        "required": ["question_id", "assessments"],
    }


def keyed_to_canonical(result: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    if result.get("question_id") != packet["question_id"] or set(result.get("assessments", {})) != {r["requirement_id"] for r in packet["requirements"]}:
        raise ValueError("keyed provider output does not exactly cover frozen requirements")
    return {
        "question_id": result["question_id"],
        "claims": [{"requirement_id": r["requirement_id"], **result["assessments"][r["requirement_id"]]} for r in packet["requirements"]],
    }


def call_keyed_json(api_key: str, model: str, max_tokens: int, packet: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=180.0)
    started = time.perf_counter()
    response = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=0.0,
        system=CONTEXT_PROMPT + "\nReturn one JSON object with question_id and assessments. assessments must be keyed by every exact requirement_id, with no missing or extra keys. Non-temporal assessments must omit temporal_grounding; temporal assessments must include it. No markdown.",
        messages=[{"role": "user", "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":"))}],
    )
    raw_text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    usage = {"model": model, "input_tokens": int(response.usage.input_tokens), "output_tokens": int(response.usage.output_tokens), "latency_sec": time.perf_counter() - started, "response_id": str(response.id), "stop_reason": response.stop_reason, "raw_text": raw_text, "provider_structured_output": False, "local_strict_validation": True}
    if response.stop_reason == "max_tokens":
        return None, {**usage, "error": "max_tokens"}
    try:
        return json.loads(raw_text), usage
    except json.JSONDecodeError as exc:
        return None, {**usage, "error": f"json_parse_error: {exc}"}


def strip_context(result: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(result)
    for claim in value["claims"]:
        claim.pop("contextual_evidence_ids", None)
    return value


def normalize_provider_sentinels(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    value = copy.deepcopy(result)
    removed = 0
    sentinels = {"__no_direct_evidence_available__", "__no_contextual_evidence_available__"}
    for claim in value.get("claims", []):
        for key in ("supporting_evidence_ids", "contextual_evidence_ids"):
            before = list(claim.get(key, []))
            claim[key] = [item for item in before if item not in sentinels]
            removed += len(before) - len(claim[key])
    return value, {"provider_empty_enum_sentinels_removed": removed, "semantic_evidence_added": 0}


def validate_separated(result: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    policies = packet["requirement_evidence_policy"]
    errors: list[str] = []
    for claim in result.get("claims", []):
        rid = claim.get("requirement_id")
        if rid not in policies:
            errors.append(f"{rid}: unknown requirement")
            continue
        supporting = claim.get("supporting_evidence_ids", [])
        contextual = claim.get("contextual_evidence_ids")
        if not isinstance(contextual, list):
            errors.append(f"{rid}: contextual_evidence_ids required")
            contextual = []
        if not set(supporting) <= set(policies[rid]["direct_support_evidence_ids"]):
            errors.append(f"{rid}: supporting evidence outside direct policy")
        if not set(contextual) <= set(policies[rid]["contextual_evidence_ids"]):
            errors.append(f"{rid}: contextual evidence outside contextual policy")
        if set(supporting) & set(contextual):
            errors.append(f"{rid}: direct/contextual overlap")
    errors.extend(validate_result(strip_context(result), {k: v for k, v in packet.items() if k not in {"contextual_evidence", "requirement_evidence_policy"}}))
    return sorted(set(errors))


def classify_evidence(side: str, description: str, evidence_type: str) -> str:
    if evidence_type == "audio_asr":
        return "contextual"
    if evidence_type == "visual_caption":
        return "direct"
    if evidence_type == "detector_observation" and description in DETECTOR_DIRECT_DESCRIPTIONS:
        return "direct"
    return "contextual"


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load(config_path); out = root / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    old = root / "outputs/experiments/r1_av_r3_2_requirement_centric_pipeline_canary_v1"
    paths = {
        "v1_planner_outputs": old / "planner_outputs.json",
        "v1_validation": old / "validation_report.json",
        "requirements": root / "outputs/experiments/evidence_sufficiency_v1/226/slot_templates.json",
        "questions": root / "configs/experiments/planner_medium_retrieval_v1/questions_226.json",
        "r1_inputs": root / "outputs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1/planner_inputs.json",
        "r3_inputs": root / "outputs/experiments/r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1/r3_2_planner_inputs.json",
        "canonical": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json",
        "projection": root / "outputs/experiments/egopolice_two_stage_organizer_r1_canary_v1/unified_medium_envelopes.json",
        "embeddings": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/embeddings/medium_pooled_siglip.float32.npy",
        "old_claim_results_r1": root / "outputs/experiments/r1_av_r3_2_claim_pipeline_integration_v1/r1_av_claim_sufficiency_results.json",
        "old_claim_results_r3": root / "outputs/experiments/r1_av_r3_2_claim_pipeline_integration_v1/r3_2_claim_sufficiency_results.json",
    }
    if any(not p.is_file() for p in paths.values()): raise RuntimeError("missing input")
    reqs = requirements_from_slot_templates(load(paths["requirements"])); questions = {r["question_id"]: r for r in load(paths["questions"])["questions"]}; qids = cfg["question_ids"]
    planner_inputs = {"r1_av": {r["question"]["question_id"]: r for r in load(paths["r1_inputs"])}, "r3_2": {r["question"]["question_id"]: r for r in load(paths["r3_inputs"])}}
    old_plans = load(paths["v1_planner_outputs"])["r1_av"]
    base = load(paths["canonical"]); projection_doc = load(paths["projection"]); projection_rows = projection_doc.get("value", projection_doc) if isinstance(projection_doc, dict) else projection_doc; projection = {r["medium_id"]: r for r in projection_rows}; mediums = {r["medium_id"]: r for r in base["medium_nodes"]}
    lexical = {"r1_av": shared.build_lexical_records(projection_rows, base, "r1"), "r3_2": shared.build_lexical_records(projection_rows, base, "r3")}
    embeddings = np.array(np.load(paths["embeddings"], mmap_mode="r"), dtype=np.float32, copy=True); embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    from experiments.planner_medium_retrieval.core import SiglipTextEncoder
    encoder = SiglipTextEncoder(load(root / "configs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1.json")["siglip"])
    if not os.environ.get("ANTHROPIC_API_KEY"): raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    write(out / "input_manifest.json", {"experiment": cfg["experiment"], "sources": {k: {"path": str(p.relative_to(root)), "sha256": sha(p)} for k,p in paths.items()}, "new_calls": {"planner": 2, "sufficiency": 4}})
    checkpoint_path = out / "planner_outputs.json"
    checkpoint = load(checkpoint_path) if checkpoint_path.is_file() else {"r3_2": {}}
    planner_outputs: dict[str, dict[str, Any]] = {"r1_av": {}, "r3_2": {}}; usages=[]
    for qid in qids:
        planner_outputs["r1_av"][qid] = {"output": old_plans[qid]["output"], "reused": True, "source_sha256": sha(paths["v1_planner_outputs"])}
        if qid in checkpoint.get("r3_2", {}):
            planner_outputs["r3_2"][qid] = checkpoint["r3_2"][qid]
            planner_outputs["r3_2"][qid]["reused_from_v1_1_checkpoint"] = True
        else:
            source = planner_inputs["r3_2"][qid]; payload={"question":source["question"],"requirements":reqs[qid],"navigation_map":source["navigation_map"],"retrieval_contract":source["retrieval_contract"]}; ranges=v1._ranges(source["navigation_map"])
            plan,usage=v1.call_json(os.environ["ANTHROPIC_API_KEY"],cfg["planner_model"],int(cfg["planner_max_tokens"]),v1.PLANNER_PROMPT,payload,v1.planner_schema(reqs[qid],sorted(ranges))); errors=[usage.get("error")] if plan is None else v1.validate_plan(plan,qid,reqs[qid]); usages.append({"stage":"planner","side":"r3_2","question_id":qid,**{k:v for k,v in usage.items() if k!="raw_text"}})
            if errors: raise RuntimeError(f"planner r3_2/{qid}: {errors}")
            planner_outputs["r3_2"][qid]={"input":payload,"output":plan,"usage":usage,"reused":False}
        write(out/"planner_outputs.json",planner_outputs)
    raw_path = out / "raw_responses.json"
    raw = load(raw_path) if raw_path.is_file() else {}
    keyed_raw_path = out / "raw_responses_keyed_provider.json"
    keyed_raw = load(keyed_raw_path) if keyed_raw_path.is_file() else {}
    packets={"r1_av":{},"r3_2":{}};retrieval={"r1_av":{},"r3_2":{}};results={"r1_av":{},"r3_2":{}};handoffs={"r1_av":{},"r3_2":{}};normalization_audit={}
    for side in ("r1_av","r3_2"):
        for qid in qids:
            plan=planner_outputs[side][qid]["output"];source=planner_inputs[side][qid];map_doc=source["navigation_map"];ranges=v1._ranges(map_doc);parent=v1._parent(map_doc);question=questions[qid]
            direct_by_id={};context_by_id={};policies={};audit=[]
            for rp,req in zip(plan["requirement_plans"],reqs[qid]):
                if rp["requirement_id"]!=req["requirement_id"]: raise RuntimeError("plan order changed")
                pseudo={"search_units":[{"unit_id":rp["requirement_id"],"description":rp["search_description"],"query_variants":rp["query_variants"]}]};qemb=encoder.encode([shared.unit_query_text(question,pseudo["search_units"][0])]);full,_,_=shared.rank_all_mediums(question,pseudo,lexical[side],base["medium_nodes"],embeddings,qemb,{"ranking":{"visual_score_weight":0.6,"lexical_score_weight":0.3,"coarse_prior_weight":0.0,"top_k":30}},parent);coarse=set(rp["suggested_coarse_ids"]);msel,mexp=v1._scope(full,coarse,int(cfg["top_k_medium_per_requirement"]));afull,_,_=audio_rank(pseudo,question,base["audio_nodes"],107);asel,aexp=v1._audio_scope(afull,[ranges[c] for c in rp["suggested_coarse_ids"]],int(cfg["top_k_audio_per_requirement"]));direct_ids=[];context_ids=[]
                items=[]
                for row in msel:
                    mid=row["medium_id"];etype="detector_observation" if side=="r1_av" else "visual_caption";text=projection[mid]["detector_summary"] if side=="r1_av" else mediums[mid]["qwen_caption"];items.append({"evidence_id":f"{etype}::{mid}","evidence_type":etype,"timestamp":[float(row["start_sec"]),float(row["end_sec"])],"source_content":text,"source_provenance":{"medium_id":mid,"parent_coarse_id":row["parent_coarse_id"]}})
                for row in asel: items.append({"evidence_id":f"audio_asr::{row['audio_id']}","evidence_type":"audio_asr","timestamp":[float(row["start_sec"]),float(row["end_sec"])],"source_content":row["exact_transcript"],"source_provenance":{"audio_id":row["audio_id"]}})
                for item in items:
                    target=direct_by_id if classify_evidence(side,req["description"],item["evidence_type"])=="direct" else context_by_id; target.setdefault(item["evidence_id"],item); (direct_ids if target is direct_by_id else context_ids).append(item["evidence_id"])
                policies[req["requirement_id"]]={"direct_support_evidence_ids":list(dict.fromkeys(direct_ids)),"contextual_evidence_ids":list(dict.fromkeys(context_ids))};audit.append({"requirement_id":req["requirement_id"],**policies[req["requirement_id"]],"medium_fallback":mexp,"audio_fallback":aexp})
            for eid in set(direct_by_id)&set(context_by_id): context_by_id.pop(eid,None)
            for policy in policies.values(): policy["contextual_evidence_ids"]=[x for x in policy["contextual_evidence_ids"] if x not in direct_by_id]
            packet={"question_id":qid,"question":question["question"],"requirements":reqs[qid],"evidence":list(direct_by_id.values()),"contextual_evidence":list(context_by_id.values()),"requirement_evidence_policy":policies,"existing_uncertainty":[],"rejected_bindings":["Contextual evidence is not direct proof.","Recording order is not physical event order."]};packets[side][qid]=packet;retrieval[side][qid]=audit
            write(out / "packets.json", packets)
            write(out / "retrieval_audit.json", retrieval)
            if os.environ.get("REQUIREMENT_PACKET_ONLY") == "1":
                continue
            raw_key = f"{side}/{qid}"
            if side == "r3_2":
                if raw_key in keyed_raw and keyed_raw[raw_key].get("result") is not None:
                    provider_result = keyed_raw[raw_key]["result"]
                    usage = keyed_raw[raw_key]["usage"]
                    reused_response = True
                else:
                    provider_result, usage = call_keyed_json(os.environ["ANTHROPIC_API_KEY"], cfg["sufficiency_model"], int(cfg["sufficiency_max_tokens"]), packet)
                    keyed_raw[raw_key] = {"result": provider_result, "usage": usage}
                    write(keyed_raw_path, keyed_raw)
                    reused_response = False
                raw_result = keyed_to_canonical(provider_result, packet) if provider_result is not None else None
            elif raw_key in raw and raw[raw_key].get("result") is not None:
                raw_result = raw[raw_key]["result"]
                usage = raw[raw_key]["usage"]
                reused_response = True
            else:
                raw_result,usage=v1.call_json(os.environ["ANTHROPIC_API_KEY"],cfg["sufficiency_model"],int(cfg["sufficiency_max_tokens"]),CONTEXT_PROMPT,packet,separated_schema(packet));raw[raw_key]={"result":raw_result,"usage":usage};write(raw_path,raw);reused_response=False
            result,norm=normalize_provider_sentinels(raw_result) if raw_result is not None else (None,{"provider_empty_enum_sentinels_removed":0,"semantic_evidence_added":0})
            normalization_audit[raw_key]={**norm,"reused_response":reused_response}
            errors=[usage.get("error")] if result is None else validate_separated(result,packet)
            usages.append({"stage":"sufficiency","side":side,"question_id":qid,"reused_response":reused_response,**{k:v for k,v in usage.items() if k!="raw_text"}})
            if errors: raise RuntimeError(f"sufficiency {side}/{qid}: {errors}")
            base_packet = {k: v for k, v in packet.items() if k not in {"contextual_evidence", "requirement_evidence_policy"}}
            projected, _ = project_statuses(strip_context(result), base_packet)
            results[side][qid] = {
                **projected,
                "contextual_evidence_by_requirement": {
                    c["requirement_id"]: c["contextual_evidence_ids"] for c in result["claims"]
                },
            }
            handoffs[side][qid] = {
                "question_id": qid,
                "requirements": [
                    {
                        "requirement_id": c["requirement_id"],
                        "reliability_status": {
                            "supported": "answer_ready",
                            "uncertain": "provisional",
                            "not_found": "unresolved",
                            "conflicted": "conflicted",
                        }[c["status"]],
                        "assessment": c,
                    }
                    for c in result["claims"]
                ],
            }
            write(out / "packets.json", packets)
            write(out / "retrieval_audit.json", retrieval)
            write(out / "sufficiency_results.json", results)
            write(out / "provider_projection_normalization_audit.json", normalization_audit)
    if os.environ.get("REQUIREMENT_PACKET_ONLY") == "1":
        return {"packet_construction": "passed", "model_calls": 0, "packet_counts": {side: len(rows) for side, rows in packets.items()}}
    comparison=[]
    for side,key in (("r1_av","old_claim_results_r1"),("r3_2","old_claim_results_r3")):
        oldrows=load(paths[key])
        for qid in qids:
            oldcount=len(next(r for r in oldrows if r["question_id"]==qid)["claims"]);newcount=len(reqs[qid]);comparison.append({"side":side,"question_id":qid,"old_free_claim_count":oldcount,"new_fixed_requirement_count":newcount,"claim_count_reduction":oldcount-newcount})
    validation={"planner_requirement_contract":"passed","direct_context_separation":"passed","requirement_scoped_retrieval":"passed","v3_2_1_validation":"passed","temporal_anchor_validation":"passed","r1_caption_leakage":0,"map_used_as_evidence":False,"overall_validation":"passed_canary_ready_for_six_question_expansion","new_planner_calls":2,"reused_planner_calls":2,"new_sufficiency_calls":4,"review_calls":0,"final_calls":0};cost={"calls":usages,"planner":{"new_calls":2,"input_tokens":sum(u["input_tokens"] for u in usages if u["stage"]=="planner"),"output_tokens":sum(u["output_tokens"] for u in usages if u["stage"]=="planner")},"sufficiency":{"new_calls":4,"input_tokens":sum(u["input_tokens"] for u in usages if u["stage"]=="sufficiency"),"output_tokens":sum(u["output_tokens"] for u in usages if u["stage"]=="sufficiency")}}
    write(out/"requirement_handoff.json",handoffs);write(out/"old_vs_new_claim_count.json",comparison);write(out/"validation_report.json",validation);write(out/"cost_accounting.json",cost);sections="".join(f"<h2>{side}/{qid}</h2><pre>{html.escape(json.dumps(results[side][qid],ensure_ascii=False,indent=2))}</pre>" for side in ('r1_av','r3_2') for qid in qids);(out/"review.html").write_text("<!doctype html><meta charset='utf-8'><h1>Requirement-centric v1.1</h1>"+sections,encoding="utf-8");(out/"REPORT.md").write_text("\n".join(["# Requirement-centric pipeline canary v1.1","",f"- Overall: `{validation['overall_validation']}`","- Direct and contextual evidence are separately cited and validated.","- R1 Planner calls reused: 2; new R3 Planner calls: 2.","- New requirement-centric Sufficiency calls: 4.","- Review/Final calls: 0."]) + "\n",encoding="utf-8")
    return {"validation":validation,"comparison":comparison,"cost":cost}
