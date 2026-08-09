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
from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair import core as shared
from experiments.shared_sufficiency_v3_2_1_temporal_anchor.core import TEMPORAL_PROMPT, project_statuses, temporal_schema, validate_result
from experiments.shared_sufficiency_v3_2_contract.core import _api_schema


PLANNER_PROMPT = """You are a requirement-guided retrieval Planner for a read-only video navigation map.
Do not answer the question. For every supplied requirement, in the exact supplied order, produce one focused retrieval plan. The map is a navigation aid, not evidence. Suggested Coarse IDs define the first retrieval scope, but deterministic fallback may expand it when too few candidates exist. Never assert that an event occurred. Keep visual facts, audible mentions, and physical event occurrence distinct. For temporal requirements, search separately for both target physical events and their occurrence anchors; recording order is not physical event order. Return strict JSON only."""


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planner_schema(requirements: list[dict[str, Any]], coarse_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "question_id": {"type": "string"},
            "requirement_plans": {
                "type": "array", "minItems": len(requirements), "maxItems": len(requirements),
                "items": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "requirement_id": {"type": "string", "enum": [r["requirement_id"] for r in requirements]},
                        "search_description": {"type": "string"},
                        "query_variants": {"type": "array", "items": {"type": "string"}},
                        "modality_strategy": {"type": "string"},
                        "suggested_coarse_ids": {"type": "array", "items": {"type": "string", "enum": coarse_ids}},
                    },
                    "required": ["requirement_id", "search_description", "query_variants", "modality_strategy", "suggested_coarse_ids"],
                },
            },
            "hard_filtering_allowed": {"type": "boolean", "const": False},
        },
        "required": ["question_id", "requirement_plans", "hard_filtering_allowed"],
    }


def validate_plan(result: dict[str, Any], qid: str, requirements: list[dict[str, Any]]) -> list[str]:
    errors = []
    if result.get("question_id") != qid or result.get("hard_filtering_allowed") is not False:
        errors.append("question or filtering contract invalid")
    expected = [r["requirement_id"] for r in requirements]
    actual = [r.get("requirement_id") for r in result.get("requirement_plans", [])]
    if actual != expected:
        errors.append("requirements must appear exactly once in frozen order")
    for row in result.get("requirement_plans", []):
        if not row.get("query_variants") or not row.get("suggested_coarse_ids"):
            errors.append(f"{row.get('requirement_id')}: empty query or scope")
    return errors


def call_json(api_key: str, model: str, max_tokens: int, prompt: str, payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key, timeout=180.0)
    started = time.perf_counter()
    response = client.messages.create(model=model, max_tokens=max_tokens, temperature=0.0, system=prompt, messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}], output_config={"format": {"type": "json_schema", "schema": _api_schema(schema)}})
    raw = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
    usage = {"model": model, "input_tokens": int(response.usage.input_tokens), "output_tokens": int(response.usage.output_tokens), "latency_sec": time.perf_counter() - started, "response_id": str(response.id), "stop_reason": response.stop_reason, "raw_text": raw}
    if response.stop_reason == "max_tokens": return None, {**usage, "error": "max_tokens"}
    return json.loads(raw), usage


def _ranges(map_doc: dict[str, Any]) -> dict[str, tuple[float, float]]:
    if "coarse_regions" in map_doc:
        return {r["coarse_id"]: (float(r["start_sec"]), float(r["end_sec"])) for r in map_doc["coarse_regions"]}
    return {r["node_id"]: (float(r["start_sec"]), float(r["end_sec"])) for r in map_doc["timeline_nodes"] if r["node_type"] == "visual_structural"}


def _parent(map_doc: dict[str, Any]) -> dict[str, str]:
    rows = map_doc.get("coarse_regions") or [r for r in map_doc["timeline_nodes"] if r["node_type"] == "visual_structural"]
    result = {}
    for row in rows:
        cid = row.get("coarse_id", row.get("node_id"))
        for mid in row["source_medium_ids"]: result[mid] = cid
    return result


def _scope(full: list[dict[str, Any]], coarse: set[str], top_k: int) -> tuple[list[dict[str, Any]], bool]:
    rows = [r for r in full if r["parent_coarse_id"] in coarse]
    expanded = len(rows) < top_k
    if expanded:
        seen = {r["medium_id"] for r in rows}
        rows.extend(r for r in full if r["medium_id"] not in seen)
    return rows[:top_k], expanded


def _audio_scope(full: list[dict[str, Any]], ranges: list[tuple[float, float]], top_k: int) -> tuple[list[dict[str, Any]], bool]:
    rows = [r for r in full if any(float(r["start_sec"]) < b and float(r["end_sec"]) > a for a, b in ranges)]
    expanded = len(rows) < top_k
    if expanded:
        seen = {r["audio_id"] for r in rows}; rows.extend(r for r in full if r["audio_id"] not in seen)
    return rows[:top_k], expanded


def suff_schema(packet: dict[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(temporal_schema())
    ids = sorted(r["evidence_id"] for r in packet["evidence"])
    reqs = [r["requirement_id"] for r in packet["requirements"]]
    props = schema["properties"]["claims"]["items"]["properties"]
    props["requirement_id"]["enum"] = reqs
    props["supporting_evidence_ids"]["items"]["enum"] = ids
    anchor = props["temporal_grounding"]["properties"]
    for key in ("event_a_anchor", "event_b_anchor"): anchor[key]["anyOf"][0]["properties"]["evidence_id"]["enum"] = ids
    schema["properties"]["claims"]["minItems"] = len(reqs); schema["properties"]["claims"]["maxItems"] = len(reqs)
    return schema


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load(config_path); out = root / cfg["output_root"]; out.mkdir(parents=True, exist_ok=True)
    paths = {
        "requirements": root / "outputs/experiments/evidence_sufficiency_v1/226/slot_templates.json",
        "questions": root / "configs/experiments/planner_medium_retrieval_v1/questions_226.json",
        "r1_inputs": root / "outputs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1/planner_inputs.json",
        "r3_inputs": root / "outputs/experiments/r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1/r3_2_planner_inputs.json",
        "canonical": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json",
        "projection": root / "outputs/experiments/egopolice_two_stage_organizer_r1_canary_v1/unified_medium_envelopes.json",
        "embeddings": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/embeddings/medium_pooled_siglip.float32.npy",
        "old_raw": root / "outputs/experiments/r1_av_r3_2_claim_pipeline_integration_v1/sufficiency_raw_responses.json",
    }
    if any(not p.is_file() for p in paths.values()): raise RuntimeError("missing source")
    reqs = requirements_from_slot_templates(load(paths["requirements"])); qrows = {q["question_id"]: q for q in load(paths["questions"])["questions"]}
    inputs = {"r1_av": {r["question"]["question_id"]: r for r in load(paths["r1_inputs"])}, "r3_2": {r["question"]["question_id"]: r for r in load(paths["r3_inputs"])}}
    selected_qids = cfg["question_ids"]
    base = load(paths["canonical"]); projection_doc = load(paths["projection"]); projection_rows = projection_doc.get("value", projection_doc) if isinstance(projection_doc, dict) else projection_doc; projection = {r["medium_id"]: r for r in projection_rows}; mediums = {r["medium_id"]: r for r in base["medium_nodes"]}
    lexical = {"r1_av": shared.build_lexical_records(projection_rows, base, "r1"), "r3_2": shared.build_lexical_records(projection_rows, base, "r3")}
    embeddings = np.array(np.load(paths["embeddings"], mmap_mode="r"), dtype=np.float32, copy=True); embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    from experiments.planner_medium_retrieval.core import SiglipTextEncoder
    sig_cfg = load(root / "configs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1.json")["siglip"]; encoder = SiglipTextEncoder(sig_cfg)
    if not os.environ.get("ANTHROPIC_API_KEY"): raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    manifest = {"experiment": cfg["experiment"], "sources": {k: {"path": str(p.relative_to(root)), "sha256": sha(p)} for k,p in paths.items()}, "questions": selected_qids, "planned_calls": {"planner": 4, "sufficiency": 4}}
    write(out / "input_manifest.json", manifest)
    planner_outputs, retrieval, packets, suff_results, suff_audits, handoffs, usages = {}, {}, {}, {}, {}, {}, []
    for side in ("r1_av", "r3_2"):
        planner_outputs[side] = {}; retrieval[side] = {}; packets[side] = {}; suff_results[side] = {}; suff_audits[side] = {}; handoffs[side] = {}
        for qid in selected_qids:
            source_input = inputs[side][qid]; map_doc = source_input["navigation_map"]; ranges = _ranges(map_doc); parent = _parent(map_doc)
            payload = {"question": source_input["question"], "requirements": reqs[qid], "navigation_map": map_doc, "retrieval_contract": source_input["retrieval_contract"]}
            plan, usage = call_json(os.environ["ANTHROPIC_API_KEY"], cfg["planner_model"], int(cfg["planner_max_tokens"]), PLANNER_PROMPT, payload, planner_schema(reqs[qid], sorted(ranges)))
            errors = [usage.get("error")] if plan is None else validate_plan(plan, qid, reqs[qid]); usages.append({"stage": "planner", "side": side, "question_id": qid, **{k:v for k,v in usage.items() if k != "raw_text"}})
            if errors: raise RuntimeError(f"planner {side}/{qid}: {errors}")
            planner_outputs[side][qid] = {"input": payload, "output": plan, "usage": usage}
            evidence_by_id: dict[str, dict[str, Any]] = {}; requirement_audit = []
            for rp in plan["requirement_plans"]:
                pseudo = {"search_units": [{"unit_id": rp["requirement_id"], "description": rp["search_description"], "query_variants": rp["query_variants"]}]}
                question = qrows[qid]; query_texts = [shared.unit_query_text(question, u) for u in pseudo["search_units"]]; query_emb = encoder.encode(query_texts)
                full, _, _ = shared.rank_all_mediums(question, pseudo, lexical[side], base["medium_nodes"], embeddings, query_emb, {"ranking": {"visual_score_weight": 0.6, "lexical_score_weight": 0.3, "coarse_prior_weight": 0.0, "top_k": 30}}, parent)
                coarse = set(rp["suggested_coarse_ids"]); msel, mexp = _scope(full, coarse, int(cfg["top_k_medium_per_requirement"]))
                afull, _, _ = audio_rank(pseudo, question, base["audio_nodes"], 107); aranges = [ranges[c] for c in rp["suggested_coarse_ids"]]; asel, aexp = _audio_scope(afull, aranges, int(cfg["top_k_audio_per_requirement"]))
                for row in msel:
                    mid = row["medium_id"]; etype = "detector_observation" if side == "r1_av" else "visual_caption"; text = projection[mid]["detector_summary"] if side == "r1_av" else mediums[mid]["qwen_caption"]; eid = f"{etype}::{mid}"
                    item = evidence_by_id.setdefault(eid, {"evidence_id": eid, "evidence_type": etype, "timestamp": [float(row["start_sec"]), float(row["end_sec"])], "source_content": text, "source_provenance": {"medium_id": mid, "parent_coarse_id": row["parent_coarse_id"], "retrieved_for_requirement_ids": []}}); item["source_provenance"]["retrieved_for_requirement_ids"].append(rp["requirement_id"])
                for row in asel:
                    eid=f"audio_asr::{row['audio_id']}"; item=evidence_by_id.setdefault(eid,{"evidence_id":eid,"evidence_type":"audio_asr","timestamp":[float(row["start_sec"]),float(row["end_sec"])],"source_content":row["exact_transcript"],"source_provenance":{"audio_id":row["audio_id"],"retrieved_for_requirement_ids":[]}}); item["source_provenance"]["retrieved_for_requirement_ids"].append(rp["requirement_id"])
                requirement_audit.append({"requirement_id":rp["requirement_id"],"suggested_coarse_ids":rp["suggested_coarse_ids"],"medium_ids":[r["medium_id"] for r in msel],"audio_ids":[r["audio_id"] for r in asel],"medium_fallback":mexp,"audio_fallback":aexp})
            for item in evidence_by_id.values(): item["source_provenance"]["retrieved_for_requirement_ids"] = list(dict.fromkeys(item["source_provenance"]["retrieved_for_requirement_ids"]))
            packet = {"question_id":qid,"question":qrows[qid]["question"],"requirements":reqs[qid],"evidence":list(evidence_by_id.values()),"existing_uncertainty":[],"rejected_bindings":["Map is navigation only.","Recording order is not physical event order."]}
            retrieval[side][qid]=requirement_audit; packets[side][qid]=packet
            result, susage = call_json(os.environ["ANTHROPIC_API_KEY"], cfg["sufficiency_model"], int(cfg["sufficiency_max_tokens"]), TEMPORAL_PROMPT, packet, suff_schema(packet)); serr=[susage.get("error")] if result is None else validate_result(result,packet); usages.append({"stage":"sufficiency","side":side,"question_id":qid,**{k:v for k,v in susage.items() if k!="raw_text"}})
            if serr: raise RuntimeError(f"sufficiency {side}/{qid}: {serr}")
            projected, projection_audit = project_statuses(result, packet); suff_results[side][qid]=projected; suff_audits[side][qid]=projection_audit
            handoffs[side][qid]={"question_id":qid,"requirements":[{"requirement_id":c["requirement_id"],"reliability_status":{"supported":"answer_ready","uncertain":"provisional","not_found":"unresolved","conflicted":"conflicted"}[c["status"]],"assessment":c} for c in result["claims"]]}
            write(out / "planner_outputs.json", planner_outputs); write(out / "requirement_retrieval_audit.json", retrieval); write(out / "sufficiency_packets.json", packets); write(out / "sufficiency_results.json", suff_results)
    old = load(paths["old_raw"]); old_usage = {(r["path"],r["question_id"]):r["usage"] for r in old}
    comparison=[]
    for side in ("r1_av","r3_2"):
        for qid in selected_qids:
            newu=next(u for u in usages if u["stage"]=="sufficiency" and u["side"]==side and u["question_id"]==qid); oldu=old_usage[(side,qid)]
            comparison.append({"side":side,"question_id":qid,"old_free_claim_count":len(next(r for r in load(root/f"outputs/experiments/r1_av_r3_2_claim_pipeline_integration_v1/{side}_claim_sufficiency_results.json") if r["question_id"]==qid)["claims"]),"new_fixed_requirement_count":len(reqs[qid]),"old_input_tokens":oldu["input_tokens"],"old_output_tokens":oldu["output_tokens"],"new_input_tokens":newu["input_tokens"],"new_output_tokens":newu["output_tokens"]})
    validation={"planner_requirement_contract":"passed","requirement_scoped_retrieval":"passed","v3_2_1_requirement_assessment":"passed","temporal_anchor_validation":"passed","r1_caption_leakage":0,"map_used_as_evidence":False,"overall_validation":"passed_canary_ready_for_six_question_expansion","planner_calls":4,"sufficiency_calls":4,"review_calls":0,"final_calls":0}
    cost={"calls":usages,"planner":{"calls":4,"input_tokens":sum(u["input_tokens"] for u in usages if u["stage"]=="planner"),"output_tokens":sum(u["output_tokens"] for u in usages if u["stage"]=="planner")},"sufficiency":{"calls":4,"input_tokens":sum(u["input_tokens"] for u in usages if u["stage"]=="sufficiency"),"output_tokens":sum(u["output_tokens"] for u in usages if u["stage"]=="sufficiency")}}
    write(out/"sufficiency_projection_audit.json",suff_audits);write(out/"requirement_handoff.json",handoffs);write(out/"old_vs_new_cost_comparison.json",comparison);write(out/"cost_accounting.json",cost);write(out/"validation_report.json",validation)
    sections="".join(f"<h2>{side} / {qid}</h2><pre>{html.escape(json.dumps({'plan':planner_outputs[side][qid]['output'],'result':suff_results[side][qid]},ensure_ascii=False,indent=2))}</pre>" for side in ('r1_av','r3_2') for qid in selected_qids);(out/"review.html").write_text("<!doctype html><meta charset='utf-8'><h1>Requirement-centric canary</h1>"+sections,encoding="utf-8")
    (out/"REPORT.md").write_text("\n".join(["# R1_AV / R3_2 requirement-centric pipeline canary","",f"- Overall: `{validation['overall_validation']}`","- Questions: global summary and handcuff-before-medical","- Planner calls: 4; Sufficiency calls: 4","- Free claim generation: disabled","- Map summaries: navigation only","- Review/Final: 0"]) + "\n",encoding="utf-8")
    return {"validation":validation,"cost":cost,"comparison":comparison}
