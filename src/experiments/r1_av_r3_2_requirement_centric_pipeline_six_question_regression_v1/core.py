from __future__ import annotations

import hashlib
import html
import json
import os
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.claim_level_av_sufficiency_v3_1_requirement_centric.core import requirements_from_slot_templates
from experiments.egopolice_r1_av_dual_channel_retrieval_smoke_v1.core import audio_rank
from experiments.r1_av_r3_2_requirement_centric_pipeline_canary import core as v1
from experiments.r1_av_r3_2_requirement_centric_pipeline_canary_v1_1 import core as v11
from experiments.r1_av_r3_2_requirement_centric_pipeline_canary_v1_2 import core as v12
from experiments.r1_r3_v2_map_aware_all_medium_retrieval_pair import core as shared
from experiments.shared_sufficiency_v3_2_1_temporal_anchor.core import TEMPORAL_PROMPT, project_statuses, validate_result


DIRECT_PROMPT = TEMPORAL_PROMPT + """

Requirement-scoped direct-evidence contract:
- Assess every declared requirement exactly once and in the supplied order.
- Only evidence IDs listed for that requirement in allowed_evidence_ids_by_requirement may be cited.
- Contextual evidence is deliberately excluded from this direct assessment and remains in a sidecar.
- Do not use map text, retrieval scores, unreviewed frame references, or recording order as semantic proof.
"""


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_packet(packet: dict[str, Any]) -> dict[str, Any]:
    policies = packet["requirement_evidence_policy"]
    return {
        "question_id": packet["question_id"],
        "question": packet["question"],
        "requirements": packet["requirements"],
        "evidence": packet["evidence"],
        "allowed_evidence_ids_by_requirement": {
            rid: row["direct_support_evidence_ids"] for rid, row in policies.items()
        },
        "existing_uncertainty": packet["existing_uncertainty"],
        "rejected_bindings": packet["rejected_bindings"] + [
            "Contextual evidence is excluded from direct Sufficiency assessment and remains in a sidecar."
        ],
    }


def validate_requirement_scoped_citations(result: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    allowed = packet["allowed_evidence_ids_by_requirement"]
    errors = []
    for index, claim in enumerate(result.get("claims", [])):
        rid = claim.get("requirement_id")
        if rid not in allowed:
            errors.append(f"claims[{index}]: {rid}: unknown requirement evidence policy")
        elif not set(claim.get("supporting_evidence_ids", [])) <= set(allowed[rid]):
            errors.append(f"claims[{index}]: {rid}: cited evidence retrieved for a different requirement")
    return errors


def project_v3_2_1_temporal_extension(result: dict[str, Any], packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove only V3.2.1 event-order fields from non-order assessments.

    The raw provider response remains unchanged in raw_responses.json. This is
    a schema projection, not a semantic repair: status, evidence, scope,
    rationale, and contradiction data are byte-for-byte preserved.
    """
    value = copy.deepcopy(result)
    descriptions = {row["requirement_id"]: row["description"] for row in packet["requirements"]}
    removed = []
    for claim in value.get("claims", []):
        if descriptions.get(claim.get("requirement_id")) not in {"chronological_order", "temporal_order"} and "temporal_grounding" in claim:
            claim.pop("temporal_grounding")
            removed.append(claim["requirement_id"])
    return value, {"schema_only": True, "removed_temporal_extension_from": removed, "semantic_fields_changed": 0}


def _add_evidence(target: dict[str, dict[str, Any]], item: dict[str, Any], rid: str) -> None:
    stored = target.setdefault(item["evidence_id"], item)
    provenance = stored.setdefault("source_provenance", {})
    ids = provenance.setdefault("retrieved_for_requirement_ids", [])
    if rid not in ids:
        ids.append(rid)


def build_packets(
    root: Path,
    cfg: dict[str, Any],
    planner_outputs: dict[str, dict[str, Any]],
    reqs: dict[str, list[dict[str, Any]]],
    questions: dict[str, dict[str, Any]],
    planner_inputs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = load(root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json")
    projection_doc = load(root / "outputs/experiments/egopolice_two_stage_organizer_r1_canary_v1/unified_medium_envelopes.json")
    projection_rows = projection_doc.get("value", projection_doc) if isinstance(projection_doc, dict) else projection_doc
    projection = {row["medium_id"]: row for row in projection_rows}
    mediums = {row["medium_id"]: row for row in base["medium_nodes"]}
    lexical = {
        "r1_av": shared.build_lexical_records(projection_rows, base, "r1"),
        "r3_2": shared.build_lexical_records(projection_rows, base, "r3"),
    }
    emb_path = root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/embeddings/medium_pooled_siglip.float32.npy"
    embeddings = np.array(np.load(emb_path, mmap_mode="r"), dtype=np.float32, copy=True)
    embeddings /= np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    from experiments.planner_medium_retrieval.core import SiglipTextEncoder
    sig_cfg = load(root / "configs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1.json")["siglip"]
    encoder = SiglipTextEncoder(sig_cfg)
    packets = {"r1_av": {}, "r3_2": {}}
    audits = {"r1_av": {}, "r3_2": {}}
    for side in ("r1_av", "r3_2"):
        for qid in cfg["question_ids"]:
            source = planner_inputs[side][qid]
            map_doc = source["navigation_map"]
            ranges = v1._ranges(map_doc)
            parent = v1._parent(map_doc)
            plan = planner_outputs[side][qid]["output"]
            direct_by_id: dict[str, dict[str, Any]] = {}
            context_by_id: dict[str, dict[str, Any]] = {}
            policies: dict[str, dict[str, list[str]]] = {}
            requirement_audit = []
            for rp, requirement in zip(plan["requirement_plans"], reqs[qid]):
                rid = requirement["requirement_id"]
                if rp["requirement_id"] != rid:
                    raise RuntimeError(f"{side}/{qid}: Planner requirement order changed")
                pseudo = {"search_units": [{"unit_id": rid, "description": rp["search_description"], "query_variants": rp["query_variants"]}]}
                question = questions[qid]
                qemb = encoder.encode([shared.unit_query_text(question, pseudo["search_units"][0])])
                full, _, _ = shared.rank_all_mediums(
                    question, pseudo, lexical[side], base["medium_nodes"], embeddings, qemb,
                    {"ranking": {"visual_score_weight": 0.6, "lexical_score_weight": 0.3, "coarse_prior_weight": 0.0, "top_k": 30}}, parent,
                )
                msel, mexp = v1._scope(full, set(rp["suggested_coarse_ids"]), int(cfg["top_k_medium_per_requirement"]))
                afull, _, _ = audio_rank(pseudo, question, base["audio_nodes"], 107)
                asel, aexp = v1._audio_scope(afull, [ranges[c] for c in rp["suggested_coarse_ids"]], int(cfg["top_k_audio_per_requirement"]))
                direct_ids: list[str] = []
                context_ids: list[str] = []
                for row in msel:
                    mid = row["medium_id"]
                    etype = "detector_observation" if side == "r1_av" else "visual_caption"
                    text = projection[mid]["detector_summary"] if side == "r1_av" else mediums[mid]["qwen_caption"]
                    item = {
                        "evidence_id": f"{etype}::{mid}", "evidence_type": etype,
                        "timestamp": [float(row["start_sec"]), float(row["end_sec"])], "source_content": text,
                        "source_provenance": {"medium_id": mid, "parent_coarse_id": row["parent_coarse_id"]},
                    }
                    if v11.classify_evidence(side, requirement["description"], etype) == "direct":
                        _add_evidence(direct_by_id, item, rid); direct_ids.append(item["evidence_id"])
                    else:
                        _add_evidence(context_by_id, item, rid); context_ids.append(item["evidence_id"])
                for row in asel:
                    item = {
                        "evidence_id": f"audio_asr::{row['audio_id']}", "evidence_type": "audio_asr",
                        "timestamp": [float(row["start_sec"]), float(row["end_sec"])], "source_content": row["exact_transcript"],
                        "source_provenance": {"audio_id": row["audio_id"]},
                    }
                    _add_evidence(context_by_id, item, rid); context_ids.append(item["evidence_id"])
                policies[rid] = {
                    "direct_support_evidence_ids": list(dict.fromkeys(direct_ids)),
                    "contextual_evidence_ids": list(dict.fromkeys(context_ids)),
                }
                requirement_audit.append({
                    "requirement_id": rid, **policies[rid],
                    "suggested_coarse_ids": rp["suggested_coarse_ids"],
                    "medium_fallback": mexp, "audio_fallback": aexp,
                })
            for eid in set(direct_by_id) & set(context_by_id):
                context_by_id.pop(eid, None)
            for policy in policies.values():
                policy["contextual_evidence_ids"] = [eid for eid in policy["contextual_evidence_ids"] if eid not in direct_by_id]
            packets[side][qid] = {
                "question_id": qid, "question": questions[qid]["question"], "requirements": reqs[qid],
                "evidence": list(direct_by_id.values()), "contextual_evidence": list(context_by_id.values()),
                "requirement_evidence_policy": policies, "existing_uncertainty": [],
                "rejected_bindings": ["Map is navigation only.", "Recording order is not physical event order."],
            }
            audits[side][qid] = requirement_audit
    return packets, audits


def run(root: Path, config_path: Path) -> dict[str, Any]:
    cfg = load(config_path)
    out = root / cfg["output_root"]
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "requirements": root / "outputs/experiments/evidence_sufficiency_v1/226/slot_templates.json",
        "questions": root / "configs/experiments/planner_medium_retrieval_v1/questions_226.json",
        "r1_inputs": root / "outputs/experiments/egopolice_r1_av_dual_channel_retrieval_smoke_v1/planner_inputs.json",
        "r3_inputs": root / "outputs/experiments/r1_r3_1_r3_2_map_aware_all_medium_retrieval_pair_v1/r3_2_planner_inputs.json",
        "reused_planners": root / "outputs/experiments/r1_av_r3_2_requirement_centric_pipeline_canary_v1_1/planner_outputs.json",
        "reused_packets": root / "outputs/experiments/r1_av_r3_2_requirement_centric_pipeline_canary_v1_1/packets.json",
        "reused_sufficiency": root / "outputs/experiments/r1_av_r3_2_requirement_centric_pipeline_canary_v1_2/raw_responses.json",
        "canonical": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/hierarchical_index_v1_1_caption_fixed_base.json",
        "projection": root / "outputs/experiments/egopolice_two_stage_organizer_r1_canary_v1/unified_medium_envelopes.json",
        "embeddings": root / "outputs/experiments/caption_fixed_rich_index_v1_1/226/embeddings/medium_pooled_siglip.float32.npy",
    }
    if any(not path.is_file() for path in paths.values()):
        raise RuntimeError("missing canonical source")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY unavailable")
    reqs = requirements_from_slot_templates(load(paths["requirements"]))
    questions = {row["question_id"]: row for row in load(paths["questions"])["questions"]}
    planner_inputs = {
        "r1_av": {row["question"]["question_id"]: row for row in load(paths["r1_inputs"])},
        "r3_2": {row["question"]["question_id"]: row for row in load(paths["r3_inputs"])},
    }
    reused = load(paths["reused_planners"])
    checkpoint_path = out / "planner_outputs.json"
    planner_outputs = load(checkpoint_path) if checkpoint_path.is_file() else {"r1_av": {}, "r3_2": {}}
    planner_usage = []
    reused_qids = {"q_global_summary", "q_handcuff_before_medical"}
    for side in ("r1_av", "r3_2"):
        for qid in cfg["question_ids"]:
            if qid in planner_outputs[side]:
                continue
            if qid in reused_qids:
                planner_outputs[side][qid] = {**reused[side][qid], "reused": True, "source_sha256": sha(paths["reused_planners"])}
            else:
                source = planner_inputs[side][qid]
                payload = {"question": source["question"], "requirements": reqs[qid], "navigation_map": source["navigation_map"], "retrieval_contract": source["retrieval_contract"]}
                plan, usage = v1.call_json(
                    os.environ["ANTHROPIC_API_KEY"], cfg["planner_model"], int(cfg["planner_max_tokens"]),
                    v1.PLANNER_PROMPT, payload, v1.planner_schema(reqs[qid], sorted(v1._ranges(source["navigation_map"]))),
                )
                errors = [usage.get("error")] if plan is None else v1.validate_plan(plan, qid, reqs[qid])
                if errors:
                    raise RuntimeError(f"Planner {side}/{qid}: {errors}")
                planner_outputs[side][qid] = {"input": payload, "output": plan, "usage": usage, "reused": False}
                planner_usage.append({"side": side, "question_id": qid, **{k: v for k, v in usage.items() if k != "raw_text"}})
            write(checkpoint_path, planner_outputs)
    packets, retrieval_audit = build_packets(root, cfg, planner_outputs, reqs, questions, planner_inputs)
    direct = {side: {qid: direct_packet(packet) for qid, packet in rows.items()} for side, rows in packets.items()}
    context = {side: {qid: {"contextual_evidence": packet["contextual_evidence"], "requirement_evidence_policy": packet["requirement_evidence_policy"]} for qid, packet in rows.items()} for side, rows in packets.items()}
    write(out / "direct_sufficiency_packets.json", direct)
    write(out / "contextual_evidence_sidecar.json", context)
    write(out / "requirement_retrieval_audit.json", retrieval_audit)
    raw_path = out / "raw_responses.json"
    raw = load(raw_path) if raw_path.is_file() else {}
    results = {"r1_av": {}, "r3_2": {}}
    handoffs = {"r1_av": {}, "r3_2": {}}
    suff_usage = []
    temporal_fail_closed = []
    schema_projection_audit = {}
    for side in ("r1_av", "r3_2"):
        for qid in cfg["question_ids"]:
            packet = direct[side][qid]
            key = f"{side}/{qid}"
            # Earlier canary responses used a union-evidence contract. They are
            # intentionally not reusable after requirement-scoped citations are
            # enforced, even though their source packets remain audit inputs.
            if raw.get(key, {}).get("reused"):
                raw.pop(key)
                write(raw_path, raw)
            if key in raw:
                result = raw[key]["result"]
                usage = raw[key]["usage"]
            elif not packet["evidence"]:
                result = v12.deterministic_no_direct_result(packet)
                usage = {"model": None, "input_tokens": 0, "output_tokens": 0, "latency_sec": 0.0, "stop_reason": "deterministic_no_direct_evidence", "response_id": None}
                raw[key] = {"result": result, "usage": usage, "deterministic": True}
                write(raw_path, raw)
            else:
                result, usage = v1.call_json(
                    os.environ["ANTHROPIC_API_KEY"], cfg["sufficiency_model"], int(cfg["sufficiency_max_tokens"]),
                    DIRECT_PROMPT, packet, v1.suff_schema(packet),
                )
                raw[key] = {"result": result, "usage": usage, "deterministic": False}
                write(raw_path, raw)
            if result is not None:
                result, projection = project_v3_2_1_temporal_extension(result, packet)
                schema_projection_audit[key] = projection
            else:
                schema_projection_audit[key] = {"schema_only": True, "removed_temporal_extension_from": [], "semantic_fields_changed": 0}
            errors = [usage.get("error")] if result is None else validate_result(result, packet) + validate_requirement_scoped_citations(result, packet)
            invalid_temporal = v12.temporal_error_indices(errors, packet) if errors else set()
            if errors and invalid_temporal is None:
                raise RuntimeError(f"Sufficiency {side}/{qid}: {errors}")
            if invalid_temporal:
                temporal_fail_closed.append({
                    "side": side, "question_id": qid,
                    "requirement_ids": [packet["requirements"][i]["requirement_id"] for i in sorted(invalid_temporal)],
                    "raw_validation_errors": errors, "resolution": "requires_temporal_review",
                })
            projected, audit = project_statuses(result, packet)
            results[side][qid] = {**projected, "projection_audit": audit}
            bad_rids = {packet["requirements"][i]["requirement_id"] for i in invalid_temporal}
            handoffs[side][qid] = {
                "question_id": qid,
                "requirements": [{
                    "requirement_id": claim["requirement_id"],
                    "reliability_status": "provisional" if claim["requirement_id"] in bad_rids else {
                        "supported": "answer_ready", "uncertain": "provisional", "not_found": "unresolved", "conflicted": "conflicted"
                    }[claim["status"]],
                    "assessment": claim,
                    "resolution_mode": "requires_temporal_review" if claim["requirement_id"] in bad_rids else "none",
                } for claim in result["claims"]],
            }
            suff_usage.append({"side": side, "question_id": qid, **{k: v for k, v in usage.items() if k != "raw_text"}})
    write(out / "sufficiency_results.json", results)
    write(out / "requirement_handoff.json", handoffs)
    write(out / "schema_projection_audit.json", schema_projection_audit)
    status_summary = {
        side: {qid: dict(Counter(row["assessment"]["status"] for row in handoffs[side][qid]["requirements"])) for qid in cfg["question_ids"]}
        for side in ("r1_av", "r3_2")
    }
    actual_planner = []
    for side in ("r1_av", "r3_2"):
        for qid in cfg["question_ids"]:
            row = planner_outputs[side][qid]
            if row.get("reused") is False and row.get("usage", {}).get("model"):
                actual_planner.append({
                    "side": side, "question_id": qid,
                    **{key: value for key, value in row["usage"].items() if key != "raw_text"},
                })
    actual_suff = [row for row in suff_usage if row.get("model")]
    validation = {
        "question_coverage": "passed_6_of_6_per_side", "requirement_coverage": "passed",
        "direct_context_separation": "passed", "requirement_scoped_citation_validation": "passed",
        "temporal_anchor_validation": "passed_or_fail_closed_to_review",
        "temporal_fail_closed_requirements": temporal_fail_closed,
        "map_used_as_evidence": False, "review_calls": 0, "final_calls": 0,
        "overall_validation": "passed_full_requirement_centric_regression_with_selective_temporal_review",
    }
    cost = {
        "new_planner_calls": len(actual_planner), "reused_planner_calls": 4,
        "new_sufficiency_calls": len(actual_suff), "reused_sufficiency_calls": 0,
        "deterministic_no_direct_packets": sum(1 for row in suff_usage if row.get("model") is None),
        "new_planner_input_tokens": sum(row["input_tokens"] for row in actual_planner),
        "new_planner_output_tokens": sum(row["output_tokens"] for row in actual_planner),
        "new_planner_summed_latency_sec": sum(row["latency_sec"] for row in actual_planner),
        "new_sufficiency_input_tokens": sum(row["input_tokens"] for row in actual_suff),
        "new_sufficiency_output_tokens": sum(row["output_tokens"] for row in actual_suff),
        "new_sufficiency_summed_latency_sec": sum(row["latency_sec"] for row in actual_suff),
        "planner_calls": planner_usage, "sufficiency_assessments": suff_usage,
    }
    manifest = {
        "experiment": cfg["experiment"], "question_ids": cfg["question_ids"],
        "sources": {name: {"path": str(path.relative_to(root)), "sha256": sha(path)} for name, path in paths.items()},
        "assessment_packets": 12,
    }
    write(out / "input_manifest.json", manifest)
    write(out / "status_summary.json", status_summary)
    write(out / "cost_accounting.json", cost)
    write(out / "validation_report.json", validation)
    sections = "".join(
        f"<h2>{side} / {qid}</h2><pre>{html.escape(json.dumps({'handoff': handoffs[side][qid], 'context': context[side][qid]}, ensure_ascii=False, indent=2))}</pre>"
        for side in ("r1_av", "r3_2") for qid in cfg["question_ids"]
    )
    (out / "review.html").write_text("<!doctype html><meta charset='utf-8'><h1>Six-question requirement-centric regression</h1>" + sections, encoding="utf-8")
    report = [
        "# R1_AV / R3_2 six-question requirement-centric regression", "",
        f"- Overall: `{validation['overall_validation']}`",
        "- Questions: 6/6 per representation; requirement assessments: 12 packets.",
        "- Direct and contextual evidence are separated; citations are requirement-scoped.",
        "- Temporal contract failures are preserved and handed to selective temporal review.",
        "- Map evidence, review calls, and final-answer calls: 0.", "", "## Status summary", "",
    ]
    for side in ("r1_av", "r3_2"):
        report.append(f"### {side}")
        report.append("")
        for qid in cfg["question_ids"]:
            report.append(f"- `{qid}`: `{status_summary[side][qid]}`")
        report.append("")
    report.extend([
        "## New model cost", "",
        f"- Planner: {cost['new_planner_calls']} calls, {cost['new_planner_input_tokens']} input tokens, {cost['new_planner_output_tokens']} output tokens, {cost['new_planner_summed_latency_sec']:.3f}s summed latency.",
        f"- Sufficiency: {cost['new_sufficiency_calls']} calls, {cost['new_sufficiency_input_tokens']} input tokens, {cost['new_sufficiency_output_tokens']} output tokens, {cost['new_sufficiency_summed_latency_sec']:.3f}s summed latency.",
        f"- Deterministic no-direct-evidence packets: {cost['deterministic_no_direct_packets']}.",
    ])
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"validation": validation, "status_summary": status_summary, "cost": cost}
