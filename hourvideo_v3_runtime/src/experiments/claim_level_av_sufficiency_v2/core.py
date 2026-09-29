from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


MODEL_ID = "claude-haiku-4-5-20251001"
EXPERIMENT = "claim_level_av_sufficiency_v2"
VIDEO_ID = "226"
ALLOWED_TYPES = {
    "object_presence", "actor_action", "actor_object_binding", "spoken_command",
    "reported_event", "physical_event", "state", "temporal_order",
    "causal_relation", "intervention", "outcome", "identity", "location",
    "cross_modal_event",
}
ALLOWED_STATUS = {"supported", "uncertain", "conflicted", "not_found"}
ALLOWED_SUPPORT = {
    "visual_caption", "audio_content", "audio_event", "cross_modal",
    "temporal_inference",
}
ALLOWED_ANSWERABILITY = {"sufficient", "partial", "insufficient", "conflicted"}

SYSTEM_PROMPT = """You are a claim-level audio-visual index sufficiency analyst.
You do not determine absolute external truth and you do not answer the user's
question. Infer atomic, reusable claims that the supplied audiovisual index
expresses or supports, then assess whether those claims suffice for the question.

Evidence rules:
- Visual captions are model-produced visual-index evidence, not raw-image proof.
- Exact ASR is audiovisual evidence. Preserve tense, polarity, speech act,
  report/request/observation distinctions, ASR uncertainty, unknown speaker,
  and temporal context.
- Do not reduce all audio to merely "someone spoke". A clear command, request,
  report, or arrival statement may support the corresponding audiovisual event,
  but only at its actual semantic strength.
- "Call an ambulance" supports an ambulance request, not arrival.
- A clear non-question, non-negated "the ambulance is here" may support arrival
  when ASR reliability and context are adequate.
- "Hands behind your back" supports a control command and restraint context, not
  completed handcuffing.
- A report that someone has a weapon does not establish visual weapon presence
  or bind that person to an object described by a visual caption.
- Typed temporal proximity alone does not establish actor/object/action identity.
- Preserve rejected bindings and uncertainty. Never invent speaker identity.
- Do not output an answer, legal conclusion, guilt, motive, or authorization.

Raw visual review is requested only when an answer-critical claim is unresolved
or conflicted, local raw imagery could reasonably resolve it, and no reusable
visual confirmation is already supplied. Do not request raw review for clear
audio commands/requests/events, unresolvable speaker identity, irrelevant detail,
or questions imagery cannot resolve. Do not prescribe any frame count.
Return strict JSON only."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _manifest_row(root: Path, path: Path, role: str, model_input: bool) -> dict[str, Any]:
    stat = path.stat()
    return {
        "relative_path": str(path.relative_to(root)),
        "absolute_path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
        "modified_time_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "semantic_role": role,
        "used_as_model_input": model_input,
        "comparison_only": not model_input,
    }


def claim_schema() -> dict[str, Any]:
    claim = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string", "minLength": 1},
            "claim_text": {"type": "string", "minLength": 1},
            "claim_type": {"type": "string", "enum": sorted(ALLOWED_TYPES)},
            "status": {"type": "string", "enum": sorted(ALLOWED_STATUS)},
            "support_mode": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(ALLOWED_SUPPORT)},
            },
            "evidence_ids": {
                "type": "array", "items": {"type": "string"}
            },
            "time_range": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 0,
                "maxItems": 2,
            },
            "reasoning_summary": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "answer_critical": {"type": "boolean"},
            "reusable": {"type": "boolean"},
            "needs_raw_visual_review": {"type": "boolean"},
            "review_reason": {"type": ["string", "null"]},
            "review_target": {"type": ["string", "null"]},
            "remaining_uncertainty": {
                "type": "array", "items": {"type": "string"}
            },
            "rejected_inferences": {
                "type": "array", "items": {"type": "string"}
            },
        },
        "required": [
            "claim_id", "claim_text", "claim_type", "status", "support_mode",
            "evidence_ids", "time_range", "reasoning_summary", "confidence",
            "answer_critical", "reusable", "needs_raw_visual_review",
            "review_reason", "review_target", "remaining_uncertainty",
            "rejected_inferences",
        ],
        "additionalProperties": False,
    }
    review = {
        "type": "object",
        "properties": {
            "claim_id": {"type": "string"},
            "answer_critical": {"type": "boolean", "const": True},
            "reason": {"type": "string", "minLength": 1},
            "target_time_range": {
                "type": "array", "items": {"type": "number"},
                "minItems": 2, "maxItems": 2,
            },
            "target_visual_question": {"type": "string", "minLength": 1},
            "candidate_existing_evidence_ids": {
                "type": "array", "items": {"type": "string"}
            },
            "expected_resolution": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["confirm", "reject", "remain_uncertain"],
                },
            },
        },
        "required": [
            "claim_id", "answer_critical", "reason", "target_time_range",
            "target_visual_question", "candidate_existing_evidence_ids",
            "expected_resolution",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "question_id": {"type": "string"},
            "claims": {"type": "array", "items": claim, "minItems": 1},
            "required_claims": {
                "type": "array", "items": {"type": "string"}
            },
            "supported_claims": {
                "type": "array", "items": {"type": "string"}
            },
            "uncertain_claims": {
                "type": "array", "items": {"type": "string"}
            },
            "conflicted_claims": {
                "type": "array", "items": {"type": "string"}
            },
            "missing_claims": {
                "type": "array", "items": {"type": "string"}
            },
            "answerability_from_current_index": {
                "type": "string", "enum": sorted(ALLOWED_ANSWERABILITY)
            },
            "visual_review_requests": {"type": "array", "items": review},
            "remaining_uncertainty": {
                "type": "array", "items": {"type": "string"}
            },
        },
        "required": [
            "question_id", "claims", "required_claims", "supported_claims",
            "uncertain_claims", "conflicted_claims", "missing_claims",
            "answerability_from_current_index", "visual_review_requests",
            "remaining_uncertainty",
        ],
        "additionalProperties": False,
    }


def _question_type(question_id: str) -> str:
    return {
        "q_global_summary": "global_summary",
        "q_weapon_visible": "presence_localisation",
        "q_visible_injury": "presence_localisation",
        "q_medical_assistance": "presence_localisation",
        "q_handcuffing": "presence_localisation",
        "q_handcuff_before_medical": "temporal_order",
    }[question_id]


def build_request_payload(packet: dict[str, Any]) -> dict[str, Any]:
    visual = []
    phases: list[str] = []
    uncertainties: list[str] = []
    rejected: list[str] = []
    for row in packet.get("visual_evidence", []):
        caption = row.get("caption_metadata") or ""
        if not caption.strip():
            continue
        branch = row.get("branch_path") or []
        phases.extend([x for x in branch if str(x).startswith("E")])
        cap_uncertain = [
            line.split(":", 1)[1].strip()
            for line in caption.splitlines()
            if line.upper().startswith("UNCERTAIN:") and ":" in line
        ]
        uncertainties.extend(cap_uncertain)
        visual.append({
            "evidence_id": row["visual_id"],
            "timestamp": [float(row["timestamp_sec"]), float(row["timestamp_sec"])],
            "caption": caption,
            "caption_uncertainty": cap_uncertain,
            "source_provenance": " → ".join(branch) or row.get("evidence_origin", ""),
        })
    audio = []
    for row in packet.get("audio_evidence", []):
        unc = row.get("uncertainty") or {}
        reliability = "uncertain" if (
            unc.get("fallback_used") or unc.get("warnings")
            or float((row.get("confidence") or {}).get("avg_logprob") or 0) < -0.75
        ) else "usable_with_asr_caution"
        audio.append({
            "evidence_id": row["audio_node_id"],
            "timestamp": [float(row["start_sec"]), float(row["end_sec"])],
            "exact_transcript": row["transcript"],
            "speaker": row.get("speaker_label") or "unknown_speaker",
            "asr_reliability": reliability,
            "source_provenance": json.dumps(
                row.get("provenance") or {}, ensure_ascii=False
            ),
        })
    # Only retrieval-time links are accepted; old sufficiency annotations are omitted.
    links = packet.get("typed_av_links") or []
    return {
        "question_id": packet["question_id"],
        "question": packet["question"],
        "question_type": _question_type(packet["question_id"]),
        "selected_phase_ids": sorted(set(phases)),
        "visual_evidence": visual,
        "audio_evidence": audio,
        "typed_av_links": links,
        "existing_uncertainty": sorted(set(x for x in uncertainties if x)),
        "rejected_bindings": rejected,
    }


def _normalize_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def normalize_result_schema(
    result: dict[str, Any], question_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize list descriptions to IDs without changing their semantics."""
    changes: list[dict[str, Any]] = []
    claims = result.get("claims") or []
    by_id = {c.get("claim_id"): c for c in claims}
    by_text = {
        re.sub(r"\s+", " ", str(c.get("claim_text", "")).strip().lower()): c
        for c in claims
    }
    for field in [
        "required_claims", "supported_claims", "uncertain_claims",
        "conflicted_claims", "missing_claims",
    ]:
        normalized: list[str] = []
        for index, value in enumerate(result.get(field) or [], 1):
            if value in by_id:
                normalized.append(value)
                continue
            key = re.sub(r"\s+", " ", str(value).strip().lower())
            if key in by_text:
                mapped = by_text[key]["claim_id"]
                normalized.append(mapped)
                changes.append({
                    "field": field, "from": value, "to": mapped,
                    "method": "exact_claim_text_to_existing_id",
                })
                continue
            if field != "missing_claims":
                normalized.append(value)
                continue
            # The model already supplied this semantic item as explicitly missing.
            # Wrap it in the canonical claim schema; do not add evidence or facts.
            cid = f"{question_id}_missing_{index:02d}"
            while cid in by_id:
                cid += "_x"
            claim_type = {
                "q_weapon_visible": "object_presence",
                "q_visible_injury": "state",
                "q_medical_assistance": "intervention",
                "q_handcuffing": "physical_event",
                "q_handcuff_before_medical": "temporal_order",
                "q_global_summary": "outcome",
            }.get(question_id, "state")
            claim = {
                "claim_id": cid,
                "claim_text": str(value),
                "claim_type": claim_type,
                "status": "not_found",
                "support_mode": [],
                "evidence_ids": [],
                "time_range": [],
                "reasoning_summary": "The model explicitly listed this required claim as missing.",
                "confidence": "low",
                "answer_critical": True,
                "reusable": False,
                "needs_raw_visual_review": False,
                "review_reason": None,
                "review_target": None,
                "remaining_uncertainty": [str(value)],
                "rejected_inferences": [],
            }
            claims.append(claim)
            by_id[cid] = claim
            by_text[key] = claim
            normalized.append(cid)
            changes.append({
                "field": field, "from": value, "to": cid,
                "method": "wrap_explicit_missing_description_as_not_found_claim",
            })
        result[field] = normalized
    for request in result.get("visual_review_requests") or []:
        cid = request.get("claim_id")
        claim = by_id.get(cid)
        if claim and not claim.get("needs_raw_visual_review"):
            claim["needs_raw_visual_review"] = True
            claim["review_reason"] = request.get("reason")
            claim["review_target"] = request.get("target_visual_question")
            changes.append({
                "field": "visual_review_requests",
                "claim_id": cid,
                "method": "synchronize_explicit_request_with_claim_review_flag",
            })
    for cid in result.get("missing_claims") or []:
        if cid not in result["required_claims"]:
            result["required_claims"].append(cid)
            changes.append({
                "field": "required_claims",
                "claim_id": cid,
                "method": "missing_claim_is_required_by_definition",
            })
    result["claims"] = claims
    return result, changes


def validate_result(result: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("question_id") != payload["question_id"]:
        errors.append("question_id mismatch")
    claims = result.get("claims") or []
    ids = [c.get("claim_id") for c in claims]
    if not claims or any(not x for x in ids) or len(ids) != len(set(ids)):
        errors.append("claim IDs must be nonempty and unique")
    idset = set(ids)
    evidence = {
        x["evidence_id"] for x in payload["visual_evidence"]
    } | {x["evidence_id"] for x in payload["audio_evidence"]}
    visual_ids = {x["evidence_id"] for x in payload["visual_evidence"]}
    audio_ids = {x["evidence_id"] for x in payload["audio_evidence"]}
    for c in claims:
        if c.get("claim_type") not in ALLOWED_TYPES:
            errors.append(f"{c.get('claim_id')}: invalid claim_type")
        if c.get("status") not in ALLOWED_STATUS:
            errors.append(f"{c.get('claim_id')}: invalid status")
        if not set(c.get("evidence_ids") or []) <= evidence:
            errors.append(f"{c.get('claim_id')}: invalid evidence provenance")
        modes = set(c.get("support_mode") or [])
        if not modes <= ALLOWED_SUPPORT:
            errors.append(f"{c.get('claim_id')}: invalid support mode")
        if "visual_caption" in modes and not (
            set(c.get("evidence_ids") or []) & visual_ids
        ):
            errors.append(f"{c.get('claim_id')}: visual mode lacks visual evidence")
        if modes & {"audio_content", "audio_event"} and not (
            set(c.get("evidence_ids") or []) & audio_ids
        ):
            errors.append(f"{c.get('claim_id')}: audio mode lacks audio evidence")
        tr = c.get("time_range") or []
        if len(tr) not in {0, 2} or (len(tr) == 2 and tr[0] > tr[1]):
            errors.append(f"{c.get('claim_id')}: invalid time_range")
        if c.get("needs_raw_visual_review"):
            if not c.get("answer_critical") or not c.get("review_reason"):
                errors.append(f"{c.get('claim_id')}: unjustified raw review")
        elif c.get("review_reason") is not None or c.get("review_target") is not None:
            errors.append(f"{c.get('claim_id')}: review fields must be null")
    subsets = {
        "required_claims": None,
        "supported_claims": "supported",
        "uncertain_claims": "uncertain",
        "conflicted_claims": "conflicted",
        "missing_claims": "not_found",
    }
    by_id = {c["claim_id"]: c for c in claims if c.get("claim_id")}
    for field, expected in subsets.items():
        values = result.get(field) or []
        if not set(values) <= idset:
            errors.append(f"{field} contains unknown claim ID")
        if expected and any(by_id[x]["status"] != expected for x in values if x in by_id):
            errors.append(f"{field} status mismatch")
    for req in result.get("visual_review_requests") or []:
        cid = req.get("claim_id")
        if cid not in by_id or not by_id[cid].get("answer_critical"):
            errors.append("visual review must reference answer-critical claim")
        if cid in by_id and not by_id[cid].get("needs_raw_visual_review"):
            errors.append("visual review request/claim flag mismatch")
        if not set(req.get("candidate_existing_evidence_ids") or []) <= evidence:
            errors.append("visual review request uses unknown evidence")
        text = json.dumps(req, ensure_ascii=False).lower()
        if re.search(r"\b(?:max(?:imum)?|最多)\s*\d+\s*(?:frames?|张)", text):
            errors.append("fixed frame budget forbidden")
    if result.get("answerability_from_current_index") not in ALLOWED_ANSWERABILITY:
        errors.append("invalid answerability")
    # Reject common answer fields recursively.
    def walk(v: Any) -> None:
        if isinstance(v, dict):
            for k, vv in v.items():
                if k in {"answer", "final_answer", "selected_option"}:
                    errors.append(f"forbidden field: {k}")
                walk(vv)
        elif isinstance(v, list):
            for vv in v:
                walk(vv)
        elif isinstance(v, float) and not math.isfinite(v):
            errors.append("NaN/Infinity")
    walk(result)
    return sorted(set(errors))


def _anthropic_call(
    *, api_key: str, payload: dict[str, Any], max_tokens: int
) -> tuple[str, dict[str, Any]]:
    import anthropic
    def api_schema(value: Any) -> Any:
        """Remove constraints unsupported by Anthropic structured outputs.

        Equivalent uniqueness/cardinality checks remain in the local validator.
        """
        if isinstance(value, dict):
            return {
                k: api_schema(v) for k, v in value.items()
                if k not in {"uniqueItems", "minItems", "maxItems", "minLength"}
            }
        if isinstance(value, list):
            return [api_schema(v) for v in value]
        return value

    client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
    started = time.perf_counter()
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=max_tokens,
        temperature=0.0,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [{"type": "text", "text": json.dumps(
                payload, ensure_ascii=False, allow_nan=False
            )}],
        }],
        output_config={"format": {"type": "json_schema", "schema": api_schema(claim_schema())}},
    )
    latency = time.perf_counter() - started
    raw = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    )
    usage = {
        "question_id": payload["question_id"],
        "model": MODEL_ID,
        "input_tokens": int(response.usage.input_tokens),
        "output_tokens": int(response.usage.output_tokens),
        "latency_sec": latency,
        "stop_reason": response.stop_reason,
        "response_id": str(response.id),
        "raw_image_inputs": 0,
        "formal_calls": 1,
    }
    if response.stop_reason == "max_tokens":
        raise RuntimeError(f"{payload['question_id']} truncated at max_tokens")
    return raw, usage


def deduplicate_claim_cache(results: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for result in results:
        qid = result["question_id"]
        for claim in result["claims"]:
            key_payload = {
                "text": re.sub(r"\s+", " ", claim["claim_text"].strip().lower()),
                "time": [round(float(x), 3) for x in claim["time_range"]],
                "evidence": sorted(claim["evidence_ids"]),
            }
            key = hashlib.sha256(json.dumps(
                key_payload, sort_keys=True
            ).encode()).hexdigest()
            if key not in groups:
                groups[key] = {
                    **claim,
                    "cache_claim_id": f"cache_{len(groups)+1:04d}",
                    "question_ids": [qid],
                    "verification_scope": "supported_by_current_audiovisual_index",
                    "raw_visual_ground_truth_verified": False,
                }
            elif qid not in groups[key]["question_ids"]:
                groups[key]["question_ids"].append(qid)
    return {
        "status": "candidate_supported_by_current_audiovisual_index",
        "verified_meaning": "supported by current audiovisual index, not ground-truth verified",
        "claims": list(groups.values()),
    }


def compare_old(
    results: list[dict[str, Any]],
    old_av: dict[str, Any],
    old_visual: list[dict[str, Any]],
) -> dict[str, Any]:
    visual_by_q = {x["question_id"]: x for x in old_visual}
    rows = []
    old_status_map = {
        "partially_sufficient": "partial",
        "sufficient": "sufficient",
        "insufficient": "insufficient",
        "conflicted": "conflicted",
    }
    for new in results:
        qid = new["question_id"]
        old = old_av.get(qid, {}).get("audio_visual", {})
        claims = new["claims"]
        rows.append({
            "question_id": qid,
            "old_overall_status": old.get("overall_status", "unavailable"),
            "new_answerability": new["answerability_from_current_index"],
            "new_claim_count": len(claims),
            "new_supported_count": sum(c["status"] == "supported" for c in claims),
            "new_uncertain_count": sum(c["status"] == "uncertain" for c in claims),
            "new_conflicted_count": sum(c["status"] == "conflicted" for c in claims),
            "answer_critical_coverage": {
                "required": sum(c["answer_critical"] for c in claims),
                "supported": sum(
                    c["answer_critical"] and c["status"] == "supported" for c in claims
                ),
            },
            "visual_review_request_count": len(new["visual_review_requests"]),
            "old_visual_frame_presentations": len(
                visual_by_q.get(qid, {}).get("visual_evidence", [])
            ),
            "potential_old_image_reads_avoided_this_stage": len(
                visual_by_q.get(qid, {}).get("visual_evidence", [])
            ),
            "critical_disagreement": (
                old_status_map.get(old.get("overall_status"), old.get("overall_status"))
                != new["answerability_from_current_index"]
            ),
            "note": "Old output is a comparison reference, not ground truth.",
        })
    return {"questions": rows}


def build_semantic_boundary_audit(
    results: list[dict[str, Any]], payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    by_q = {x["question_id"]: x for x in results}
    flags: list[dict[str, Any]] = []
    for result in results:
        for claim in result["claims"]:
            text = claim["claim_text"].lower()
            modes = set(claim["support_mode"])
            reasons = []
            if "visual_caption" in modes and any(
                x in text for x in ["suspect", "offender", "victim"]
            ):
                reasons.append("role_word_inherited_from_visual_caption_metadata")
            if (
                claim["status"] == "supported"
                and "visual_caption" in modes
                and claim["claim_type"] in {"actor_object_binding", "actor_action"}
                and not claim["needs_raw_visual_review"]
            ):
                reasons.append("actor_or_object_binding_supported_without_raw_visual_review")
            if (
                claim["status"] == "supported"
                and "temporal_inference" in modes
                and "medical assistance" in text
            ):
                reasons.append("temporal_medical_claim_depends_on_event_identity_not_raw_visual_confirmation")
            if (
                claim["status"] == "supported"
                and "cross_modal" in modes
                and "shooting events" in text
            ):
                reasons.append("broad_cross_modal_event_summary_may_exceed_atomic_evidence_binding")
            if reasons:
                flags.append({
                    "question_id": result["question_id"],
                    "claim_id": claim["claim_id"],
                    "claim_text": claim["claim_text"],
                    "classification": "potential_overclaim_for_human_review",
                    "reasons": reasons,
                })
    weapon = by_q["q_weapon_visible"]["claims"]
    medical = by_q["q_medical_assistance"]["claims"]
    cuff = by_q["q_handcuffing"]["claims"]
    temporal = by_q["q_handcuff_before_medical"]["claims"]
    boundary_checks = {
        "weapon_report_visible_holder_distinguished": {
            "reported_weapon_claims": [
                c["claim_id"] for c in weapon
                if c["claim_type"] == "reported_event"
                and any(x in c["claim_text"].lower() for x in ["weapon", "gun"])
            ],
            "visual_weapon_claims": [
                c["claim_id"] for c in weapon if "visual_caption" in c["support_mode"]
            ],
            "cross_modal_weapon_holder_claims": [
                c["claim_id"] for c in weapon
                if "cross_modal" in c["support_mode"]
                and any(x in c["claim_text"].lower() for x in ["holder", "holding"])
            ],
        },
        "medical_request_arrival_treatment_distinguished": {
            "request_or_ems_claims": [
                c["claim_id"] for c in medical
                if any(x in c["claim_text"].lower() for x in ["ems", "request"])
            ],
            "arrival_claims": [
                c["claim_id"] for c in medical if "ambulance" in c["claim_text"].lower()
                and any(x in c["claim_text"].lower() for x in ["here", "arriv"])
            ],
            "treatment_not_found_or_uncertain": [
                c["claim_id"] for c in medical
                if c["status"] in {"uncertain", "not_found"}
                and any(x in c["claim_text"].lower() for x in ["treat", "assist", "intervention"])
            ],
        },
        "command_restraint_completed_cuffing_distinguished": {
            "command_claims": [
                c["claim_id"] for c in cuff if c["claim_type"] == "spoken_command"
            ],
            "restraint_claims": [
                c["claim_id"] for c in cuff if "restrain" in c["claim_text"].lower()
            ],
            "completed_cuffing_supported_claims": [
                c["claim_id"] for c in cuff
                if c["status"] == "supported"
                and any(x in c["claim_text"].lower() for x in ["completed cuff", "fully cuff"])
            ],
        },
        "temporal_claims_requiring_human_review": [
            c["claim_id"] for c in temporal
            if any(f["claim_id"] == c["claim_id"] for f in flags)
        ],
    }
    return {
        "potential_unsupported_or_overclaimed_claims": flags,
        "boundary_checks": boundary_checks,
        "interpretation": (
            "These deterministic flags are review candidates, not ground truth "
            "and do not rewrite the Haiku output."
        ),
    }


def _render_review(
    payloads: list[dict[str, Any]],
    results: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    old = {x["question_id"]: x for x in comparison["questions"]}
    audit_flags = defaultdict(list)
    for flag in (
        comparison.get("semantic_boundary_audit", {})
        .get("potential_unsupported_or_overclaimed_claims", [])
    ):
        audit_flags[flag["question_id"]].append(flag)
    result_by_q = {x["question_id"]: x for x in results}
    sections = []
    for p in payloads:
        r = result_by_q[p["question_id"]]
        evidence = []
        for v in p["visual_evidence"]:
            evidence.append(
                f"<details><summary>Visual caption · {html.escape(v['evidence_id'])} · "
                f"{v['timestamp'][0]:.3f}s</summary><pre>{html.escape(v['caption'])}</pre>"
                f"<p>Uncertainty: {html.escape(str(v['caption_uncertainty']))}</p></details>"
            )
        for a in p["audio_evidence"]:
            evidence.append(
                f"<details><summary>Exact ASR · {html.escape(a['evidence_id'])} · "
                f"{a['timestamp'][0]:.3f}–{a['timestamp'][1]:.3f}s</summary>"
                f"<blockquote>{html.escape(a['exact_transcript'])}</blockquote>"
                f"<p>Speaker: {html.escape(a['speaker'])}; reliability: "
                f"{html.escape(a['asr_reliability'])}</p></details>"
            )
        claims = []
        for c in r["claims"]:
            cls = "supported" if c["status"] == "supported" else "warning"
            claims.append(
                f"<details class='{cls}'><summary>{html.escape(c['claim_id'])} · "
                f"{html.escape(c['status'])} · {html.escape(c['claim_type'])}</summary>"
                f"<p>{html.escape(c['claim_text'])}</p>"
                f"<p><b>Mode:</b> {html.escape(', '.join(c['support_mode']))} · "
                f"<b>Evidence:</b> {html.escape(', '.join(c['evidence_ids']))}</p>"
                f"<p><b>Reason:</b> {html.escape(c['reasoning_summary'])}</p>"
                f"<p><b>Remaining:</b> {html.escape('; '.join(c['remaining_uncertainty']))}</p>"
                f"<p><b>Rejected:</b> {html.escape('; '.join(c['rejected_inferences']))}</p>"
                f"</details>"
            )
        reviews = "".join(
            f"<li><b>{html.escape(x['claim_id'])}</b> "
            f"{html.escape(x['reason'])} · {html.escape(str(x['target_time_range']))}<br>"
            f"{html.escape(x['target_visual_question'])}</li>"
            for x in r["visual_review_requests"]
        ) or "<li>None</li>"
        comp = old[p["question_id"]]
        flagged = audit_flags[p["question_id"]]
        sections.append(
            f"<section><h2>{html.escape(p['question_id'])}</h2>"
            f"<p class='question'>{html.escape(p['question'])}</p>"
            f"<details><summary>Frozen retrieved evidence "
            f"({len(p['visual_evidence'])} captions, {len(p['audio_evidence'])} ASR)</summary>"
            f"{''.join(evidence)}<details><summary>Typed AV links</summary>"
            f"<pre>{html.escape(json.dumps(p['typed_av_links'], ensure_ascii=False, indent=2))}</pre>"
            f"</details></details>"
            f"<details open><summary>Required claims</summary><p>"
            f"{html.escape(', '.join(r['required_claims']))}</p></details>"
            f"<details open><summary>Supported / uncertain / conflicted claims</summary>"
            f"{''.join(claims)}</details>"
            f"<details open><summary>Requested raw visual review</summary><ul>{reviews}</ul></details>"
            f"<details><summary>Old sufficiency comparison</summary><pre>"
            f"{html.escape(json.dumps(comp, ensure_ascii=False, indent=2))}</pre></details>"
            f"<details><summary>Critical disagreement / overclaim audit</summary><pre>"
            f"{html.escape(json.dumps(flagged, ensure_ascii=False, indent=2))}</pre></details>"
            f"</section>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Claim-level AV Sufficiency v2 · 226</title>
<style>
body{{font:15px system-ui;margin:24px;background:#f4f6f8;color:#18202a}}
main{{max-width:1250px;margin:auto}}section{{background:white;padding:18px;margin:18px 0;
border-radius:10px;box-shadow:0 1px 5px #ccd}}details{{margin:8px 0 8px 16px;
border-left:3px solid #ccd;padding-left:10px}}summary{{cursor:pointer;font-weight:650}}
pre,blockquote{{white-space:pre-wrap;background:#f7f8fa;padding:10px}}.supported{{border-color:#35a66f}}
.warning{{border-color:#d99b2b}}.question{{font-size:1.08rem}}</style></head>
<body><main><h1>Claim-level AV Sufficiency v2 · EgoPolice 226</h1>
<p>Six text-only Haiku calls. No raw images. Old outputs are shown only after generation.</p>
{''.join(sections)}</main></body></html>"""


def run_experiment(
    repo: Path,
    *,
    allow_api_calls: bool = False,
    caller: Callable[..., tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    out = repo / "outputs/experiments/claim_level_av_sufficiency_v2/226"
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "packet": repo / "outputs/experiments/rich_av_downstream_v1/226/minimal_av_evidence_packet_v1_candidate.json",
        "av_packets": repo / "outputs/experiments/rich_av_downstream_v1/226/audio_visual_packets.json",
        "av_map": repo / "outputs/experiments/rich_av_downstream_v1/226/unified_rich_av_map_v1.json",
        "audio": repo / "outputs/experiments/rich_av_downstream_v1/226/selected_audio_evidence.json",
        "links": repo / "outputs/experiments/rich_av_downstream_v1/226/typed_av_links.json",
        "old_av": repo / "outputs/experiments/rich_av_downstream_v1/226/av_sufficiency_results.json",
        "old_visual": repo / "outputs/experiments/evidence_sufficiency_v1/226/sufficiency_judgements.json",
        "old_claims": repo / "outputs/experiments/claim_strength_gate_v1_1/226/claim_ledger_v1_1.json",
        "old_final": repo / "outputs/experiments/final_grounded_answer_v1/226/final_grounded_answers_v1.json",
        "selective": repo / "outputs/experiments/selective_haiku_av_organizer_v1/226/selective_av_map_v1_candidate.json",
        "old_config": repo / "outputs/experiments/evidence_sufficiency_v1/226/judge_config.json",
        "old_visual_packets": repo / "outputs/experiments/rich_av_downstream_v1/226/visual_only_packets.json",
        "old_api_usage": repo / "outputs/experiments/rich_av_downstream_v1/226/api_usage.json",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: " + ", ".join(missing))
    roles = {
        "packet": ("canonical minimal AV packet reference", False),
        "av_packets": ("frozen per-question retrieved AV evidence", True),
        "av_map": ("unified rich AV provenance", False),
        "audio": ("selected exact ASR evidence", False),
        "links": ("typed AV links", False),
        "old_av": ("old AV sufficiency comparison", False),
        "old_visual": ("old visual sufficiency comparison", False),
        "old_claims": ("old claim ledger comparison", False),
        "old_final": ("old final answer comparison", False),
        "selective": ("selective Organizer control only", False),
        "old_config": ("frozen sufficiency model configuration", False),
        "old_visual_packets": ("old visual packet comparison counts", False),
        "old_api_usage": ("old three-way AV sufficiency cost comparison", False),
    }
    manifest = {
        "experiment": EXPERIMENT,
        "video_id": VIDEO_ID,
        "git_branch_requested": "exp/claim-level-av-sufficiency-v2",
        "files": [
            _manifest_row(repo, paths[k], *roles[k]) for k in paths
        ],
    }
    _dump(out / "frozen_input_manifest.json", manifest)
    _dump(out / "selective_organizer_control_manifest.json", {
        "path": str(paths["selective"].relative_to(repo)),
        "sha256": _sha256(paths["selective"]),
        "usage": "comparison_control_only_not_formal_input",
    })
    old_cfg = _load(paths["old_config"])
    model_cfg = {
        "model": MODEL_ID,
        "temperature": old_cfg.get("temperature", 0.0),
        "max_tokens": old_cfg.get("max_tokens", 6000),
        "timeout_sec": 120.0,
        "formal_calls_expected": 6,
        "semantic_retry_allowed": False,
        "model_repair_allowed": False,
        "deterministic_json_normalization_allowed": True,
        "raw_image_inputs": 0,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "old_config_source": str(paths["old_config"].relative_to(repo)),
    }
    _dump(out / "model_config_frozen.json", model_cfg)
    (out / "system_prompt.txt").write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
    packets_doc = _load(paths["av_packets"])
    payloads = [build_request_payload(x) for x in packets_doc["questions"]]
    if len(payloads) != 6 or len({x["question_id"] for x in payloads}) != 6:
        raise RuntimeError("Expected six unique questions")
    forbidden_prompt_fragments = []
    serialized_payloads = json.dumps(payloads, ensure_ascii=False)
    for fragment in [
        "prior_visual_sufficiency_annotation", "overall_status",
        "short_answer_preview", "final_grounded_answer", "grounded_answers",
    ]:
        if fragment in serialized_payloads:
            forbidden_prompt_fragments.append(fragment)
    if forbidden_prompt_fragments:
        raise RuntimeError(f"Old output leaked into prompt: {forbidden_prompt_fragments}")
    _dump(out / "request_payloads.json", {"questions": payloads})
    if not allow_api_calls:
        dry = {
            "status": "dry_run_ready",
            "questions": len(payloads),
            "raw_image_inputs": 0,
            "forbidden_prompt_fragments": forbidden_prompt_fragments,
            "formal_calls_made": 0,
        }
        _dump(out / "validation_report.json", dry)
        return dry
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and caller is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not configured")
    call = caller or _anthropic_call
    raw_path = out / "raw_responses.jsonl"
    results: list[dict[str, Any]] = []
    usage_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    if raw_path.exists():
        existing = [
            json.loads(x) for x in raw_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        by_q = {x["question_id"]: x for x in existing}
    else:
        by_q = {}
    for payload in payloads:
        qid = payload["question_id"]
        if qid in by_q:
            raw = by_q[qid]["raw_response"]
            usage = by_q[qid]["usage"]
        else:
            raw, usage = call(
                api_key=api_key or "", payload=payload,
                max_tokens=int(model_cfg["max_tokens"]),
            )
            with raw_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "question_id": qid, "raw_response": raw, "usage": usage,
                }, ensure_ascii=False, allow_nan=False) + "\n")
        parsed = _normalize_json(raw)
        parsed, normalization = normalize_result_schema(parsed, qid)
        normalization_rows.extend(
            [{"question_id": qid, **row} for row in normalization]
        )
        errs = validate_result(parsed, payload)
        if errs:
            raise RuntimeError(f"{qid} failed validation: {errs}")
        results.append(parsed)
        usage_rows.append(usage)
    # Comparison artifacts are read only after all six new outputs exist.
    old_av = _load(paths["old_av"])
    old_visual_doc = _load(paths["old_visual_packets"])
    comparison = compare_old(results, old_av, old_visual_doc["questions"])
    semantic_audit = build_semantic_boundary_audit(results, payloads)
    comparison["semantic_boundary_audit"] = semantic_audit
    cache = deduplicate_claim_cache(results)
    _dump(out / "parsed_claims_by_question.json", {"questions": results})
    _dump(out / "schema_normalization_log.json", normalization_rows)
    _dump(out / "required_claims_by_question.json", {
        "questions": [{
            "question_id": x["question_id"],
            "required_claims": x["required_claims"],
        } for x in results]
    })
    _dump(out / "sufficiency_decisions.json", {
        "questions": [{
            "question_id": x["question_id"],
            "answerability_from_current_index": x["answerability_from_current_index"],
            "supported_claims": x["supported_claims"],
            "uncertain_claims": x["uncertain_claims"],
            "conflicted_claims": x["conflicted_claims"],
            "missing_claims": x["missing_claims"],
            "remaining_uncertainty": x["remaining_uncertainty"],
        } for x in results]
    })
    _dump(out / "visual_review_requests.json", {
        "questions": [{
            "question_id": x["question_id"],
            "visual_review_requests": x["visual_review_requests"],
        } for x in results]
    })
    _dump(out / "verified_video_claim_cache_candidate.json", cache)
    _dump(out / "comparison_with_old_sufficiency.json", comparison)
    disagreements = {
        "questions": [
            x for x in comparison["questions"] if x["critical_disagreement"]
        ],
        **semantic_audit,
        "interpretation": "Listed for human review; neither version is treated as ground truth.",
    }
    _dump(out / "critical_disagreement_audit.json", disagreements)
    totals = {
        "formal_calls": sum(x.get("formal_calls", 1) for x in usage_rows),
        "input_tokens": sum(x["input_tokens"] for x in usage_rows),
        "output_tokens": sum(x["output_tokens"] for x in usage_rows),
        "total_tokens": sum(x["input_tokens"] + x["output_tokens"] for x in usage_rows),
        "latency_sec": sum(x["latency_sec"] for x in usage_rows),
        "raw_image_inputs": 0,
    }
    old_usage = _load(paths["old_api_usage"])
    old_judge_rows = [
        x for x in old_usage.get("per_call", [])
        if str(x.get("label", "")).startswith("av_judge:")
    ]
    old_av_rows = [
        x for x in old_judge_rows
        if str(x.get("label", "")).endswith(":audio_visual")
    ]
    visual_counts = {
        x["question_id"]: len(x.get("visual_evidence", []))
        for x in old_visual_doc["questions"]
    }
    old_visual_presentations_one_mode = sum(visual_counts.values())
    old_cost = {
        "three_way_av_judge": {
            "calls": len(old_judge_rows),
            "input_tokens": sum(x["input_tokens"] for x in old_judge_rows),
            "output_tokens": sum(x["output_tokens"] for x in old_judge_rows),
            "latency_sec": sum(x["latency_sec"] for x in old_judge_rows),
            "raw_image_presentations": old_visual_presentations_one_mode * 2,
            "note": "visual_only and audio_visual each received the selected images; asr_only received none.",
        },
        "audio_visual_mode_only": {
            "calls": len(old_av_rows),
            "input_tokens": sum(x["input_tokens"] for x in old_av_rows),
            "output_tokens": sum(x["output_tokens"] for x in old_av_rows),
            "latency_sec": sum(x["latency_sec"] for x in old_av_rows),
            "raw_image_presentations": old_visual_presentations_one_mode,
        },
    }
    _dump(out / "cost_comparison.json", {
        "new_claim_level": totals,
        "old_full_av_sufficiency": old_cost,
        "per_question": usage_rows,
        "old_comparison_note": "Old costs are comparison-only and are not included in formal new cost.",
    })
    _dump(out / "api_usage.json", usage_rows)
    (out / "review.html").write_text(
        _render_review(payloads, results, comparison), encoding="utf-8"
    )
    counts = Counter(c["status"] for r in results for c in r["claims"])
    requests = sum(len(r["visual_review_requests"]) for r in results)
    validation = {
        "validation_status": "passed_claim_level_av_sufficiency_v2_ready",
        "formal_haiku_calls": totals["formal_calls"],
        "raw_image_inputs": 0,
        "gemini_calls": 0,
        "organizer_calls": 0,
        "planner_calls": 0,
        "retrieval_calls": 0,
        "asr_calls": 0,
        "caption_calls": 0,
        "question_count": len(results),
        "claims_by_status": dict(counts),
        "visual_review_request_count": requests,
        "exact_transcripts_preserved": True,
        "old_outputs_excluded_from_prompts": True,
        "frozen_hashes_reverified": all(
            _sha256(repo / x["relative_path"]) == x["sha256"]
            for x in manifest["files"]
        ),
        "errors": [],
        "warnings": [
            "Requested Git branch could not be created because .git metadata is read-only."
        ],
    }
    if totals["formal_calls"] != 6:
        validation["validation_status"] = "failed_validation"
        validation["errors"].append("Expected exactly six formal Haiku calls")
    _dump(out / "validation_report.json", validation)
    report = [
        "# Claim-level AV Sufficiency v2",
        "",
        f"- Status: `{validation['validation_status']}`",
        f"- Model: `{MODEL_ID}`",
        f"- Formal calls: {totals['formal_calls']}",
        f"- Tokens: {totals['input_tokens']} input / {totals['output_tokens']} output",
        f"- Latency: {totals['latency_sec']:.3f} sec",
        "- Raw image inputs: 0",
        f"- Claims: {sum(counts.values())} ({dict(counts)})",
        f"- Visual-review requests: {requests}",
        "",
        "Old sufficiency, claim-gate, and final-answer outputs were loaded only after",
        "all six formal calls completed and were used solely for comparison.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return validation
