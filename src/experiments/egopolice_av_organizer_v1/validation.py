from __future__ import annotations

import json
import re
from typing import Any

from .artifact_manifest import verify_artifact_hashes
from .config import PHASE_IDS, ROOT, VIDEO_ID

V7 = ROOT / "outputs/experiments/egopolice_visual_full_asr_fusion_v0_7_typed_minimal_evidence_map"
RECOVERY = ROOT / "outputs/experiments/egopolice_v0_7_minimal_map_staged_presenter_v0_2_p09_continuity_recovery"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_evidence_map() -> dict[str, Any]:
    evidence_map = _load(V7 / "minimal_evidence_map.json")
    phases = evidence_map["phases"]
    errors: list[str] = []
    if evidence_map.get("video_id") != VIDEO_ID:
        errors.append("video_id")
    if [x["phase_id"] for x in phases] != PHASE_IDS:
        errors.append("phase_order")
    if abs(sum(x["end_sec"] - x["start_sec"] for x in phases) - 1235.307) > 0.001:
        errors.append("duration_coverage")
    visual_count = transcript_count = 0
    for phase in phases:
        visual_ids = {x["visual_id"] for x in phase["visual_atoms"]}
        audio_ids = {x["audio_id"] for x in phase["transcript_atoms"]}
        visual_count += len(visual_ids)
        transcript_count += len(audio_ids)
        for link in phase["av_links"]:
            if "note" in link:
                errors.append(f"{phase['phase_id']}:free_text_note")
            if not set(link["visual_ids"]) <= visual_ids:
                errors.append(f"{phase['phase_id']}:link_visual_modality")
            if not set(link["audio_ids"]) <= audio_ids:
                errors.append(f"{phase['phase_id']}:link_audio_modality")
        for context in phase["boundary_context"]:
            if context["modality"] not in {"visual", "audio"}:
                errors.append(f"{phase['phase_id']}:context_modality")
            if context["relation"] not in {"previous", "next"}:
                errors.append(f"{phase['phase_id']}:context_relation")
    copy_audit = _load(V7 / "atom_copy_validation.json")
    if not copy_audit.get("valid") or not copy_audit.get("exact_copy_checked"):
        errors.append("exact_copy_audit")
    if visual_count != 18 or transcript_count != 33:
        errors.append("atom_counts")
    return {
        "valid": not errors, "errors": sorted(set(errors)),
        "phase_count": len(phases), "visual_atoms": visual_count,
        "transcript_atoms": transcript_count,
        "exact_copy_visual_atoms": copy_audit.get("valid", False),
        "exact_copy_transcript_atoms": copy_audit.get("valid", False),
    }


def validate_presenter() -> dict[str, Any]:
    timeline = _load(RECOVERY / "complete_phase_accounts.json")
    continuity = _load(RECOVERY / "continuity_role_audit.json")
    errors: list[str] = []
    if [x["phase_id"] for x in timeline] != PHASE_IDS:
        errors.append("timeline_order")
    by_id = {x["phase_id"]: x for x in timeline}
    p08 = json.dumps(by_id["P08"], ensure_ascii=False).lower()
    p09 = json.dumps(by_id["P09"], ensure_ascii=False).lower()
    p06 = json.dumps(by_id["P06"], ensure_ascii=False).lower()
    p07 = json.dumps(by_id["P07"], ensure_ascii=False).lower()
    for term in ("tourniquet", "do-not-move", "legs down", "coordination"):
        if term not in p08:
            errors.append(f"P08:{term}")
    for term in ("ems", "ambulance", "transport", "chest", "groin"):
        if term not in p09:
            errors.append(f"P09:{term}")
    if "knife" not in p06 or not re.search(r"(audio|speaker|transcript).{0,100}knife|knife.{0,100}(audio|speaker|transcript)", p06):
        errors.append("P06:knife_attribution")
    if "live round" not in p07 or not re.search(r"(audio|speaker|transcript).{0,100}live round", p07):
        errors.append("P07:live_round_attribution")
    if not re.search(r"(assignment.{0,80}uncertain|unclear whether|specific individuals)", p09):
        errors.append("P09:injury_assignment")
    if not continuity.get("valid"):
        errors.append("P09:continuity_audit")
    if continuity.get("role_source") != "contextual_continuity_reference":
        errors.append("P09:continuity_role_source")
    if continuity.get("direct_visual_role") is not False:
        errors.append("P09:direct_visual_role")
    if continuity.get("identity_tracking_claimed") is not False:
        errors.append("P09:identity_tracking")
    return {"valid": not errors, "errors": errors, "phase_count": len(timeline)}


def validate_hashes() -> dict[str, Any]:
    results = verify_artifact_hashes()
    failed = [x["path"] for x in results if not x["matches"]]
    return {"valid": not failed, "failed": failed, "checked": len(results)}
