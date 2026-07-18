"""Technical identity and serialization contracts for Canonical Baseline v1.

These contracts describe existing entity levels and validate serialization.
They do not choose evidence, assign roles, rank candidates, or alter budgets.
"""

from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Iterable


EVIDENCE_CONTRACT_VERSION = "baseline-v1-technical-contract-v1"

# A count at one level must never be substituted for a count at another level.
ENTITY_TAXONOMY = {
    "candidate_entity": "A Task5B/Task5C candidate entering Task6 selection.",
    "canonical_evidence_entity": "One retained Task6 entity exposed by evidence_id.",
    "refined_evidence": "A candidate representation with refined temporal/media provenance.",
    "visual_frame_asset": "A JPG asset owned by one visual evidence entity; not an evidence entity.",
    "audio_clip_asset": "A WAV asset owned by one acoustic evidence entity; not an evidence entity.",
    "speech_segment": "A speech evidence entity with transcript and effective interval.",
    "relation": "A directed edge between evidence entities; never counted as evidence.",
    "evidence_group": "A container of model-facing evidence entities and closed relations.",
    "packet_record": "The case-level serialized evidence packet.",
}

RELATION_IDENTITY_FIELDS = (
    "source_candidate_id",
    "relation_type",
    "target_candidate_id",
    "temporal_gap_sec",
    "relation_confidence",
    "relation_basis",
)


class EvidenceContractError(ValueError):
    """Raised when a producer emits an internally inconsistent evidence record."""


def canonical_relation_identity(relation: dict[str, Any]) -> tuple[Any, ...]:
    """Return the directed semantic identity of a canonical relation."""
    gap = relation.get("temporal_gap_sec")
    normalized_gap = round(float(gap), 9) if isinstance(gap, (int, float)) else gap
    values = dict(relation)
    values["temporal_gap_sec"] = normalized_gap
    return tuple(values.get(field) for field in RELATION_IDENTITY_FIELDS)


def stabilize_relations(
    relations: Iterable[dict[str, Any]],
    supplied_evidence_ids: Iterable[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Remove only dangling or duplicate relation records, preserving order.

    A dangling record is not converted into evidence and a duplicate is not
    counted twice. Both are retained in an explicit technical audit trail.
    """
    supplied = set(supplied_evidence_ids)
    output: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for source in relations:
        item = copy.deepcopy(source)
        source_id = item.get("source_candidate_id")
        target_id = item.get("target_candidate_id")
        if source_id not in supplied or target_id not in supplied:
            invalid.append(
                {
                    "relation": item,
                    "reason": "relation_endpoint_not_supplied_evidence",
                    "missing_endpoint_ids": sorted(
                        {value for value in (source_id, target_id) if value not in supplied},
                        key=str,
                    ),
                }
            )
            continue
        identity = canonical_relation_identity(item)
        if identity in seen:
            duplicates.append(
                {
                    "relation": item,
                    "reason": "duplicate_canonical_relation_identity",
                    "identity": list(identity),
                }
            )
            continue
        seen.add(identity)
        output.append(item)
    return output, {
        "contract_version": EVIDENCE_CONTRACT_VERSION,
        "identity_fields": list(RELATION_IDENTITY_FIELDS),
        "removed_invalid_relations": invalid,
        "removed_duplicate_relations": duplicates,
        "retained_relation_count": len(output),
        "relations_unique": len(output) == len({canonical_relation_identity(item) for item in output}),
        "endpoint_closure": all(
            item["source_candidate_id"] in supplied and item["target_candidate_id"] in supplied
            for item in output
        ),
    }


def _candidate_ids(items: Iterable[dict[str, Any]], field: str = "candidate_id") -> list[str]:
    values = [item.get(field) for item in items]
    if any(not isinstance(value, str) or not value for value in values):
        raise EvidenceContractError(f"Missing or invalid {field} in evidence contract")
    return values  # type: ignore[return-value]


def validate_packet_contract(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate Task6 candidate, group, relation, and asset identity levels."""
    retained_ids = _candidate_ids(packet.get("retained_candidates", []))
    if len(retained_ids) != len(set(retained_ids)):
        raise EvidenceContractError("Task6 retained candidate IDs are not unique")

    non_model_facing = packet.get("intentionally_non_model_facing_candidates", [])
    non_model_ids = _candidate_ids(non_model_facing) if non_model_facing else []
    for item in non_model_facing:
        if not str(item.get("reason", "")).strip():
            raise EvidenceContractError("Intentionally non-model-facing evidence requires a reason")

    group_ids = _candidate_ids(packet.get("retained_evidence_groups", []), "group_id")
    if len(group_ids) != len(set(group_ids)):
        raise EvidenceContractError("Task6 evidence group IDs are not unique")
    member_ids = [
        candidate_id
        for group in packet.get("retained_evidence_groups", [])
        for candidate_id in _candidate_ids(group.get("retained_candidates", []))
    ]
    member_counts = Counter(member_ids)
    duplicate_members = sorted(key for key, count in member_counts.items() if count > 1)
    if duplicate_members:
        raise EvidenceContractError(
            f"Task6 model-facing evidence appears in multiple serialized groups: {duplicate_members}"
        )
    retained = set(retained_ids)
    grouped = set(member_ids)
    explicit_non_model = set(non_model_ids)
    if grouped & explicit_non_model:
        raise EvidenceContractError("Evidence cannot be both model-facing and intentionally non-model-facing")
    orphaned = retained - grouped - explicit_non_model
    unexpected = (grouped | explicit_non_model) - retained
    if orphaned:
        raise EvidenceContractError(f"Task6 retained evidence has no model-facing group: {sorted(orphaned)}")
    if unexpected:
        raise EvidenceContractError(f"Task6 group references non-retained evidence: {sorted(unexpected)}")

    relations = packet.get("relations", [])
    relation_identities = [canonical_relation_identity(item) for item in relations]
    if len(relation_identities) != len(set(relation_identities)):
        raise EvidenceContractError("Task6 canonical relations are not unique")
    for item in relations:
        if item.get("source_candidate_id") not in grouped or item.get("target_candidate_id") not in grouped:
            raise EvidenceContractError("Task6 model-facing relation endpoint is not supplied evidence")

    group_relations = [
        relation
        for group in packet.get("retained_evidence_groups", [])
        for relation in group.get("relations", [])
    ]
    group_relation_identities = [canonical_relation_identity(item) for item in group_relations]
    if len(group_relation_identities) != len(set(group_relation_identities)):
        raise EvidenceContractError("Task6 group relations are not globally unique")
    if set(group_relation_identities) != set(relation_identities):
        raise EvidenceContractError("Task6 packet relations and grouped relations do not match")

    return {
        "contract_version": EVIDENCE_CONTRACT_VERSION,
        "entity_taxonomy": copy.deepcopy(ENTITY_TAXONOMY),
        "candidate_entity_count": len(packet.get("candidates_before_reranking", [])),
        "canonical_evidence_entity_count": len(retained_ids),
        "model_facing_evidence_entity_count": len(grouped),
        "intentionally_non_model_facing_count": len(explicit_non_model),
        "visual_frame_asset_count": sum(
            len(item.get("canonical_visual_frames", []))
            for item in packet.get("retained_candidates", [])
        ),
        "audio_clip_asset_count": sum(
            bool(item.get("local_audio_clip_reference"))
            for item in packet.get("retained_candidates", [])
        ),
        "relation_count": len(relations),
        "evidence_group_count": len(group_ids),
        "retained_group_membership_complete": True,
        "retained_group_membership_exactly_once": True,
        "relation_endpoint_closure": True,
        "relation_identity_unique": True,
    }


def payload_evidence_ids(payload: dict[str, Any]) -> list[str]:
    """Return evidence-entity IDs only; frame/audio assets are excluded."""
    return [
        item["evidence_id"]
        for group in payload.get("evidence_groups", [])
        for field in ("visual_evidence", "speech_evidence", "acoustic_evidence")
        for item in group.get(field, [])
    ]


def validate_payload_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate Task7A's exact model-facing identity and relation closure."""
    ids = payload_evidence_ids(payload)
    if any(not isinstance(value, str) or not value for value in ids):
        raise EvidenceContractError("Task7A payload contains an invalid evidence_id")
    if len(ids) != len(set(ids)):
        raise EvidenceContractError("Task7A model-facing evidence IDs are not globally unique")
    supplied = set(ids)
    relations = [
        item
        for group in payload.get("evidence_groups", [])
        for item in group.get("relations", [])
    ]
    identities = [canonical_relation_identity(item) for item in relations]
    if len(identities) != len(set(identities)):
        raise EvidenceContractError("Task7A model-facing relations are not globally unique")
    for relation in relations:
        if relation.get("source_candidate_id") not in supplied or relation.get("target_candidate_id") not in supplied:
            raise EvidenceContractError("Task7A relation endpoint is absent from supplied evidence")
    group_ids = [group.get("group_id") for group in payload.get("evidence_groups", [])]
    if any(not isinstance(value, str) or not value for value in group_ids):
        raise EvidenceContractError("Task7A payload contains an invalid group_id")
    if len(group_ids) != len(set(group_ids)):
        raise EvidenceContractError("Task7A evidence group IDs are not unique")
    return {
        "contract_version": EVIDENCE_CONTRACT_VERSION,
        "evidence_ids": ids,
        "evidence_id_count": len(ids),
        "evidence_ids_globally_unique": True,
        "relation_count": len(relations),
        "relation_endpoint_closure": True,
        "relation_identity_unique": True,
        "group_ids_unique": True,
    }
