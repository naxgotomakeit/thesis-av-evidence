"""Immutable, local-only evidence package contract for Direct API.

No provider transport belongs in this module.  The hard cap is enforced before
any future transport adapter is permitted to serialise a request.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


MAX_PHYSICAL_IMAGES = 16
FORBIDDEN_TRACE_MARKERS = {
    "shared_investigation", "direct_final", "final_answer", "fine_results",
    "claim_execution", "shared_attempt_audit", "live_model_calls",
    "correct_answer", "gold_answer", "answer_key",
}


@dataclass(frozen=True)
class EvidenceItem:
    source_evidence_identity: str
    selector_rank: int
    timestamp_sec: float
    frame_path: str
    frame_sha256: str
    text_evidence: str | None = None
    interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class EvidencePackage:
    schema_version: str
    question_id: str
    method: str
    selection_state: str
    map_text_evidence: dict[str, Any] | None
    candidates_before_cap: tuple[EvidenceItem, ...]
    final_evidence: tuple[EvidenceItem, ...]
    count_before_hard_cap: int | None
    count_after_hard_cap: int | None
    number_removed_by_hard_cap: int | None
    gold_loaded: bool
    cost_boundaries: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidates_before_cap"] = [asdict(item) for item in self.candidates_before_cap]
        result["final_evidence"] = [
            {**asdict(item), "final_chronological_order": index}
            for index, item in enumerate(self.final_evidence)
        ]
        result["physical_image_count"] = len(self.final_evidence)
        return result


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_safe_direct_evidence(value: Any) -> None:
    """Reject persisted staged trace fields/paths from Direct evidence."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in FORBIDDEN_TRACE_MARKERS:
                raise ValueError(f"forbidden prior staged trace field: {key}")
            assert_safe_direct_evidence(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            assert_safe_direct_evidence(child)
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(marker in lowered for marker in FORBIDDEN_TRACE_MARKERS):
            raise ValueError("forbidden prior staged trace path/content")


def verify_frame(item: EvidenceItem) -> None:
    path = Path(item.frame_path)
    if not path.is_file():
        raise FileNotFoundError(f"Direct evidence frame is absent: {path}")
    observed = sha256_file(path)
    if observed != item.frame_sha256:
        raise ValueError(f"Direct evidence frame SHA mismatch: {path}")


def cap_and_order(candidates: Iterable[EvidenceItem], *, verify: bool = True) -> tuple[tuple[EvidenceItem, ...], int, int]:
    """Rank-cap first, then chronologically order presentation.

    A smaller rank is better.  Ties are deterministic by timestamp, identity,
    and path.  This function is the code-level <=16 physical-image guard.
    """
    ranked = sorted(candidates, key=lambda row: (row.selector_rank, row.timestamp_sec, row.source_evidence_identity, row.frame_path))
    if len({(row.source_evidence_identity, row.frame_path) for row in ranked}) != len(ranked):
        raise ValueError("duplicate source evidence identity/frame path in candidate evidence")
    selected = ranked[:MAX_PHYSICAL_IMAGES]
    if verify:
        for item in selected:
            verify_frame(item)
    chronological = tuple(sorted(selected, key=lambda row: (row.timestamp_sec, row.selector_rank, row.source_evidence_identity, row.frame_path)))
    assert len(chronological) <= MAX_PHYSICAL_IMAGES
    return chronological, len(ranked), len(ranked) - len(chronological)


def finalized_package(question_id: str, method: str, candidates: Iterable[EvidenceItem], *, map_text_evidence: dict[str, Any] | None, cost_boundaries: dict[str, Any], verify: bool = True) -> EvidencePackage:
    assert_safe_direct_evidence(map_text_evidence)
    candidate_rows = tuple(candidates)
    final, before, removed = cap_and_order(candidate_rows, verify=verify)
    package = EvidencePackage(
        schema_version="hourvideo_direct_evidence_package_v1", question_id=question_id,
        method=method, selection_state="finalized", map_text_evidence=map_text_evidence,
        candidates_before_cap=candidate_rows, final_evidence=final,
        count_before_hard_cap=before, count_after_hard_cap=len(final),
        number_removed_by_hard_cap=removed, gold_loaded=False, cost_boundaries=cost_boundaries,
    )
    assert len(package.final_evidence) <= MAX_PHYSICAL_IMAGES
    return package


def pending_navigation_package(question_id: str, method: str, *, map_text_evidence: dict[str, Any], cost_boundaries: dict[str, Any]) -> EvidencePackage:
    """Record immutable source identity without pretending navigation ran."""
    assert_safe_direct_evidence(map_text_evidence)
    return EvidencePackage(
        schema_version="hourvideo_direct_evidence_package_v1", question_id=question_id,
        method=method, selection_state="pending_question_time_navigation",
        map_text_evidence=map_text_evidence, candidates_before_cap=(), final_evidence=(),
        count_before_hard_cap=None, count_after_hard_cap=None,
        number_removed_by_hard_cap=None, gold_loaded=False, cost_boundaries=cost_boundaries,
    )
