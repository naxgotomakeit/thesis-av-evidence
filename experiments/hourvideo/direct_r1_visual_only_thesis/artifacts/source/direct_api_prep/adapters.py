"""Adapters from frozen R1/R3/GenS evidence into the Direct package schema."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from .evidence import EvidenceItem, EvidencePackage, finalized_package, sha256_file


def map_identity(map_path: Path, *, representation: str) -> dict[str, Any]:
    return {"representation": representation, "path": str(map_path), "sha256": sha256_file(map_path)}


def r1_or_r3_package(*, question_id: str, method: str, navigation_map_path: Path, representation: str, ranked_navigation_frames: Iterable[EvidenceItem], navigation_cost: dict[str, Any], verify: bool = True) -> EvidencePackage:
    if method not in {"R1", "R3"}:
        raise ValueError("R1/R3 adapter requires method R1 or R3")
    return finalized_package(
        question_id, method, ranked_navigation_frames,
        map_text_evidence=map_identity(navigation_map_path, representation=representation),
        cost_boundaries={"offline_construction_cost": "source-recorded", "question_time_navigation_retrieval": navigation_cost, "direct_answering_api_cost": "not-run", "total_operational_cost": "not-run"}, verify=verify,
    )


def gens_package(row: dict[str, Any], *, resolve_frame_path: Callable[[str], Path], verify: bool = True) -> EvidencePackage:
    if set(row) & {"selection_provenance", "winning_option", "option_scores"}:
        raise ValueError("GenS audit-only selection provenance cannot enter Direct evidence")
    frames = row["selected_frames"]
    if row["selected_frame_count"] != len(frames):
        raise ValueError("GenS selected_frame_count mismatch")
    candidates = [EvidenceItem(
        source_evidence_identity=f"gens:{row['qa_uid']}:{frame['selection_order']}",
        selector_rank=int(frame["selection_order"]), timestamp_sec=float(frame["timestamp_sec"]),
        frame_path=str(resolve_frame_path(frame["relative_frame_path"])), frame_sha256=frame["frame_sha256"],
    ) for frame in frames]
    return finalized_package(
        row["qa_uid"], "GenS", candidates, map_text_evidence=None,
        cost_boundaries={"offline_construction_cost": "frozen GenS selector metadata", "question_time_navigation_retrieval": "frozen native selection; no rerun", "direct_answering_api_cost": "not-run", "total_operational_cost": "not-run"}, verify=verify,
    )
