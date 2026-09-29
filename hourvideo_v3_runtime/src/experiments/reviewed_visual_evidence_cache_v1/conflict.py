from __future__ import annotations

from collections import defaultdict
from typing import Any


PRIORITY = {"reviewed_visual_frame": 3, "visual_caption": 2, "detector_observation": 1}


def resolve_exact_scope(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer reviewed evidence only for identical image/time/fact scope.

    Audio is retained as a separate modality and is never superseded here.
    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    separate = []
    for item in items:
        if item["evidence_type"] == "audio_asr":
            separate.append(item); continue
        key = (item.get("image_sha256"), item.get("fine_id"), item.get("timestamp_sec"), item.get("fact_scope"))
        if any(value is None for value in key):
            separate.append(item); continue
        groups[key].append(item)
    resolutions = []
    for key, rows in groups.items():
        reviewed = [row for row in rows if row["evidence_type"] == "reviewed_visual_frame"]
        if len({row.get("assertion") for row in reviewed}) > 1:
            resolutions.append({"scope_key": key, "status": "conflicting_reviewed_visual_evidence", "preferred_evidence_ids": [], "superseded_evidence_ids": []})
            continue
        top = max(PRIORITY.get(row["evidence_type"], 0) for row in rows)
        preferred = [row for row in rows if PRIORITY.get(row["evidence_type"], 0) == top]
        superseded = [row for row in rows if row not in preferred and row.get("assertion") != preferred[0].get("assertion")]
        resolutions.append({
            "scope_key": {"image_sha256":key[0],"fine_id":key[1],"timestamp_sec":key[2],"fact_scope":key[3]},
            "status":"resolved_exact_scope","preferred_evidence_ids":[row["evidence_id"] for row in preferred],
            "superseded_evidence_ids":[row["evidence_id"] for row in superseded],
            "scope_limited":True,
        })
    return {"resolutions": resolutions, "separate_or_unscoped_evidence_ids": [row["evidence_id"] for row in separate], "global_supersession_allowed": False}
