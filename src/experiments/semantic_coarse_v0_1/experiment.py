"""Frozen-input loading, adjacency grouping, validation, and persistence."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def frozen_fluid_loose_frontiers(
    *, metrics_path: Path, hierarchy_path: Path, expected_video_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = [
        row for row in metrics
        if row.get("source_group") == "long_primary" and row.get("method") == "fluid_loose"
    ]
    if len(selected) != expected_video_count:
        raise RuntimeError(f"Expected {expected_video_count} long Fluid Loose records, found {len(selected)}")
    hierarchies = {str(row["video_id"]): row for row in load_jsonl(hierarchy_path)}
    output = []
    for row in sorted(selected, key=lambda item: str(item["video_id"])):
        video_id = str(row["video_id"])
        hierarchy = hierarchies.get(video_id)
        if hierarchy is None:
            raise RuntimeError(f"Frozen Safe-Merge tree missing for {video_id}")
        nodes = {str(node["node_id"]): node for node in hierarchy["nodes"]}
        medium_ids = list(row["medium_ids"])
        if len(medium_ids) != len(set(medium_ids)):
            raise RuntimeError(f"Duplicate Fluid Loose Medium ID for {video_id}")
        medium = []
        seen_leaves: list[str] = []
        for medium_id in medium_ids:
            if medium_id not in nodes:
                raise RuntimeError(f"Fluid Loose node {medium_id} absent from frozen tree")
            node = nodes[medium_id]
            medium.append(node)
            seen_leaves.extend(str(item) for item in node["leaf_ids"])
        medium.sort(key=lambda node: (float(node["start"]), str(node["node_id"])))
        expected_leaves = list(hierarchy["fine_leaf_ids"])
        if seen_leaves != expected_leaves:
            # The source IDs may be stored in chronological order already; validate the sorted frontier too.
            seen_leaves = [str(leaf) for node in medium for leaf in node["leaf_ids"]]
        if seen_leaves != expected_leaves:
            raise RuntimeError(f"Fluid Loose frontier does not preserve Fine leaves exactly for {video_id}")
        for left, right in zip(medium, medium[1:]):
            if abs(float(left["end"]) - float(right["start"])) > 1e-6:
                raise RuntimeError(f"Gap/overlap in Fluid Loose frontier for {video_id}")
        frontier_identity = [
            {
                "medium_id": str(node["node_id"]), "start": float(node["start"]),
                "end": float(node["end"]), "leaf_ids": list(node["leaf_ids"]),
            }
            for node in medium
        ]
        output.append(
            {
                "video_id": video_id,
                "video_duration": float(row["video_duration"]),
                "fine_count": int(row["fine_count"]),
                "medium_count": len(medium),
                "medium_ids": [str(node["node_id"]) for node in medium],
                "medium_nodes": medium,
                "hierarchy": hierarchy,
                "frontier_sha256": sha256_json(frontier_identity),
            }
        )
    provenance = {
        "adaptive_metrics_path": metrics_path.as_posix(),
        "adaptive_metrics_sha256": sha256_file(metrics_path),
        "safe_merge_hierarchies_path": hierarchy_path.as_posix(),
        "safe_merge_hierarchies_sha256": sha256_file(hierarchy_path),
        "selected_method": "fluid_loose",
        "selected_source_group": "long_primary",
        "video_count": len(output),
        "medium_count": sum(item["medium_count"] for item in output),
    }
    return output, provenance


def parse_boundary_decision(raw: str) -> dict[str, Any]:
    object_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if object_match is not None:
        try:
            value = json.loads(object_match.group(0))
            fields = ("same_specific_setting", "same_immediate_activity", "same_event_state")
            if all(type(value.get(field)) is bool for field in fields):
                decision = "MERGE" if all(value[field] for field in fields) else "STOP"
                return {
                    "decision": decision,
                    "reason": str(value.get("reason") or "structured immediate-state comparison")[:160],
                    "parse_status": "parsed_structured_all_conditions_required",
                    "same_specific_setting": value["same_specific_setting"],
                    "same_immediate_activity": value["same_immediate_activity"],
                    "same_event_state": value["same_event_state"],
                }
        except json.JSONDecodeError:
            pass
    exact = raw.strip().upper().rstrip(".")
    if exact in {"MERGE", "STOP"}:
        return {"decision": exact, "reason": "strict local-Qwen same-phase decision", "parse_status": "parsed_exact"}
    decision_match = re.search(r"DECISION\s*:\s*(MERGE|STOP)\b", raw, flags=re.IGNORECASE)
    reason_match = re.search(r"REASON\s*:\s*(.+)", raw, flags=re.IGNORECASE)
    if decision_match is None:
        return {"decision": "STOP", "reason": "unparseable output; conservative stop", "parse_status": "fallback_stop"}
    return {
        "decision": decision_match.group(1).upper(),
        "reason": (reason_match.group(1).strip() if reason_match else "model supplied no reason")[:160],
        "parse_status": "parsed",
    }


def photometric_transition_diagnostic(
    *, frame_root: Path, video_id: str, boundary_time: float,
    grayscale_correlation_min: float, mean_luminance_delta_min: float, mean_rgb_delta_min: float,
) -> dict[str, Any]:
    """Cheap boundary-frame guard: photometric changes do not independently become event transitions."""
    directory = frame_root / video_id
    before_index = max(0, int(round(boundary_time)) - 1)
    after_index = max(0, int(round(boundary_time)))
    before_path = directory / f"frame_{before_index:05d}.jpg"
    after_path = directory / f"frame_{after_index:05d}.jpg"
    if not before_path.is_file() or not after_path.is_file():
        return {
            "available": False, "suspected_photometric_only": False,
            "before_path": before_path.as_posix(), "after_path": after_path.as_posix(),
            "reason": "boundary frame unavailable; no photometric bypass",
        }

    def load(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB").resize((96, 54)), dtype=np.float32) / 255.0

    before = load(before_path)
    after = load(after_path)
    before_gray = 0.299 * before[..., 0] + 0.587 * before[..., 1] + 0.114 * before[..., 2]
    after_gray = 0.299 * after[..., 0] + 0.587 * after[..., 1] + 0.114 * after[..., 2]
    before_standard = (before_gray - before_gray.mean()) / max(float(before_gray.std()), 1e-6)
    after_standard = (after_gray - after_gray.mean()) / max(float(after_gray.std()), 1e-6)
    structural_correlation = float(np.mean(before_standard * after_standard))
    luminance_delta = abs(float(before_gray.mean()) - float(after_gray.mean()))
    mean_rgb_delta = float(np.mean(np.abs(before.mean(axis=(0, 1)) - after.mean(axis=(0, 1)))))
    photometric_magnitude = max(
        luminance_delta / max(float(mean_luminance_delta_min), 1e-9),
        mean_rgb_delta / max(float(mean_rgb_delta_min), 1e-9),
    )
    suspected = bool(
        structural_correlation >= float(grayscale_correlation_min)
        and (luminance_delta >= float(mean_luminance_delta_min) or mean_rgb_delta >= float(mean_rgb_delta_min))
    )
    return {
        "available": True, "before_path": before_path.as_posix(), "after_path": after_path.as_posix(),
        "standardized_grayscale_correlation": structural_correlation,
        "mean_luminance_delta": luminance_delta, "mean_rgb_delta": mean_rgb_delta,
        "photometric_magnitude_relative_to_threshold": photometric_magnitude,
        "suspected_photometric_only": suspected,
        "rule": "high grayscale structural persistence plus strong luminance/mean-colour change",
    }


def posture_state_tokens(caption: str, vocabulary: list[str]) -> list[str]:
    lowered = caption.lower()
    return sorted(token for token in vocabulary if re.search(rf"\b{re.escape(token)}\b", lowered))


def build_adjacent_groups(
    *, video_id: str, medium_records: list[dict[str, Any]], decisions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(decisions) != max(0, len(medium_records) - 1):
        raise RuntimeError("Semantic boundary decision count is incomplete")
    groups: list[list[dict[str, Any]]] = []
    current = [medium_records[0]] if medium_records else []
    for index, decision in enumerate(decisions):
        left = medium_records[index]
        right = medium_records[index + 1]
        if decision["left_medium_id"] != left["medium_id"] or decision["right_medium_id"] != right["medium_id"]:
            raise RuntimeError("Semantic boundary decision is not aligned with adjacent Mediums")
        if decision["decision"] == "MERGE":
            current.append(right)
        else:
            groups.append(current)
            current = [right]
    if current:
        groups.append(current)
    result = []
    for index, members in enumerate(groups, start=1):
        result.append(
            {
                "video_id": video_id,
                "coarse_id": f"semantic_coarse_{index:04d}",
                "start": float(members[0]["start"]),
                "end": float(members[-1]["end"]),
                "duration": float(members[-1]["end"] - members[0]["start"]),
                "child_medium_ids": [str(item["medium_id"]) for item in members],
                "child_count": len(members),
                "child_captions": [str(item["caption"]) for item in members],
            }
        )
    return result


def validate_semantic_coarse(
    *, frontier: dict[str, Any], medium_records: list[dict[str, Any]], coarse_records: list[dict[str, Any]]
) -> dict[str, Any]:
    frozen_ids = list(frontier["medium_ids"])
    observed_ids = [str(item["medium_id"]) for item in medium_records]
    child_ids = [str(item) for coarse in coarse_records for item in coarse["child_medium_ids"]]
    intervals_unchanged = all(
        str(node["node_id"]) == record["medium_id"]
        and abs(float(node["start"]) - float(record["start"])) <= 1e-6
        and abs(float(node["end"]) - float(record["end"])) <= 1e-6
        for node, record in zip(frontier["medium_nodes"], medium_records)
    )
    adjacent = all(
        abs(float(left["end"]) - float(right["start"])) <= 1e-6
        for left, right in zip(coarse_records, coarse_records[1:])
    )
    result = {
        "medium_ids_exact": observed_ids == frozen_ids,
        "medium_intervals_unchanged": intervals_unchanged,
        "every_medium_exactly_once": child_ids == frozen_ids,
        "coarse_temporally_adjacent_complete": adjacent,
        "fine_count_unchanged": int(frontier["fine_count"]),
        "frontier_sha256": frontier["frontier_sha256"],
    }
    result["passed"] = all(value for key, value in result.items() if isinstance(value, bool))
    return result
