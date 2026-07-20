"""Deterministic sampling and schema primitives for taxonomy validation.

This module intentionally reads only question identifiers/text. Dataset answers,
options, timestamps, media, and prior predictions never enter the experiment
record or provider prompt.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_NAME = "planner_taxonomy_validation_v0_1"
SCOPES = ("local", "multi_event", "global", "unclear")
NATURES = ("static", "dynamic", "uncertain")
SECONDARY_NATURES = ("static", "dynamic", None)
MODALITIES = ("visual", "audio", "audio_visual", "indeterminate_from_question")

STRESS_BUCKET_PATTERNS: dict[str, tuple[str, ...]] = {
    "temporal_relation": (
        r"\bbefore\b", r"\bafter\b", r"\bthen\b", r"\bwhile\b",
        r"\bfirst\b", r"\blast\b", r"\bearlier\b", r"\blater\b",
    ),
    "global_progression": (
        r"\boverall\b", r"\bthroughout\b", r"\bmain\b", r"\bsequence\b",
        r"\bsteps?\b", r"\bentire\b", r"\bwhole\b", r"\bprogress",
    ),
    "object_state": (
        r"\bwhat object\b", r"\bwhich object\b", r"\bcolor\b", r"\bwhere\b",
        r"\blocation\b", r"\bstate\b", r"\bwearing\b", r"\bholding\b",
    ),
    "action_or_causality": (
        r"\bhow\b", r"\bwhy\b", r"\baction\b", r"\bdoing\b",
        r"\bhappen", r"\bcaus", r"\bresult\b",
    ),
    "auditory": (
        r"\bsound", r"\bhear", r"\bnoise", r"\bsay\b", r"\bsays\b",
        r"\bsaid\b", r"\bspeak", r"\bvoice", r"\blisten", r"\baudible",
    ),
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_question(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        raise ValueError("Question text must be non-empty")
    return text


def load_question_only_sources(source_files: dict[str, str]) -> dict[str, list[dict[str, str]]]:
    """Load only IDs/text and deliberately discard all other source fields."""
    output: dict[str, list[dict[str, str]]] = {}
    specs = {
        "egoschema": ("q_uid", "question"),
        "egosound": ("question_id", "question"),
    }
    for dataset, (id_key, question_key) in specs.items():
        path = Path(source_files[dataset])
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{dataset} question source must be a JSON list")
        clean: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in rows:
            identifier = str(row[id_key])
            question = _normalized_question(row[question_key])
            if identifier in seen:
                raise ValueError(f"Duplicate {dataset} question ID: {identifier}")
            seen.add(identifier)
            clean.append(
                {
                    "record_id": f"{dataset}:{identifier}",
                    "question_id": identifier,
                    "question": question,
                    "dataset": dataset,
                    "text_sha256": sha256_text(question),
                }
            )
        output[dataset] = clean
    return output


def _rank_key(seed: int, purpose: str, row: dict[str, str]) -> tuple[str, str]:
    payload = f"{seed}|{purpose}|{row['dataset']}|{row['question_id']}|{row['text_sha256']}"
    return sha256_text(payload), row["record_id"]


def stress_buckets(question: str) -> list[str]:
    lowered = question.lower()
    matched = [
        bucket
        for bucket, patterns in STRESS_BUCKET_PATTERNS.items()
        if any(re.search(pattern, lowered) for pattern in patterns)
    ]
    auditory = "auditory" in matched
    visual_or_actor = bool(
        re.search(r"\b(person|man|woman|speaker|object|look|see|doing|action|where)\b", lowered)
    )
    if auditory and visual_or_actor:
        matched.append("cross_modal_wording")
    temporal_or_global = {"temporal_relation", "global_progression"}.intersection(matched)
    if not temporal_or_global:
        matched.append("no_obvious_temporal_cue")
    return sorted(set(matched))


def _public_sample_row(
    row: dict[str, str], *, sample_type: str, seed: int, cue_buckets: list[str] | None = None
) -> dict[str, Any]:
    return {
        **row,
        "sample_type": sample_type,
        "sampling_seed": seed,
        "cue_buckets": list(cue_buckets or []),
    }


def freeze_samples(
    sources: dict[str, list[dict[str, str]]],
    *,
    seed: int,
    representative_per_dataset: int,
    stress_max_per_dataset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create disjoint deterministic hash-ranked representative and cue stress sets."""
    representative: list[dict[str, Any]] = []
    stress: list[dict[str, Any]] = []
    bucket_order = [
        "temporal_relation",
        "global_progression",
        "object_state",
        "action_or_causality",
        "auditory",
        "cross_modal_wording",
        "no_obvious_temporal_cue",
    ]
    for dataset in ("egoschema", "egosound"):
        rows = list(sources[dataset])
        if len(rows) < representative_per_dataset + stress_max_per_dataset:
            raise ValueError(f"Not enough {dataset} questions for disjoint samples")
        ordered = sorted(rows, key=lambda row: _rank_key(seed, "representative", row))
        selected_rep = ordered[:representative_per_dataset]
        representative.extend(
            _public_sample_row(row, sample_type="representative", seed=seed)
            for row in selected_rep
        )
        excluded = {row["record_id"] for row in selected_rep}
        candidates = [row for row in rows if row["record_id"] not in excluded]
        memberships = {row["record_id"]: stress_buckets(row["question"]) for row in candidates}
        queues: dict[str, list[dict[str, str]]] = {}
        for bucket in bucket_order:
            queues[bucket] = sorted(
                [row for row in candidates if bucket in memberships[row["record_id"]]],
                key=lambda row, b=bucket: _rank_key(seed, f"stress:{b}", row),
            )
        selected: list[dict[str, str]] = []
        selected_ids: set[str] = set()
        positions = {bucket: 0 for bucket in bucket_order}
        while len(selected) < stress_max_per_dataset:
            made_progress = False
            for bucket in bucket_order:
                queue = queues[bucket]
                while positions[bucket] < len(queue):
                    row = queue[positions[bucket]]
                    positions[bucket] += 1
                    if row["record_id"] in selected_ids:
                        continue
                    selected.append(row)
                    selected_ids.add(row["record_id"])
                    made_progress = True
                    break
                if len(selected) >= stress_max_per_dataset:
                    break
            if not made_progress:
                break
        if len(selected) < stress_max_per_dataset:
            remaining = sorted(
                [row for row in candidates if row["record_id"] not in selected_ids],
                key=lambda row: _rank_key(seed, "stress:fill", row),
            )
            selected.extend(remaining[: stress_max_per_dataset - len(selected)])
        stress.extend(
            _public_sample_row(
                row,
                sample_type="diversity_stress",
                seed=seed,
                cue_buckets=memberships[row["record_id"]],
            )
            for row in selected
        )
    if {row["record_id"] for row in representative}.intersection(
        row["record_id"] for row in stress
    ):
        raise AssertionError("Representative and stress samples must be disjoint")
    return representative, stress


def validate_annotation(row: dict[str, Any], expected_item_id: str | None = None) -> dict[str, Any]:
    """Validate the strict experimental annotation schema without fabricating defaults."""
    required = {
        "item_id", "scope", "nature_primary", "nature_secondary", "nature_ambiguity",
        "modality", "confidence", "reason_short", "taxonomy_failure",
        "taxonomy_failure_reason",
    }
    if set(row) != required:
        raise ValueError(f"Annotation fields differ: missing={required-set(row)}, extra={set(row)-required}")
    if expected_item_id is not None and str(row["item_id"]) != expected_item_id:
        raise ValueError("Annotation item_id does not match requested item")
    if row["scope"] not in SCOPES:
        raise ValueError(f"Invalid scope: {row['scope']}")
    if row["nature_primary"] not in NATURES:
        raise ValueError(f"Invalid nature_primary: {row['nature_primary']}")
    if row["nature_secondary"] not in SECONDARY_NATURES:
        raise ValueError(f"Invalid nature_secondary: {row['nature_secondary']}")
    if row["modality"] not in MODALITIES:
        raise ValueError(f"Invalid modality: {row['modality']}")
    if not isinstance(row["nature_ambiguity"], bool) or not isinstance(row["taxonomy_failure"], bool):
        raise ValueError("Ambiguity/failure flags must be booleans")
    if row["nature_secondary"] is not None and row["nature_secondary"] == row["nature_primary"]:
        raise ValueError("Secondary nature must differ from primary")
    if row["nature_secondary"] is not None and row["nature_primary"] == "uncertain":
        raise ValueError("Uncertain primary cannot also carry a forced secondary")
    if row["nature_secondary"] is not None and not row["nature_ambiguity"]:
        raise ValueError("Secondary nature requires nature_ambiguity=true")
    if set(row["confidence"]) != {"scope", "nature", "modality"}:
        raise ValueError("Confidence object must contain exactly the three axes")
    if set(row["reason_short"]) != {"scope", "nature", "modality"}:
        raise ValueError("Reason object must contain exactly the three axes")
    for axis in ("scope", "nature", "modality"):
        value = row["confidence"].get(axis)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= float(value) <= 1:
            raise ValueError(f"Invalid {axis} confidence")
        reason = row["reason_short"].get(axis)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"Missing {axis} reason")
        if len(reason.split()) > 20:
            raise ValueError(f"{axis} reason exceeds concise limit")
    failure_reason = row["taxonomy_failure_reason"]
    if row["taxonomy_failure"] and (
        not isinstance(failure_reason, str) or not failure_reason.strip()
    ):
        raise ValueError("Taxonomy failure requires a reason")
    if not row["taxonomy_failure"] and failure_reason is not None:
        raise ValueError("Non-failure taxonomy_failure_reason must be null")
    return row


def select_stability_subset(
    annotations: list[dict[str, Any]], *, per_dataset: int, seed: int
) -> list[dict[str, Any]]:
    """Freeze a balanced deterministic subset of high/low/ambiguous annotations."""
    selected: list[dict[str, Any]] = []
    for dataset in ("egoschema", "egosound"):
        rows = [row for row in annotations if row["dataset"] == dataset]
        if len(rows) < per_dataset:
            raise ValueError(f"Insufficient {dataset} annotations for stability subset")
        chosen: list[dict[str, Any]] = []
        chosen_ids: set[str] = set()

        def minimum_confidence(row: dict[str, Any]) -> float:
            return min(float(value) for value in row["confidence"].values())

        ambiguous = sorted(
            rows,
            key=lambda row: (
                not bool(row["taxonomy_failure"]),
                not bool(row["nature_ambiguity"]),
                minimum_confidence(row),
                _rank_key(seed, "stability:ambiguous", row),
            ),
        )
        high = sorted(
            rows,
            key=lambda row: (-minimum_confidence(row), _rank_key(seed, "stability:high", row)),
        )

        def take(pool: Iterable[dict[str, Any]], count: int, reason: str) -> None:
            for row in pool:
                if len([item for item in chosen if item["stability_stratum"] == reason]) >= count:
                    break
                if row["record_id"] in chosen_ids:
                    continue
                chosen.append({**row, "stability_stratum": reason})
                chosen_ids.add(row["record_id"])

        low_count = min(8, per_dataset)
        high_count = min(6, max(0, per_dataset - low_count))
        take(ambiguous, low_count, "low_confidence_or_ambiguous")
        take(high, high_count, "high_confidence")

        remaining_target = per_dataset - len(chosen)
        if remaining_target:
            label_counts: Counter[str] = Counter()
            for row in chosen:
                label_counts.update((f"scope:{row['scope']}", f"nature:{row['nature_primary']}", f"modality:{row['modality']}"))
            pool = [row for row in rows if row["record_id"] not in chosen_ids]
            while len(chosen) < per_dataset and pool:
                def diversity_key(row: dict[str, Any]) -> tuple[Any, ...]:
                    labels = (f"scope:{row['scope']}", f"nature:{row['nature_primary']}", f"modality:{row['modality']}")
                    return (sum(label_counts[label] for label in labels), _rank_key(seed, "stability:coverage", row))
                row = min(pool, key=diversity_key)
                pool.remove(row)
                chosen.append({**row, "stability_stratum": "label_coverage"})
                chosen_ids.add(row["record_id"])
                label_counts.update((f"scope:{row['scope']}", f"nature:{row['nature_primary']}", f"modality:{row['modality']}"))
        if len(chosen) != per_dataset:
            raise AssertionError("Stability subset size invariant failed")
        selected.extend(chosen)
    return selected


def mean(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return sum(rows) / len(rows) if rows else None


def normalized_mutual_information(rows: list[dict[str, Any]], left: str, right: str) -> float | None:
    """Symmetric normalized mutual information for categorical axis-dependence audit."""
    if not rows:
        return None
    joint = Counter((str(row[left]), str(row[right])) for row in rows)
    left_counts = Counter(str(row[left]) for row in rows)
    right_counts = Counter(str(row[right]) for row in rows)
    size = float(len(rows))
    mi = 0.0
    for (a, b), count in joint.items():
        probability = count / size
        mi += probability * math.log(probability / ((left_counts[a] / size) * (right_counts[b] / size)))
    h_left = -sum((count / size) * math.log(count / size) for count in left_counts.values())
    h_right = -sum((count / size) * math.log(count / size) for count in right_counts.values())
    denominator = math.sqrt(h_left * h_right)
    return mi / denominator if denominator else 0.0


def count_distribution(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    counts = Counter(str(row[field]) for row in rows)
    total = len(rows)
    return {
        "count": total,
        "counts": dict(sorted(counts.items())),
        "rates": {key: value / total for key, value in sorted(counts.items())} if total else {},
        "dominant_class": counts.most_common(1)[0][0] if counts else None,
        "dominant_rate": counts.most_common(1)[0][1] / total if counts else None,
    }


def group_by(items: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        output[str(item[key])].append(item)
    return dict(output)
