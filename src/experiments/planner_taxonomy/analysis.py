"""Deterministic post-annotation analyses for the taxonomy experiment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .core import count_distribution, mean, normalized_mutual_information


LABEL_FIELDS = {
    "scope": ("scope", "scope"),
    "nature": ("nature_primary", "nature"),
    "modality": ("modality", "modality"),
}


def _axis_summary(rows: list[dict[str, Any]], axis: str) -> dict[str, Any]:
    label_field, confidence_field = LABEL_FIELDS[axis]
    distribution = count_distribution(rows, label_field)
    confidences = [float(row["confidence"][confidence_field]) for row in rows]
    summary = {
        **distribution,
        "mean_confidence": mean(confidences),
        "low_confidence_count": sum(value < 0.70 for value in confidences),
        "low_confidence_rate": sum(value < 0.70 for value in confidences) / len(rows) if rows else None,
        "high_confidence_count": sum(value >= 0.85 for value in confidences),
        "high_confidence_rate": sum(value >= 0.85 for value in confidences) / len(rows) if rows else None,
    }
    if axis == "scope":
        summary["unclear_rate"] = distribution["rates"].get("unclear", 0.0)
    elif axis == "nature":
        summary["uncertain_rate"] = distribution["rates"].get("uncertain", 0.0)
        summary["secondary_label_rate"] = (
            sum(row["nature_secondary"] is not None for row in rows) / len(rows) if rows else None
        )
        summary["ambiguity_rate"] = (
            sum(bool(row["nature_ambiguity"]) for row in rows) / len(rows) if rows else None
        )
    else:
        summary["indeterminate_rate"] = distribution["rates"].get(
            "indeterminate_from_question", 0.0
        )
    return summary


def distribution_report(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    representative = [row for row in annotations if row["sample_type"] == "representative"]
    stress = [row for row in annotations if row["sample_type"] == "diversity_stress"]
    groups = {
        "egoschema_representative": [row for row in representative if row["dataset"] == "egoschema"],
        "egosound_representative": [row for row in representative if row["dataset"] == "egosound"],
        "combined_representative": representative,
        "egoschema_diversity_stress": [row for row in stress if row["dataset"] == "egoschema"],
        "egosound_diversity_stress": [row for row in stress if row["dataset"] == "egosound"],
        "combined_diversity_stress": stress,
    }
    return {
        "schema_version": "planner-taxonomy-distribution-v1",
        "representative_is_natural_prevalence_sample": True,
        "stress_is_natural_prevalence_sample": False,
        "groups": {
            name: {
                "count": len(rows),
                "scope": _axis_summary(rows, "scope"),
                "nature": _axis_summary(rows, "nature"),
                "modality": _axis_summary(rows, "modality"),
                "taxonomy_failure_count": sum(bool(row["taxonomy_failure"]) for row in rows),
                "taxonomy_failure_rate": (
                    sum(bool(row["taxonomy_failure"]) for row in rows) / len(rows) if rows else None
                ),
            }
            for name, rows in groups.items()
        },
    }


def stability_report(
    first_pass: list[dict[str, Any]], second_pass: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    first_by_id = {row["record_id"]: row for row in first_pass}
    second_by_id = {row["record_id"]: row for row in second_pass}
    if set(first_by_id) != set(second_by_id):
        raise ValueError("Stability passes contain different question identities")
    output = []
    confusion = {axis: Counter() for axis in ("scope", "nature_primary", "nature_ambiguity", "modality")}
    agreement_counts = Counter()
    by_dataset: dict[str, Counter[str]] = {
        "egoschema": Counter(),
        "egosound": Counter(),
    }
    for record_id in sorted(first_by_id):
        left, right = first_by_id[record_id], second_by_id[record_id]
        agreements = {
            "scope": left["scope"] == right["scope"],
            "nature_primary": left["nature_primary"] == right["nature_primary"],
            "nature_ambiguity": left["nature_ambiguity"] == right["nature_ambiguity"],
            "modality": left["modality"] == right["modality"],
        }
        agreements["full_3_axis_tuple"] = (
            left["scope"], left["nature_primary"], left["modality"]
        ) == (right["scope"], right["nature_primary"], right["modality"])
        for axis in agreements:
            agreement_counts[axis] += int(agreements[axis])
            by_dataset[left["dataset"]][axis] += int(agreements[axis])
        for axis in confusion:
            if left[axis] != right[axis]:
                pair = " <-> ".join(sorted((str(left[axis]), str(right[axis]))))
                confusion[axis][pair] += 1
        label_keys = (
            "scope", "nature_primary", "nature_secondary", "nature_ambiguity", "modality",
            "confidence", "reason_short", "taxonomy_failure", "taxonomy_failure_reason",
        )
        output.append(
            {
                "record_id": record_id,
                "question_id": left["question_id"],
                "question": left["question"],
                "dataset": left["dataset"],
                "sample_type": left["sample_type"],
                "stability_stratum": left.get("stability_stratum"),
                "pass_1": {key: left[key] for key in label_keys},
                "pass_2": {key: right[key] for key in label_keys},
                "agreements": agreements,
                "any_disagreement": not all(agreements.values()),
            }
        )
    count = len(output)
    metrics = {
        "schema_version": "planner-taxonomy-stability-metrics-v1",
        "count": count,
        "agreement": {key: value / count for key, value in agreement_counts.items()},
        "disagreement": {key: 1 - value / count for key, value in agreement_counts.items()},
        "by_dataset": {
            dataset: {
                "count": dataset_count,
                "agreement": {
                    key: counter[key] / dataset_count if dataset_count else None
                    for key in agreement_counts
                },
            }
            for dataset, counter in by_dataset.items()
            for dataset_count in [sum(row["dataset"] == dataset for row in output)]
        },
        "confusion_pairs": {
            axis: dict(counter.most_common()) for axis, counter in confusion.items()
        },
    }
    return output, metrics


def cross_axis_report(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in annotations if row["sample_type"] == "representative"]
    tuple_counts = Counter(
        (row["scope"], row["nature_primary"], row["modality"]) for row in rows
    )
    tuples = []
    for combination, count in tuple_counts.most_common():
        matching = [
            row for row in rows
            if (row["scope"], row["nature_primary"], row["modality"]) == combination
        ]
        tuples.append(
            {
                "scope": combination[0],
                "nature": combination[1],
                "modality": combination[2],
                "count": count,
                "rate": count / len(rows),
                "mean_min_axis_confidence": mean(
                    min(float(value) for value in row["confidence"].values()) for row in matching
                ),
                "low_confidence_count": sum(
                    min(float(value) for value in row["confidence"].values()) < 0.70
                    for row in matching
                ),
            }
        )
    special = {
        "global_dynamic": sum(
            row["scope"] == "global" and row["nature_primary"] == "dynamic" for row in rows
        ),
        "local_dynamic": sum(
            row["scope"] == "local" and row["nature_primary"] == "dynamic" for row in rows
        ),
        "static_global": sum(
            row["scope"] == "global" and row["nature_primary"] == "static" for row in rows
        ),
        "modality_indeterminate": sum(
            row["modality"] == "indeterminate_from_question" for row in rows
        ),
        "nature_secondary_present": sum(row["nature_secondary"] is not None for row in rows),
    }
    axis_collapse = {}
    for axis, field in (("scope", "scope"), ("nature", "nature_primary"), ("modality", "modality")):
        distribution = count_distribution(rows, field)
        axis_collapse[axis] = {
            "dominant_class": distribution["dominant_class"],
            "dominant_rate": distribution["dominant_rate"],
            "nearly_constant_at_85_percent": bool((distribution["dominant_rate"] or 0) >= 0.85),
        }
    uncertain_rows = [row for row in rows if row["nature_primary"] == "uncertain"]
    return {
        "schema_version": "planner-taxonomy-cross-axis-v1",
        "representative_count": len(rows),
        "unique_combinations": len(tuple_counts),
        "combination_frequencies": tuples,
        "rare_combinations_count_le_2": [row for row in tuples if row["count"] <= 2],
        "review_worthy_combinations": [
            {
                "combination": [row["scope"], row["nature"], row["modality"]],
                "reason": "Rare or low-confidence combination; human review recommended, not automatically contradictory.",
            }
            for row in tuples
            if row["count"] <= 2 or (row["mean_min_axis_confidence"] or 0) < 0.70
        ],
        "special_questions": {
            key: {"count": value, "rate": value / len(rows)} for key, value in special.items()
        },
        "axis_collapse_check": axis_collapse,
        "nature_uncertain_label_diagnostic": {
            "count": len(uncertain_rows),
            "rate": len(uncertain_rows) / len(rows),
            "mean_nature_confidence": mean(float(row["confidence"]["nature"]) for row in uncertain_rows),
            "interpretation": "The uncertain class denotes representational uncertainty/mixed necessity; it must not be conflated with the numeric annotator-confidence field.",
        },
        "axis_dependence_normalized_mutual_information": {
            "scope_nature": normalized_mutual_information(rows, "scope", "nature_primary"),
            "scope_modality": normalized_mutual_information(rows, "scope", "modality"),
            "nature_modality": normalized_mutual_information(rows, "nature_primary", "modality"),
            "interpretation": "0 indicates categorical independence; 1 indicates perfect dependence. This is descriptive, not causal.",
        },
    }


def _policy_for(row: dict[str, Any]) -> dict[str, str]:
    scope = {
        "local": "narrow_temporal_search",
        "multi_event": "multiple_relation_aware_regions",
        "global": "broad_coverage",
        "unclear": "avoid_scope_pruning",
    }[row["scope"]]
    nature = {
        "static": "state_or_keyframe_oriented",
        "dynamic": "preserve_temporal_resolution_and_transitions",
        "uncertain": "avoid_nature_based_early_pruning",
    }[row["nature_primary"]]
    modality = {
        "visual": "visual_branch",
        "audio": "audio_branch",
        "audio_visual": "audio_visual_with_temporal_alignment",
        "indeterminate_from_question": "do_not_prune_modality_from_question",
    }[row["modality"]]
    return {"scope_policy": scope, "nature_policy": nature, "modality_policy": modality}


def routing_actionability_report(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in annotations if row["sample_type"] == "representative"]
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        taxonomy = (row["scope"], row["nature_primary"], row["modality"])
        policy = _policy_for(row)
        entry = observed.setdefault(
            taxonomy,
            {
                "taxonomy": {
                    "scope": taxonomy[0], "nature": taxonomy[1], "modality": taxonomy[2]
                },
                "provisional_policy": policy,
                "count": 0,
            },
        )
        entry["count"] += 1
    policy_keys = {
        tuple(value for value in entry["provisional_policy"].values()) for entry in observed.values()
    }
    unique_by_removed_axis = {}
    for removed in ("scope", "nature", "modality"):
        kept = [axis for axis in ("scope", "nature", "modality") if axis != removed]
        unique_by_removed_axis[removed] = len(
            {
                tuple(entry["taxonomy"][axis] for axis in kept)
                for entry in observed.values()
            }
        )
    return {
        "schema_version": "planner-taxonomy-routing-actionability-v1",
        "analysis_only": True,
        "must_not_be_used_as_runtime_routing": True,
        "observed_taxonomy_combinations": len(observed),
        "distinct_provisional_policy_combinations": len(policy_keys),
        "combinations": sorted(observed.values(), key=lambda row: (-row["count"], json.dumps(row["taxonomy"], sort_keys=True))),
        "combination_count_if_axis_removed": unique_by_removed_axis,
        "redundant_labels_or_axes_under_frozen_mapping": [],
        "combinations_with_no_clear_routing_implication": [],
        "actionability_caution": (
            "Every label has a distinct provisional string mapping by construction; this does not prove "
            "empirical utility. Conservative labels imply non-pruning rather than a specialized branch."
        ),
        "axis_policy_implications": {
            "scope": "Changes temporal search breadth/region relation requirement.",
            "nature": "Changes state-like versus transition-preserving evidence preference.",
            "modality": "Changes which modality branches can safely be considered or pruned.",
        },
        "limitations": [
            "The mapping is provisional analysis, not an implemented policy.",
            "Distinct string policies do not prove measurable downstream utility.",
            "Unclear/uncertain/indeterminate labels deliberately map to conservative non-pruning.",
        ],
    }


def assess_axes(
    distribution: dict[str, Any],
    stability: dict[str, Any],
    routing: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    combined = distribution["groups"]["combined_representative"]
    results = {}
    agreement_keys = {"scope": "scope", "nature": "nature_primary", "modality": "modality"}
    for axis in ("scope", "nature", "modality"):
        summary = combined[axis]
        agreement = float(stability["agreement"][agreement_keys[axis]])
        failure = float(combined["taxonomy_failure_rate"])
        dominant = float(summary["dominant_rate"] or 0)
        confidence = float(summary["mean_confidence"] or 0)
        if dominant >= thresholds["reject_dominant_class_rate"] or agreement < thresholds["reject_stability_below"]:
            status = "REJECT_OR_MERGE"
        elif (
            failure <= thresholds["supported_max_taxonomy_failure_rate"]
            and agreement >= thresholds["supported_min_stability_agreement"]
            and confidence >= thresholds["supported_min_mean_confidence"]
        ):
            status = "SUPPORTED"
        else:
            status = "NEEDS_REVISION"
        reasons = [
            f"representative mean confidence={confidence:.3f}",
            f"stability agreement={agreement:.3f}",
            f"dominant class rate={dominant:.3f}",
            f"taxonomy failure rate={failure:.3f}",
            routing["axis_policy_implications"][axis],
        ]
        if axis == "nature" and float(summary.get("ambiguity_rate") or 0) >= 0.20:
            reasons.append("Frequent static/dynamic overlap indicates primary/secondary semantics need review.")
            if status == "SUPPORTED":
                status = "NEEDS_REVISION"
        if axis == "modality" and float(summary.get("indeterminate_rate") or 0) >= 0.30:
            reasons.append("Question-only modality is often indeterminate; aggressive pruning would be unsafe.")
            if status == "SUPPORTED":
                status = "NEEDS_REVISION"
        results[axis] = {"assessment": status, "evidence": reasons}
    return results


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
