#!/usr/bin/env python3
"""Prepare the gold-blind stratified EgoSound Pilot 20 Phase-1 artifacts."""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.pilot_reporting import pilot_report_schema, selection_html  # noqa: E402
from src.evaluation.pilot_selection import expected_modalities, select_case_ids, selection_tags, temporal_metadata  # noqa: E402
from src.evaluation.registry import baseline_metric_registry  # noqa: E402


SOURCE = ROOT / "data/manifests/egosound_manifest.jsonl"
EXISTING = ROOT / "data/manifests/mvp_cases_6.json"
MANIFEST = ROOT / "data/manifests/egosound_pilot_20.json"
SELECTION_AUDIT = ROOT / "data/manifests/egosound_pilot_20_selection_audit.json"
OUT = ROOT / "outputs/pilot_20/baseline_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    rows = load_jsonl(SOURCE)
    existing = json.loads(EXISTING.read_text(encoding="utf-8"))
    visual = {path.name for path in (ROOT / "outputs/visual_index").iterdir() if path.is_dir()}
    audio = {path.name for path in (ROOT / "outputs/audio_index").iterdir() if path.is_dir()}
    indexed = visual & audio
    case_ids, stratification = select_case_ids(rows, existing, indexed, 20)
    by_id = {str(row["case_id"]): row for row in rows}
    existing_ids = {str(row["case_id"]) for row in existing}
    selected: list[dict[str, Any]] = []
    audit_cases: list[dict[str, Any]] = []
    for case_id in case_ids:
        source = dict(by_id[case_id])
        temporal = temporal_metadata(source)
        modalities = expected_modalities(source)
        tags = selection_tags(source)
        source.update(temporal)
        source["expected_modality_stratum"] = modalities
        source["pilot_selection_tags"] = tags
        source["pilot_existing_development_case"] = case_id in existing_ids
        source["pilot_selection_version"] = "baseline_v1_phase1"
        selected.append(source)
        audit_cases.append({
            "case_id": case_id,
            "video_id": source["video_id"],
            "question": source["question"],
            "question_type": source["question_type"],
            "existing_development_case": case_id in existing_ids,
            **temporal,
            "expected_modalities": modalities,
            "selection_tags": tags,
            "selection_reason": f"Contributes to the {source['question_type']} quota; hint={temporal['temporal_hint_strength']}; expected modalities={','.join(modalities)}; tags={','.join(tags) or 'local_event'}.",
        })
    save_json(MANIFEST, selected)
    widths = [float(item["provided_timestamp_width_sec"]) for item in audit_cases if item["provided_timestamp_width_sec"] is not None]
    summary = {
        "phase": "Phase 1 selection and evaluation-scaffold validation; no live calls",
        "case_count": len(selected),
        "existing_case_count": sum(item["existing_development_case"] for item in audit_cases),
        "new_case_count": sum(not item["existing_development_case"] for item in audit_cases),
        "indexed_video_ids": sorted(indexed),
        "question_type_distribution": dict(sorted(Counter(item["question_type"] for item in audit_cases).items())),
        "expected_modality_distribution": dict(sorted(Counter(modality for item in audit_cases for modality in item["expected_modalities"]).items())),
        "temporal_hint_distribution": dict(sorted(Counter(item["temporal_hint_strength"] for item in audit_cases).items())),
        "provided_timestamp": {"available_count": sum(item["provided_timestamp_available"] for item in audit_cases), "available_fraction": sum(item["provided_timestamp_available"] for item in audit_cases) / len(audit_cases), "mean_width_sec": statistics.fmean(widths) if widths else None, "median_width_sec": statistics.median(widths) if widths else None},
        "selection_tag_distribution": dict(sorted(Counter(tag for item in audit_cases for tag in item["selection_tags"]).items())),
        "stratification": stratification,
        "gold_or_prediction_used_for_selection": False,
        "live_api_calls": 0,
        "offline_index_policy": "Reuse only the six already indexed videos; no offline index rebuild in Phase 1.",
    }
    audit = {"summary": summary, "cases": audit_cases}
    save_json(SELECTION_AUDIT, audit)
    OUT.mkdir(parents=True, exist_ok=True)
    save_json(OUT / "phase1_selection_audit.json", audit)
    registry = baseline_metric_registry()
    registry_status = {"registered_metrics": list(registry.names), "metric_count": len(registry.names), "placeholders": ["semantic_similarity", "llm_judge", "task_specific_metric"], "lexical_metric_category": "diagnostic_reference_overlap", "authoritative_accuracy_metric": None}
    save_json(OUT / "metric_registry_status.json", registry_status)
    save_json(OUT / "pilot_report_schema.json", pilot_report_schema())
    save_json(OUT / "failure_analysis_schema.json", {"status": "awaiting_live_results", "automatic_algorithm_changes": False, "categories": pilot_report_schema()["failure_categories"], "required_fields": ["case_id", "categories", "observations", "evidence_status", "answer_status", "human_review_required"]})
    (OUT / "phase1_selection_review.html").write_text(selection_html(audit_cases, summary), encoding="utf-8")
    lines = ["# EgoSound Pilot 20 — Phase 1", "", "No live API/model calls were made.", "", f"- Cases: {summary['case_count']} ({summary['existing_case_count']} existing + {summary['new_case_count']} new)", f"- Question types: `{summary['question_type_distribution']}`", f"- Expected modalities: `{summary['expected_modality_distribution']}`", f"- Temporal hints: `{summary['temporal_hint_distribution']}`", f"- Provided timestamp availability: {summary['provided_timestamp']['available_count']}/{summary['case_count']}", f"- Mean/median provided timestamp width: {summary['provided_timestamp']['mean_width_sec']:.3f}s / {summary['provided_timestamp']['median_width_sec']:.3f}s", f"- Selection tags: `{summary['selection_tag_distribution']}`", "", "Exact Match, normalized Exact Match, BLEU and exploratory single-reference CIDEr are registered only as diagnostic reference-overlap metrics. Semantic similarity, LLM judge and task-specific metrics remain explicitly unavailable.", ""]
    (OUT / "phase1_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"case_ids": case_ids, "summary": summary, "metric_registry": registry_status, "live_api_calls": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
