"""Prepare, annotate, and report the isolated planner taxonomy experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.planner_taxonomy.analysis import (  # noqa: E402
    assess_axes,
    cross_axis_report,
    distribution_report,
    routing_actionability_report,
    stability_report,
    write_json,
)
from src.experiments.planner_taxonomy.core import (  # noqa: E402
    freeze_samples,
    load_question_only_sources,
    select_stability_subset,
    sha256_file,
    sha256_text,
)
from src.experiments.planner_taxonomy.provider import (  # noqa: E402
    SYSTEM_RUBRIC_A,
    SYSTEM_RUBRIC_B,
    AnthropicTaxonomyAnnotator,
    annotation_json_schema,
    configured_anthropic,
)
from src.experiments.planner_taxonomy.reporting import render_review_html  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config/experiments/planner_taxonomy_validation_v0_1.json"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/experiments/planner_taxonomy_validation_v0_1"
DOC_PATH = PROJECT_ROOT / "docs/experiments/planner_taxonomy_validation_v0_1.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def records_from_sample(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    return list(payload["questions"])


def prepare() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    sources = load_question_only_sources(config["source_files"])
    representative, stress = freeze_samples(
        sources,
        seed=int(config["sampling_seed"]),
        representative_per_dataset=int(config["representative_per_dataset"]),
        stress_max_per_dataset=int(config["stress_max_per_dataset"]),
    )
    source_identity = {
        dataset: {
            "path": config["source_files"][dataset],
            "sha256": sha256_file(Path(config["source_files"][dataset])),
            "available_question_count": len(sources[dataset]),
        }
        for dataset in ("egoschema", "egosound")
    }
    rep_payload = {
        "schema_version": "planner-taxonomy-question-sample-v1",
        "purpose": "natural_distribution_estimation",
        "sampling_method": "deterministic SHA-256 rank by frozen seed; no balancing",
        "sampling_seed": config["sampling_seed"],
        "question_count": len(representative),
        "questions": representative,
    }
    stress_payload = {
        "schema_version": "planner-taxonomy-question-sample-v1",
        "purpose": "taxonomy_boundary_stress_only_not_prevalence",
        "selection_method": "deterministic question-text-only lexical cue bucket round-robin",
        "sampling_seed": config["sampling_seed"],
        "question_count": len(stress),
        "questions": stress,
    }
    atomic_json(OUTPUT_ROOT / "representative_sample.json", rep_payload)
    atomic_json(OUTPUT_ROOT / "diversity_stress_sample.json", stress_payload)
    manifest = {
        "schema_version": "planner-taxonomy-frozen-question-manifest-v1",
        "experiment": config["experiment_name"],
        "sampling_seed": config["sampling_seed"],
        "question_only_contract": {
            "allowed_provider_fields": ["opaque_item_id", "question"],
            "forbidden": [
                "gold_answer", "answer_options", "correctness", "gold_timestamp",
                "dataset_identity", "video", "audio", "retrieved_evidence",
            ],
        },
        "source_identity": source_identity,
        "dataset_modality_availability_audit_metadata": config["dataset_modality_availability"],
        "counts": {
            "representative": len(representative),
            "diversity_stress": len(stress),
            "total_first_pass": len(representative) + len(stress),
            "stability": 0,
        },
        "questions": [*representative, *stress],
        "stability_subset": [],
    }
    atomic_json(OUTPUT_ROOT / "frozen_question_manifest.json", manifest)
    model = os.environ.get(config["model_env"])
    frozen_prompt = {
        "schema_version": "planner-taxonomy-frozen-rubric-v1",
        "provider": config["provider"],
        "model": model,
        "temperature": config["temperature"],
        "max_output_tokens": config["max_output_tokens"],
        "batch_size": config["annotation_batch_size"],
        "pass_1_system_rubric": SYSTEM_RUBRIC_A,
        "pass_2_system_rubric": SYSTEM_RUBRIC_B,
        "pass_1_rubric_sha256": sha256_text(SYSTEM_RUBRIC_A),
        "pass_2_rubric_sha256": sha256_text(SYSTEM_RUBRIC_B),
        "output_schema": annotation_json_schema(config["annotation_batch_size"]),
        "question_only_payload_contract": "opaque item_id + question text; no dataset or other metadata",
    }
    atomic_json(OUTPUT_ROOT / "frozen_prompt_and_rubric.json", frozen_prompt)
    return manifest


def _journal_rows(pass_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    annotations: list[dict[str, Any]] = []
    usage: list[dict[str, Any]] = []
    for path in sorted((OUTPUT_ROOT / "api_journal").glob(f"{pass_name}_batch_*.json")):
        journal = read_json(path)
        if journal.get("state") != "validated":
            continue
        annotations.extend(journal["annotations"])
        usage.append({**journal["usage"], "pass": pass_name, "batch_index": journal["batch_index"]})
    return annotations, usage


def _run_annotation_batches(
    *,
    rows: list[dict[str, Any]],
    pass_name: str,
    rubric: str,
    max_new_batches: int | None,
) -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    model, api_key = configured_anthropic()
    if model != read_json(OUTPUT_ROOT / "frozen_prompt_and_rubric.json")["model"]:
        raise RuntimeError("Configured model differs from the frozen experiment model")
    annotator = AnthropicTaxonomyAnnotator(
        model=model,
        api_key=api_key,
        max_tokens=int(config["max_output_tokens"]),
        temperature=float(config["temperature"]),
    )
    batch_size = int(config["annotation_batch_size"])
    new_calls = 0
    for batch_index, start in enumerate(range(0, len(rows), batch_size)):
        batch = rows[start : start + batch_size]
        journal_path = OUTPUT_ROOT / "api_journal" / f"{pass_name}_batch_{batch_index:03d}.json"
        if max_new_batches is not None and new_calls >= max_new_batches and not journal_path.exists():
            continue
        result = annotator.annotate_batch(
            rows=batch,
            system_rubric=rubric,
            pass_name=pass_name,
            batch_index=batch_index,
            journal_dir=OUTPUT_ROOT / "api_journal",
        )
        new_calls += int(result.api_call_made)
    annotations, usage = _journal_rows(pass_name)
    expected_ids = {row["record_id"] for row in rows}
    actual_ids = [row["record_id"] for row in annotations]
    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(f"Duplicate validated annotations in {pass_name}")
    return {
        "pass": pass_name,
        "expected_questions": len(rows),
        "validated_questions": len(annotations),
        "complete": set(actual_ids) == expected_ids,
        "new_api_calls": new_calls,
        "validated_api_calls": len(usage),
    }


def annotate_pass1(max_new_batches: int | None) -> dict[str, Any]:
    rows = [
        *records_from_sample(OUTPUT_ROOT / "representative_sample.json"),
        *records_from_sample(OUTPUT_ROOT / "diversity_stress_sample.json"),
    ]
    result = _run_annotation_batches(
        rows=rows,
        pass_name="pass1",
        rubric=SYSTEM_RUBRIC_A,
        max_new_batches=max_new_batches,
    )
    annotations, _ = _journal_rows("pass1")
    annotations = sorted(annotations, key=lambda row: row["record_id"])
    write_jsonl(OUTPUT_ROOT / "planner_annotations.jsonl", annotations)
    if result["complete"]:
        config = read_json(CONFIG_PATH)
        subset = select_stability_subset(
            annotations,
            per_dataset=int(config["stability_per_dataset"]),
            seed=int(config["sampling_seed"]),
        )
        atomic_json(
            OUTPUT_ROOT / "stability_subset.json",
            {
                "schema_version": "planner-taxonomy-stability-subset-v1",
                "selection": "deterministic high/low/ambiguous plus label coverage after pass 1",
                "question_count": len(subset),
                "questions": subset,
            },
        )
        manifest = read_json(OUTPUT_ROOT / "frozen_question_manifest.json")
        manifest["counts"]["stability"] = len(subset)
        manifest["stability_subset"] = [
            {
                "record_id": row["record_id"],
                "question_id": row["question_id"],
                "text_sha256": row["text_sha256"],
                "dataset": row["dataset"],
                "stability_stratum": row["stability_stratum"],
            }
            for row in subset
        ]
        atomic_json(OUTPUT_ROOT / "frozen_question_manifest.json", manifest)
    return result


def annotate_pass2(max_new_batches: int | None) -> dict[str, Any]:
    subset_path = OUTPUT_ROOT / "stability_subset.json"
    if not subset_path.exists():
        raise RuntimeError("Pass 1 must be complete before freezing the stability subset")
    rows = records_from_sample(subset_path)
    return _run_annotation_batches(
        rows=rows,
        pass_name="pass2",
        rubric=SYSTEM_RUBRIC_B,
        max_new_batches=max_new_batches,
    )


def runtime_metrics() -> dict[str, Any]:
    _, pass1 = _journal_rows("pass1")
    _, pass2 = _journal_rows("pass2")
    calls = [*pass1, *pass2]
    journals = [
        read_json(path)
        for path in sorted((OUTPUT_ROOT / "api_journal").glob("*.json"))
    ]
    definitive_rejections = sum(
        journal.get("state") == "definitive_rejection" for journal in journals
    )
    ambiguous_attempts = sum(journal.get("state") in {"pending", "sent"} for journal in journals)
    provider_attempts = len(calls) + definitive_rejections + ambiguous_attempts
    return {
        "schema_version": "planner-taxonomy-runtime-cost-v1",
        "provider": calls[0]["provider"] if calls else "anthropic",
        "model": calls[0]["model"] if calls else os.environ.get("ANTHROPIC_MODEL"),
        "pass_1_api_calls": len(pass1),
        "pass_2_api_calls": len(pass2),
        "total_api_calls": provider_attempts,
        "successful_annotation_calls": len(calls),
        "definitive_pre_execution_rejections": definitive_rejections,
        "ambiguous_provider_attempts": ambiguous_attempts,
        "total_input_tokens": sum(int(row["input_tokens"]) for row in calls),
        "total_output_tokens": sum(int(row["output_tokens"]) for row in calls),
        "total_tokens": sum(int(row["input_tokens"]) + int(row["output_tokens"]) for row in calls),
        "total_api_wall_clock_sec": sum(float(row["latency_sec"]) for row in calls),
        "mean_api_call_latency_sec": (
            sum(float(row["latency_sec"]) for row in calls) / len(calls) if calls else None
        ),
        "estimated_cost_usd": None,
        "estimated_cost_note": "Unavailable: repository has no reliable Anthropic pricing table.",
        "questions_first_pass": len(_journal_rows("pass1")[0]),
        "questions_second_pass": len(_journal_rows("pass2")[0]),
        "token_usage_covers_successful_annotation_calls_only": True,
        "billing_note": "Three setup attempts were rejected with HTTP 400 before any model response; billing status is not inferred locally.",
        "final_qa_calls": 0,
        "video_or_audio_model_calls": 0,
    }


def _structural_diagnostic(annotations: list[dict[str, Any]]) -> dict[str, Any]:
    """Use prior cached segmentation metrics only where exact question IDs overlap."""
    prior = PROJECT_ROOT / "outputs/experiments/hierarchical_refinement_comparison_v0_1/per_case_metrics.json"
    if not prior.exists():
        return {"available": False, "reason": "Prior hierarchical metrics not found", "label": "STRUCTURAL DIAGNOSTIC ONLY"}
    payload = read_json(prior)
    rows = payload if isinstance(payload, list) else payload.get("cases", payload.get("per_case", []))
    by_question = {
        str(row.get("question_id") or row.get("q_uid") or row.get("case_id")): row
        for row in rows
        if row.get("question_id") or row.get("q_uid") or row.get("case_id")
    }
    overlaps = []
    for annotation in annotations:
        if annotation["dataset"] != "egoschema" or annotation["question_id"] not in by_question:
            continue
        source = by_question[annotation["question_id"]]
        overlaps.append(
            {
                "record_id": annotation["record_id"],
                "scope": annotation["scope"],
                "nature": annotation["nature_primary"],
                "cached_structure": source,
            }
        )
    return {
        "available": bool(overlaps),
        "label": "STRUCTURAL DIAGNOSTIC ONLY — CLIP/KTS/CoMET outputs are not temporal ground truth",
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }


def analyze() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    annotations, _ = _journal_rows("pass1")
    second, _ = _journal_rows("pass2")
    expected_first = int(config["representative_per_dataset"] + config["stress_max_per_dataset"]) * 2
    expected_second = int(config["stability_per_dataset"]) * 2
    if len(annotations) != expected_first or len(second) != expected_second:
        raise RuntimeError(
            f"Cannot analyze incomplete annotations: pass1={len(annotations)}/{expected_first}, pass2={len(second)}/{expected_second}"
        )
    subset_ids = {row["record_id"] for row in records_from_sample(OUTPUT_ROOT / "stability_subset.json")}
    first_subset = [row for row in annotations if row["record_id"] in subset_ids]
    stability_annotations, stability_metrics = stability_report(first_subset, second)
    distribution = distribution_report(annotations)
    cross_axis = cross_axis_report(annotations)
    routing = routing_actionability_report(annotations)
    assessments = assess_axes(
        distribution,
        stability_metrics,
        routing,
        config["assessment_thresholds"],
    )
    runtime = runtime_metrics()
    failures = [row for row in annotations if row["taxonomy_failure"]]
    write_json(OUTPUT_ROOT / "taxonomy_failures.json", {
        "schema_version": "planner-taxonomy-failures-v1", "count": len(failures), "failures": failures
    })
    write_json(OUTPUT_ROOT / "aggregate_distribution.json", distribution)
    write_json(OUTPUT_ROOT / "cross_axis_analysis.json", {**cross_axis, "axis_assessments": assessments})
    write_json(OUTPUT_ROOT / "stability_annotations.json", {
        "schema_version": "planner-taxonomy-stability-annotations-v1",
        "count": len(stability_annotations), "annotations": stability_annotations
    })
    write_json(OUTPUT_ROOT / "stability_metrics.json", stability_metrics)
    write_json(OUTPUT_ROOT / "routing_actionability_analysis.json", routing)
    write_json(OUTPUT_ROOT / "runtime_cost_metrics.json", runtime)
    structural = _structural_diagnostic(annotations)
    write_json(OUTPUT_ROOT / "structural_diagnostic.json", structural)
    render_review_html(
        output_path=OUTPUT_ROOT / "review.html",
        annotations=annotations,
        distribution=distribution,
        cross_axis=cross_axis,
        stability_metrics=stability_metrics,
        stability_annotations=stability_annotations,
        routing=routing,
        axis_assessments=assessments,
        runtime=runtime,
    )
    _write_readme(distribution, stability_metrics, cross_axis, assessments, runtime, structural)
    return {
        "annotations": len(annotations),
        "stability": len(second),
        "axis_assessments": assessments,
        "runtime": runtime,
    }


def _write_readme(
    distribution: dict[str, Any],
    stability: dict[str, Any],
    cross_axis: dict[str, Any],
    assessments: dict[str, Any],
    runtime: dict[str, Any],
    structural: dict[str, Any],
) -> None:
    combined = distribution["groups"]["combined_representative"]
    text = f"""# Planner taxonomy validation v0.1

This isolated teacher experiment tests whether question-only evidence requirements can be described by temporal scope, evidence nature, and modality requirement. Labels are hypotheses, not ground truth, and no routing was implemented.

## Leakage boundary

Provider input contained only opaque item IDs and question text. Gold answers, EgoSchema options, correctness, timestamps, dataset identity, video, audio, and retrieved evidence were excluded. No final QA was run.

## Samples

- Representative: 100 EgoSchema + 100 EgoSound (natural-distribution estimate).
- Diversity stress: 50 EgoSchema + 50 EgoSound (boundary stress only; not prevalence).
- Stability: 20 EgoSchema + 20 EgoSound independently reannotated.

## Provider

- Provider/model: {runtime['provider']} / {runtime['model']}
- Provider request attempts: {runtime['total_api_calls']} ({runtime['successful_annotation_calls']} successful annotation calls; {runtime['definitive_pre_execution_rejections']} pre-execution schema rejections)
- Input/output tokens: {runtime['total_input_tokens']} / {runtime['total_output_tokens']}
- API wall-clock: {runtime['total_api_wall_clock_sec']:.3f} s
- Estimated cost: unavailable because the repository has no maintained provider price table.

## Representative headline

- Taxonomy failures: {combined['taxonomy_failure_count']}/{combined['count']}
- Scope stability: {stability['agreement']['scope']:.3f}
- Nature-primary stability: {stability['agreement']['nature_primary']:.3f}
- Modality stability: {stability['agreement']['modality']:.3f}
- Full tuple stability: {stability['agreement']['full_3_axis_tuple']:.3f}
- Observed tuples: {cross_axis['unique_combinations']}
- Axis assessments: {json.dumps({axis: value['assessment'] for axis, value in assessments.items()}, ensure_ascii=False)}

## Structural diagnostic

{structural.get('label', structural.get('reason'))}; overlap count={structural.get('overlap_count', 0)}. This diagnostic uses cached segmentation outputs only and is not QA evidence or ground truth.

Open `review.html` for filterable question-by-question manual review. Browser-entered review fields are stored locally in the browser and do not alter experiment JSON.
"""
    (OUTPUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def status() -> dict[str, Any]:
    config = read_json(CONFIG_PATH)
    first_expected = 2 * (int(config["representative_per_dataset"]) + int(config["stress_max_per_dataset"]))
    second_expected = 2 * int(config["stability_per_dataset"])
    first_calls = math.ceil(first_expected / int(config["annotation_batch_size"]))
    second_calls = math.ceil(second_expected / int(config["annotation_batch_size"]))
    first, first_usage = _journal_rows("pass1")
    second, second_usage = _journal_rows("pass2")
    return {
        "provider": config["provider"],
        "model": os.environ.get(config["model_env"]),
        "api_key_present": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "first_pass_questions": first_expected,
        "stability_questions": second_expected,
        "batch_size": config["annotation_batch_size"],
        "estimated_initial_calls": first_calls + second_calls,
        "estimated_first_pass_calls": first_calls,
        "estimated_stability_calls": second_calls,
        "validated_first_pass_questions": len(first),
        "validated_second_pass_questions": len(second),
        "validated_calls": len(first_usage) + len(second_usage),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "status", "annotate-pass1", "annotate-pass2", "analyze"))
    parser.add_argument("--max-new-batches", type=int)
    args = parser.parse_args()
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    if args.command == "prepare":
        result = prepare()
        print(json.dumps(result["counts"], indent=2))
    elif args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "annotate-pass1":
        print(json.dumps(annotate_pass1(args.max_new_batches), indent=2))
    elif args.command == "annotate-pass2":
        print(json.dumps(annotate_pass2(args.max_new_batches), indent=2))
    else:
        print(json.dumps(analyze(), indent=2))


if __name__ == "__main__":
    main()
