#!/usr/bin/env python3
"""Run only the pinned official QaEgo4D OpenQA lexical metrics.

This runner deliberately has no model, CUDA, semantic-judge, or network path.
It imports the authors' pinned ``eval/eval.py`` directly and refuses to use a
locally reimplemented metric.  The source prediction JSONL is read only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO = Path(__file__).resolve().parents[2]
EXPECTED_CONDITIONS = ("blind", "uniform_8", "oracle_leq8")
EXPECTED_RUN_ID = "qaego4d-e1-open-fp16-0db47cc05774-ab895660428e"
EXPECTED_EVALUATOR_SHA256 = "fd483c1c191a4d57c10c92124d6e84d8e5d6607b1115e34de586d97fe8fc1e04"
EXPECTED_OFFICIAL_COMMIT = "bc1e29d3138a7db4ba35e19d9c1f05302e0c34c3"
MAIN_METRICS = (
    "plain_acc",
    "BLEU",
    "ROUGE.rouge1.f",
    "ROUGE.rouge2.f",
    "ROUGE.rougeL.f",
    "METEOR",
)
ADDITIVE_METRICS = (
    "plain_acc",
    "ROUGE.rouge1.f",
    "ROUGE.rouge2.f",
    "ROUGE.rougeL.f",
    "METEOR",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=REPO / "outputs/experiments/qaego4d_e1_open_fp16_v1/open_e1_formal_predictions.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "outputs/experiments/qaego4d_e1_open_fp16_v1/official_open_evaluation",
    )
    parser.add_argument(
        "--official-source",
        type=Path,
        required=True,
        help="Read-only checkout of lbaermann/qaego4d at the pinned commit.",
    )
    parser.add_argument(
        "--nltk-data",
        type=Path,
        required=True,
        help="Controlled cache holding only resources required by official METEOR.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--cluster-bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--bootstrap-chunk-size",
        type=int,
        default=25,
        help="Atomic checkpoint granularity. A crash repeats at most one chunk per bootstrap mode.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def import_official_evaluator(source_root: Path, nltk_data: Path):
    evaluator_path = source_root / "eval/eval.py"
    if not evaluator_path.is_file():
        raise RuntimeError(f"Pinned evaluator is absent: {evaluator_path}")
    actual_hash = sha256_file(evaluator_path)
    if actual_hash != EXPECTED_EVALUATOR_SHA256:
        raise RuntimeError(
            "Official evaluator SHA256 mismatch: "
            f"expected {EXPECTED_EVALUATOR_SHA256}, got {actual_hash}"
        )
    completed = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    commit = completed.stdout.strip()
    if commit != EXPECTED_OFFICIAL_COMMIT:
        raise RuntimeError(f"Official checkout mismatch: expected {EXPECTED_OFFICIAL_COMMIT}, got {commit}")

    os.environ["NLTK_DATA"] = str(nltk_data)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    import nltk

    # Do not fall back to any uncontrolled user/system NLTK cache.
    nltk.data.path[:] = [str(nltk_data)]
    for resource in ("corpora/wordnet.zip", "corpora/omw-1.4.zip"):
        nltk.data.find(resource)

    from eval import eval as official_eval

    imported_path = Path(official_eval.__file__).resolve()
    if imported_path != evaluator_path.resolve():
        raise RuntimeError(f"Imported non-pinned evaluator: {imported_path}")
    return official_eval, evaluator_path, actual_hash, commit


def read_and_validate_records(predictions: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(predictions.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            raise RuntimeError(f"Blank JSONL record at line {line_number}")
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Malformed JSON at line {line_number}: {exc}") from exc
        records.append(record)

    if len(records) != 5550:
        raise RuntimeError(f"Expected 5550 formal records, found {len(records)}")

    by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    malformed = []
    formal_configuration_tuples = set()
    for record in records:
        sample_id = record.get("sample_id")
        condition = record.get("condition")
        if not isinstance(sample_id, str) or not sample_id:
            malformed.append("missing sample_id")
            continue
        if condition not in EXPECTED_CONDITIONS:
            malformed.append(f"{sample_id}: unexpected condition {condition!r}")
            continue
        if condition in by_sample[sample_id]:
            raise RuntimeError(f"Duplicate successful sample_id x condition record: {sample_id} x {condition}")
        generated = record.get("generated_answer")
        reference = record.get("reference_answer")
        question = record.get("question")
        clip_uid = record.get("clip_uid")
        if not all(isinstance(item, str) and item.strip() for item in (generated, reference, question, clip_uid)):
            malformed.append(f"{sample_id} x {condition}: empty/malformed answer/question/clip_uid")
            continue
        if record.get("run_id") != EXPECTED_RUN_ID:
            malformed.append(f"{sample_id} x {condition}: non-formal run_id {record.get('run_id')!r}")
            continue
        if record.get("backend") != "huggingface" or record.get("dtype") != "float16" or record.get("max_pixels") != 262144:
            malformed.append(f"{sample_id} x {condition}: backend/dtype/max_pixels provenance mismatch")
            continue
        if record.get("api_calls") not in (0, None):
            malformed.append(f"{sample_id} x {condition}: nonzero API calls in formal generation metadata")
            continue
        formal_configuration_tuples.add(
            (
                record.get("run_id"),
                record.get("config_hash"),
                record.get("manifest_hash"),
                record.get("backend"),
                record.get("dtype"),
                record.get("max_pixels"),
                record.get("context_scope"),
            )
        )
        by_sample[sample_id][condition] = record

    if malformed:
        raise RuntimeError("Input integrity failure: " + "; ".join(malformed[:10]))
    if len(by_sample) != 1850:
        raise RuntimeError(f"Expected 1850 unique sample_ids, found {len(by_sample)}")
    missing = {
        sample_id: sorted(set(EXPECTED_CONDITIONS) - set(condition_rows))
        for sample_id, condition_rows in by_sample.items()
        if set(condition_rows) != set(EXPECTED_CONDITIONS)
    }
    if missing:
        first = next(iter(missing.items()))
        raise RuntimeError(f"Missing condition(s), first example: {first}")
    if len(formal_configuration_tuples) != 1:
        raise RuntimeError(
            "Formal configuration provenance is inconsistent / possibly mixed: "
            f"{sorted(formal_configuration_tuples)!r}"
        )

    ordered_ids = sorted(by_sample)
    reference_mismatches = []
    metadata_mismatches = []
    for sample_id in ordered_ids:
        rows = by_sample[sample_id]
        reference_answers = {rows[condition]["reference_answer"] for condition in EXPECTED_CONDITIONS}
        questions = {rows[condition]["question"] for condition in EXPECTED_CONDITIONS}
        clips = {rows[condition]["clip_uid"] for condition in EXPECTED_CONDITIONS}
        if len(reference_answers) != 1:
            reference_mismatches.append(sample_id)
        if len(questions) != 1 or len(clips) != 1:
            metadata_mismatches.append(sample_id)
    if reference_mismatches or metadata_mismatches:
        raise RuntimeError(
            f"Cross-condition formal metadata mismatch: references={reference_mismatches[:5]}, "
            f"question_or_clip={metadata_mismatches[:5]}"
        )

    config_tuple = next(iter(formal_configuration_tuples))
    integrity = {
        "prediction_rows": len(records),
        "unique_sample_ids": len(ordered_ids),
        "count_by_condition": {condition: len(ordered_ids) for condition in EXPECTED_CONDITIONS},
        "duplicate_successful_sample_condition_records": 0,
        "missing_sample_condition_records": 0,
        "malformed_prediction_records": 0,
        "matched_triplets": len(ordered_ids),
        "formal_configuration": {
            "run_id": config_tuple[0],
            "config_hash": config_tuple[1],
            "manifest_hash": config_tuple[2],
            "backend": config_tuple[3],
            "dtype": config_tuple[4],
            "max_pixels": config_tuple[5],
            "context_scope": config_tuple[6],
        },
    }
    return by_sample, integrity


def prepare_condition_inputs(by_sample: dict[str, dict[str, dict[str, Any]]]) -> tuple[list[str], dict[str, list[str]], list[list[str]], list[str]]:
    sample_ids = sorted(by_sample)
    references = [[by_sample[sample_id]["blind"]["reference_answer"]] for sample_id in sample_ids]
    clip_uids = [by_sample[sample_id]["blind"]["clip_uid"] for sample_id in sample_ids]
    predictions = {
        condition: [by_sample[sample_id][condition]["generated_answer"] for sample_id in sample_ids]
        for condition in EXPECTED_CONDITIONS
    }
    return sample_ids, predictions, references, clip_uids


def metrics_for_conditions(official_eval: Any, predictions: dict[str, list[str]], references: list[list[str]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for condition in EXPECTED_CONDITIONS:
        official = official_eval.calc_metrics(predictions[condition], references)
        missing = [metric for metric in MAIN_METRICS if metric not in official]
        if missing:
            raise RuntimeError(f"Pinned official evaluator omitted required metrics: {missing}")
        result[condition] = {key: float(value) for key, value in official.items()}
    return result


def additive_values(official_eval: Any, predictions: dict[str, list[str]], references: list[list[str]]) -> dict[str, dict[str, np.ndarray]]:
    """Compute values whose official aggregations are arithmetic means.

    This does not substitute a new metric: for one-reference samples, the
    official _calc_accuracy, _calc_rouge, and _calc_meteor implementations are
    arithmetic means.  Their exact per-item contributions make resampling their
    official aggregate mathematically identical while avoiding invalid
    per-question significance claims for corpus BLEU.
    """
    result: dict[str, dict[str, np.ndarray]] = {}
    for condition in EXPECTED_CONDITIONS:
        values = {metric: [] for metric in ADDITIVE_METRICS}
        for prediction, reference in zip(predictions[condition], references):
            # Direct calls remain the pinned evaluator's own implementation.
            rouge = official_eval._calc_rouge([prediction], [reference])
            meteor = official_eval._calc_meteor([prediction], [reference])
            values["plain_acc"].append(float(official_eval._calc_accuracy([prediction], [reference])))
            values["ROUGE.rouge1.f"].append(float(rouge["rouge1"]["f"]))
            values["ROUGE.rouge2.f"].append(float(rouge["rouge2"]["f"]))
            values["ROUGE.rougeL.f"].append(float(rouge["rougeL"]["f"]))
            values["METEOR"].append(float(meteor))
        result[condition] = {metric: np.asarray(series, dtype=np.float64) for metric, series in values.items()}
    return result


def percentile_interval(values: np.ndarray) -> dict[str, float]:
    return {
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def bootstrap_chunk_values(
    official_eval: Any,
    predictions: dict[str, list[str]],
    references: list[list[str]],
    additive: dict[str, dict[str, np.ndarray]],
    index_sets: Iterable[np.ndarray],
    comparisons: list[tuple[str, str]],
) -> dict[str, dict[str, list[float]]]:
    distributions = {
        f"{first}_to_{second}": {metric: [] for metric in MAIN_METRICS}
        for first, second in comparisons
    }
    for indices in index_sets:
        # BLEU is corpus-level, so recompute the exact official corpus score.
        bleu = {
            condition: float(official_eval._calc_bleu([predictions[condition][int(i)] for i in indices], [references[int(i)] for i in indices])["BLEU"])
            for condition in EXPECTED_CONDITIONS
        }
        for first, second in comparisons:
            key = f"{first}_to_{second}"
            distributions[key]["BLEU"].append(bleu[second] - bleu[first])
            for metric in ADDITIVE_METRICS:
                distributions[key][metric].append(float(additive[second][metric][indices].mean() - additive[first][metric][indices].mean()))
    return distributions


def bootstrap_plan_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def deterministic_question_indices(seed: int, replicate_index: int, n_samples: int) -> np.ndarray:
    # Replicate-specific SeedSequence means resume does not depend on prior RNG state.
    generator = np.random.default_rng(np.random.SeedSequence([seed, 0, replicate_index]))
    return generator.integers(0, n_samples, size=n_samples, endpoint=False)


def deterministic_cluster_indices(seed: int, replicate_index: int, clip_groups: list[np.ndarray]) -> np.ndarray:
    generator = np.random.default_rng(np.random.SeedSequence([seed, 1, replicate_index]))
    drawn = generator.integers(0, len(clip_groups), size=len(clip_groups), endpoint=False)
    return np.concatenate([clip_groups[int(index)] for index in drawn])


def checkpointed_bootstrap(
    *,
    mode: str,
    total_replicates: int,
    chunk_size: int,
    checkpoint_root: Path,
    plan: dict[str, Any],
    official_eval: Any,
    predictions: dict[str, list[str]],
    references: list[list[str]],
    additive: dict[str, dict[str, np.ndarray]],
    comparisons: list[tuple[str, str]],
    index_for_replicate: Any,
) -> dict[str, Any]:
    """Run/resume a bootstrap in atomic chunks and return percentile summaries."""
    if chunk_size < 1:
        raise RuntimeError("bootstrap chunk size must be positive")
    mode_root = checkpoint_root / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    plan_path = mode_root / "plan.json"
    plan_with_signature = {
        **plan,
        "mode": mode,
        "total_replicates": total_replicates,
        "chunk_size": chunk_size,
    }
    signature = bootstrap_plan_signature(plan_with_signature)
    expected_plan = {**plan_with_signature, "signature": signature}
    if plan_path.exists():
        existing_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing_plan != expected_plan:
            raise RuntimeError(f"Existing {mode} bootstrap checkpoints have incompatible provenance: {plan_path}")
    else:
        atomic_write_json(plan_path, expected_plan)

    segments = [(start, min(start + chunk_size, total_replicates)) for start in range(0, total_replicates, chunk_size)]
    aggregate = {
        f"{first}_to_{second}": {metric: [] for metric in MAIN_METRICS}
        for first, second in comparisons
    }
    completed_chunks = 0
    for start, end in segments:
        checkpoint_path = mode_root / f"{start:06d}_{end:06d}.json"
        if checkpoint_path.exists():
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            expected_header = {"signature": signature, "mode": mode, "start": start, "end": end}
            if {key: payload.get(key) for key in expected_header} != expected_header:
                raise RuntimeError(f"Invalid checkpoint header: {checkpoint_path}")
            values = payload.get("values")
            if not isinstance(values, dict):
                raise RuntimeError(f"Invalid checkpoint payload: {checkpoint_path}")
            source = "reused"
        else:
            index_sets = [index_for_replicate(replicate_index) for replicate_index in range(start, end)]
            values = bootstrap_chunk_values(official_eval, predictions, references, additive, index_sets, comparisons)
            for comparison, metric_values in values.items():
                for metric, series in metric_values.items():
                    if len(series) != end - start:
                        raise RuntimeError(f"Incomplete {mode} checkpoint values for {comparison} / {metric}")
            atomic_write_json(
                checkpoint_path,
                {"signature": signature, "mode": mode, "start": start, "end": end, "values": values},
            )
            source = "computed"
        for comparison, metric_values in values.items():
            for metric, series in metric_values.items():
                aggregate[comparison][metric].extend(series)
        completed_chunks += 1
        progress = {
            "mode": mode,
            "signature": signature,
            "total_replicates": total_replicates,
            "chunk_size": chunk_size,
            "completed_chunks": completed_chunks,
            "total_chunks": len(segments),
            "completed_replicates": end,
            "last_chunk": [start, end],
            "last_chunk_source": source,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(mode_root / "progress.json", progress)
        print(
            f"bootstrap mode={mode} chunk={start}:{end} source={source} "
            f"completed={completed_chunks}/{len(segments)}",
            flush=True,
        )

    summary = {
        comparison: {
            metric: {"replicates": len(values), **percentile_interval(np.asarray(values, dtype=np.float64))}
            for metric, values in metric_values.items()
        }
        for comparison, metric_values in aggregate.items()
    }
    return {
        "summary": summary,
        "checkpoint_root": str(mode_root),
        "plan_signature": signature,
        "completed_chunks": completed_chunks,
        "total_chunks": len(segments),
    }


def mcnemar_exact_pvalue(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower_tail = sum(math.comb(discordant, k) for k in range(0, min(first_only, second_only) + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * lower_tail)


def transition_analysis(first: str, second: str, values: dict[str, dict[str, np.ndarray]], point_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    first_correct = values[first]["plain_acc"].astype(bool)
    second_correct = values[second]["plain_acc"].astype(bool)
    both_correct = int(np.sum(first_correct & second_correct))
    first_correct_second_wrong = int(np.sum(first_correct & ~second_correct))
    first_wrong_second_correct = int(np.sum(~first_correct & second_correct))
    both_wrong = int(np.sum(~first_correct & ~second_correct))
    return {
        "first_condition": first,
        "second_condition": second,
        "both_correct": both_correct,
        "first_correct_second_wrong": first_correct_second_wrong,
        "first_wrong_second_correct": first_wrong_second_correct,
        "both_wrong": both_wrong,
        "absolute_plain_acc_difference_second_minus_first": point_metrics[second]["plain_acc"] - point_metrics[first]["plain_acc"],
        "mcnemar_two_sided_exact_p": mcnemar_exact_pvalue(first_correct_second_wrong, first_wrong_second_correct),
    }


def write_pairs(path: Path, by_sample: dict[str, dict[str, dict[str, Any]]]) -> None:
    rows = []
    for sample_id in sorted(by_sample):
        for condition in EXPECTED_CONDITIONS:
            source = by_sample[sample_id][condition]
            rows.append(
                {
                    "sample_id": sample_id,
                    "clip_uid": source["clip_uid"],
                    "question": source["question"],
                    "reference_answer": source["reference_answer"],
                    "candidate_answer": source["generated_answer"],
                    "condition": condition,
                }
            )
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    atomic_write_text(path, text)


def write_condition_maps(output_dir: Path, sample_ids: list[str], predictions: dict[str, list[str]], references: list[list[str]]) -> dict[str, str]:
    input_dir = output_dir / "official_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(input_dir / "gold_answers.json", {sample_id: references[index][0] for index, sample_id in enumerate(sample_ids)})
    hashes = {"gold_answers.json": sha256_file(input_dir / "gold_answers.json")}
    for condition in EXPECTED_CONDITIONS:
        path = input_dir / f"{condition}_hypotheses.json"
        atomic_write_json(path, {sample_id: predictions[condition][index] for index, sample_id in enumerate(sample_ids)})
        hashes[path.name] = sha256_file(path)
    return hashes


def environment_artifact(output_dir: Path, evaluator_path: Path, evaluator_hash: str, commit: str, args: argparse.Namespace) -> dict[str, Any]:
    packages = {name: importlib.metadata.version(name) for name in ("nltk", "rouge-score", "sacrebleu", "numpy")}
    resources = []
    for resource in ("corpora/wordnet.zip", "corpora/omw-1.4.zip"):
        resources.append({"resource": resource, "path": str(args.nltk_data / resource), "sha256": sha256_file(args.nltk_data / resource)})
    payload = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment_prefix": sys.prefix,
        "packages": packages,
        "nltk_data_path": str(args.nltk_data),
        "nltk_resources": resources,
        "official_repository": "https://github.com/lbaermann/qaego4d",
        "official_commit": commit,
        "official_evaluator_path": str(evaluator_path),
        "official_evaluator_sha256": evaluator_hash,
        "exact_command": " ".join([shlex_quote(item) for item in sys.argv]),
    }
    lines = [
        "QaEgo4D official evaluator isolated environment",
        f"python={payload['python_version'].splitlines()[0]}",
        f"environment_prefix={payload['environment_prefix']}",
        *[f"{name}=={version}" for name, version in packages.items()],
        f"NLTK_DATA={payload['nltk_data_path']}",
        *[f"nltk_resource={item['resource']} sha256={item['sha256']}" for item in resources],
        f"official_repository={payload['official_repository']}",
        f"official_commit={commit}",
        f"official_evaluator_sha256={evaluator_hash}",
        f"command={payload['exact_command']}",
        "No semantic judge/API or VLM inference is part of this environment.",
    ]
    atomic_write_text(output_dir / "evaluator_environment.txt", "\n".join(lines) + "\n")
    return payload


def shlex_quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def format_metric(value: float) -> str:
    return f"{value:.6f}"


def report_markdown(summary: dict[str, Any], bootstrap: dict[str, Any]) -> str:
    metrics = summary["official_metrics"]
    lines = [
        "# Formal Open E1 — official QaEgo4D metrics",
        "",
        "**Scope:** deterministic benchmark-comparability metrics only. These lexical-overlap metrics are not semantic correctness, and Oracle<=8 is not an absolute upper bound.",
        "",
        "## Input integrity",
        "",
        f"- Formal rows: {summary['input_integrity']['prediction_rows']}/5550.",
        f"- Matched sample triplets: {summary['input_integrity']['matched_triplets']}/1850.",
        "- Duplicates, missing records, malformed predictions: 0 / 0 / 0.",
        "",
        "## Official metrics",
        "",
        "| Condition | plain_acc | SacreBLEU | ROUGE-1 F | ROUGE-2 F | ROUGE-L F | METEOR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in EXPECTED_CONDITIONS:
        result = metrics[condition]
        lines.append(
            "| " + condition + " | " + " | ".join(
                format_metric(result[key])
                for key in ("plain_acc", "BLEU", "ROUGE.rouge1.f", "ROUGE.rouge2.f", "ROUGE.rougeL.f", "METEOR")
            ) + " |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    for name, value in summary["paired_plain_acc"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- both correct: {value['both_correct']}",
                f"- first correct / second wrong: {value['first_correct_second_wrong']}",
                f"- first wrong / second correct: {value['first_wrong_second_correct']}",
                f"- both wrong: {value['both_wrong']}",
                f"- second − first plain_acc: {value['absolute_plain_acc_difference_second_minus_first']:.6f}",
                f"- exact two-sided McNemar p: {value['mcnemar_two_sided_exact_p']:.6g}",
                "",
            ]
        )
        for bootstrap_name in ("question_level_paired_bootstrap", "clip_uid_cluster_bootstrap"):
            ci = bootstrap[bootstrap_name][name]["plain_acc"]
            lines.append(f"- {bootstrap_name} 95% CI: [{ci['lower_95']:.6f}, {ci['upper_95']:.6f}].")
        lines.append("")
    lines.extend(
        [
            "## Provenance and boundary",
            "",
            f"- Official evaluator commit: `{summary['official_evaluator']['commit']}`.",
            f"- Official evaluator SHA256: `{summary['official_evaluator']['source_sha256']}`.",
            "- No semantic judge, external API, VLM inference, E2, B1, or B2 was run.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1 or args.cluster_bootstrap_replicates < 1:
        raise RuntimeError("Bootstrap replicate counts must be positive")
    source_prediction_hash_before = sha256_file(args.predictions)
    official_eval, evaluator_path, evaluator_hash, commit = import_official_evaluator(args.official_source, args.nltk_data)
    by_sample, integrity = read_and_validate_records(args.predictions)
    sample_ids, predictions, references, clip_uids = prepare_condition_inputs(by_sample)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_pairs(args.output_dir / "official_open_eval_pairs.jsonl", by_sample)
    input_hashes = write_condition_maps(args.output_dir, sample_ids, predictions, references)
    metrics = metrics_for_conditions(official_eval, predictions, references)
    additive = additive_values(official_eval, predictions, references)

    comparisons = [("uniform_8", "oracle_leq8"), ("blind", "uniform_8"), ("blind", "oracle_leq8")]
    comparison_names = [f"{first}_to_{second}" for first, second in comparisons]
    n = len(sample_ids)
    by_clip: dict[str, list[int]] = defaultdict(list)
    for index, clip_uid in enumerate(clip_uids):
        by_clip[clip_uid].append(index)
    clip_groups = [np.asarray(indices, dtype=np.int64) for _, indices in sorted(by_clip.items())]
    checkpoint_root = args.output_dir / "bootstrap_checkpoints"
    plan = {
        "schema_version": "official-open-eval-bootstrap-v1",
        "seed": args.seed,
        "comparison_direction": "second_condition_minus_first_condition",
        "comparisons": [list(item) for item in comparisons],
        "prediction_sha256": source_prediction_hash_before,
        "official_evaluator_sha256": evaluator_hash,
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "clip_uid_sequence_sha256": hashlib.sha256("\n".join(clip_uids).encode("utf-8")).hexdigest(),
        "n_samples": n,
        "n_unique_clip_uids": len(clip_groups),
    }
    question_bootstrap_result = checkpointed_bootstrap(
        mode="question_level_paired",
        total_replicates=args.bootstrap_replicates,
        chunk_size=args.bootstrap_chunk_size,
        checkpoint_root=checkpoint_root,
        plan=plan,
        official_eval=official_eval,
        predictions=predictions,
        references=references,
        additive=additive,
        comparisons=comparisons,
        index_for_replicate=lambda replicate_index: deterministic_question_indices(args.seed, replicate_index, n),
    )
    cluster_bootstrap_result = checkpointed_bootstrap(
        mode="clip_uid_cluster",
        total_replicates=args.cluster_bootstrap_replicates,
        chunk_size=args.bootstrap_chunk_size,
        checkpoint_root=checkpoint_root,
        plan=plan,
        official_eval=official_eval,
        predictions=predictions,
        references=references,
        additive=additive,
        comparisons=comparisons,
        index_for_replicate=lambda replicate_index: deterministic_cluster_indices(args.seed, replicate_index, clip_groups),
    )
    question_bootstrap = question_bootstrap_result["summary"]
    cluster_bootstrap = cluster_bootstrap_result["summary"]

    paired_plain_acc = {
        name: transition_analysis(first, second, additive, metrics)
        for name, (first, second) in zip(comparison_names, comparisons)
    }
    environment = environment_artifact(args.output_dir, evaluator_path, evaluator_hash, commit, args)
    source_prediction_hash_after = sha256_file(args.predictions)
    if source_prediction_hash_before != source_prediction_hash_after:
        raise RuntimeError("Frozen prediction file hash changed during evaluation; refusing to publish results")

    bootstrap = {
        "seed": args.seed,
        "comparison_direction": "second_condition_minus_first_condition",
        "question_level_paired_bootstrap": question_bootstrap,
        "clip_uid_cluster_bootstrap": cluster_bootstrap,
        "n_unique_clip_uids": len(clip_groups),
        "checkpointing": {
            "chunk_size": args.bootstrap_chunk_size,
            "question_level": {key: value for key, value in question_bootstrap_result.items() if key != "summary"},
            "clip_uid_cluster": {key: value for key, value in cluster_bootstrap_result.items() if key != "summary"},
            "resume_policy": "Reuse only atomic chunk checkpoints whose full provenance signature matches the input prediction hash, evaluator hash, seed, sample ordering, clip ordering, and comparison plan.",
        },
        "note": "BLEU is recomputed with the exact pinned official corpus implementation for every resample. plain_acc, ROUGE, and METEOR use exact per-sample contributions from the pinned official implementation because its aggregate is an arithmetic mean for this one-reference formal set.",
    }
    summary = {
        "status": "OFFICIAL_QAEGO4D_OPEN_METRICS_COMPLETE",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_integrity": integrity,
        "prediction_file": str(args.predictions),
        "prediction_file_sha256": source_prediction_hash_after,
        "official_evaluator": {
            "repository": "https://github.com/lbaermann/qaego4d",
            "commit": commit,
            "source_path": str(evaluator_path),
            "source_sha256": evaluator_hash,
            "implementation": "eval/eval.py:calc_metrics",
        },
        "official_metrics": metrics,
        "paired_plain_acc": paired_plain_acc,
        "bootstrap_file": "official_open_bootstrap_results.json",
        "official_input_hashes": input_hashes,
        "environment": environment,
        "execution_boundary": {
            "semantic_judge_calls": 0,
            "external_api_calls": 0,
            "vlm_inference_calls": 0,
            "E2_started": False,
            "B1_started": False,
            "B2_started": False,
        },
    }
    atomic_write_json(args.output_dir / "official_open_bootstrap_results.json", bootstrap)
    atomic_write_json(args.output_dir / "official_open_metrics_summary.json", summary)
    atomic_write_text(args.output_dir / "official_open_metrics_report.md", report_markdown(summary, bootstrap))
    print(json.dumps({
        "status": summary["status"],
        "summary": str(args.output_dir / "official_open_metrics_summary.json"),
        "report": str(args.output_dir / "official_open_metrics_report.md"),
        "pairs": str(args.output_dir / "official_open_eval_pairs.jsonl"),
        "prediction_sha256": source_prediction_hash_after,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
