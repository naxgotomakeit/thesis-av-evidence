from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CREATION_DATE = "2026-07-22"
SELECTION_SEED = 20260722
MCQ_CLASSES = ("1s", "10s", "60s")
DURATION_BIN_ORDER = ("le_300", "301_600", "601_1200", "gt_1200")
QUESTION_TARGETS = {"1s": 2, "10s": 2, "60s": 1}
EXCLUSIONS = {
    "pasadena/bMMuC": (
        "Used for iterative Oracle-5 and B0-5 smoke/debug inference before the formal ablation freeze."
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SEED}|{namespace}|{value}".encode()).hexdigest()


def _hamilton(counts: Mapping[str, int], total: int, namespace: str) -> dict[str, int]:
    population = sum(counts.values())
    if population <= 0 or total > population:
        raise ValueError("Invalid Hamilton allocation")
    exact = {name: total * count / population for name, count in counts.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda name: (-(exact[name] - quotas[name]), stable_key(namespace, name)),
    )
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def select_ablation20(parent: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent_rows = parent.get("videos")
    if not isinstance(parent_rows, list) or len(parent_rows) != 50:
        raise ValueError("Parent manifest must contain exactly 50 videos")
    eligible = [row for row in parent_rows if row["video_id"] not in EXCLUSIONS]
    source_counts = Counter(str(row["source_collection"]) for row in eligible)
    source_quotas = _hamilton(source_counts, 20, "source-quota")
    stratum_quotas: dict[str, dict[str, int]] = {}
    selected: list[dict[str, Any]] = []
    for source in sorted(source_quotas):
        bin_counts = Counter(
            str(row["duration_bin"])
            for row in eligible
            if row["source_collection"] == source
        )
        bin_quotas = _hamilton(bin_counts, source_quotas[source], f"bin-quota:{source}")
        stratum_quotas[source] = {
            bin_name: bin_quotas.get(bin_name, 0) for bin_name in DURATION_BIN_ORDER
        }
        for bin_name in DURATION_BIN_ORDER:
            pool = sorted(
                (
                    row for row in eligible
                    if row["source_collection"] == source and row["duration_bin"] == bin_name
                ),
                key=lambda row: stable_key("video-selection", str(row["video_id"])),
            )
            selected.extend(pool[: bin_quotas.get(bin_name, 0)])
    if len(selected) != 20 or len({row["video_id"] for row in selected}) != 20:
        raise ValueError("Ablation selection must contain exactly 20 unique videos")
    selected.sort(
        key=lambda row: (
            str(row["source_collection"]),
            DURATION_BIN_ORDER.index(str(row["duration_bin"])),
            stable_key("video-selection", str(row["video_id"])),
        )
    )
    allocation = {
        "eligible_video_count": len(eligible),
        "eligible_source_counts": dict(sorted(source_counts.items())),
        "source_quotas": dict(sorted(source_quotas.items())),
        "stratum_quotas": stratum_quotas,
    }
    return selected, allocation


def _video_summary(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "selection_rank": rank,
        "video_id": row["video_id"],
        "source_collection": row["source_collection"],
        "duration_lower_bound_sec": float(row["duration_lower_bound_sec"]),
        "duration_lower_bound_basis": row["duration_lower_bound_basis"],
        "duration_bin": row["duration_bin"],
        "question_count_by_type": {
            duration_class: int(row["question_count_by_type"][duration_class])
            for duration_class in MCQ_CLASSES
        },
        "question_count": int(row["question_count"]),
        "mcq_metadata_files": [f"mcq_{duration_class}.json" for duration_class in MCQ_CLASSES],
        "relative_video_path": row["relative_video_path"],
    }


def build_video_manifest(parent: dict[str, Any], parent_path: Path) -> dict[str, Any]:
    selected, allocation = select_ablation20(parent)
    counts = Counter()
    for row in selected:
        counts.update(row["question_count_by_type"])
    per_video = [int(row["question_count"]) for row in selected]
    return {
        "schema_version": "egopolice-ablation20-v1",
        "creation_date": CREATION_DATE,
        "parent_frozen_50_manifest": {
            "path": parent_path.as_posix(),
            "sha256": sha256_file(ROOT / parent_path),
        },
        "selection_seed": SELECTION_SEED,
        "selection_rule": {
            "candidate_universe": "Only videos in the parent frozen-50 manifest.",
            "exclusion_then_stratification": (
                "Exclude documented result-driven smoke/debug videos, then stratify by parent "
                "source_collection and parent duration_lower_bound_bin."
            ),
            "quota_allocation": (
                "Two-stage Hamilton largest-remainder apportionment: first source collection, then "
                "duration bin within source. Equal remainders are ordered by the seeded SHA-256 key."
            ),
            "within_stratum_rank": "SHA256('20260722|video-selection|<video_id>'), ascending.",
            "duration_note": (
                "Duration strata use maximum MCQ annotation-end lower bounds from the frozen parent, "
                "not local download status or verified media duration."
            ),
            "downloaded_status_used": False,
            "model_results_used": False,
            "question_answers_or_options_used": False,
            "individual_gt_interval_values_used": False,
        },
        "exclusions": [
            {"video_id": video_id, "reason": reason} for video_id, reason in EXCLUSIONS.items()
        ],
        "allocation": allocation,
        "statistics": {
            "video_count": 20,
            "distinct_video_count": 20,
            "source_collection_counts": dict(Counter(row["source_collection"] for row in selected)),
            "duration_bin_counts": dict(Counter(row["duration_bin"] for row in selected)),
            "associated_question_count": sum(per_video),
            "associated_question_count_by_type": {
                duration_class: counts[duration_class] for duration_class in MCQ_CLASSES
            },
            "questions_per_video": {
                "min": min(per_video),
                "max": max(per_video),
                "mean": statistics.fmean(per_video),
                "median": statistics.median(per_video),
            },
        },
        "videos": [_video_summary(row, rank) for rank, row in enumerate(selected, start=1)],
    }


def load_mcq_metadata(data_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for duration_class in MCQ_CLASSES:
        path = data_root / f"mcq_{duration_class}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        indexed: dict[str, dict[str, Any]] = {}
        for row in rows:
            question_id = str(row["id"])
            if question_id in indexed:
                raise ValueError(f"Duplicate official question ID: {question_id}")
            indexed[question_id] = row
        output[duration_class] = indexed
    return output


def select_question_ids(
    video_manifest: dict[str, Any], metadata: Mapping[str, Mapping[str, dict[str, Any]]],
) -> tuple[dict[str, list[tuple[str, str]]], list[dict[str, Any]]]:
    by_video: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for duration_class in MCQ_CLASSES:
        for question_id, row in metadata[duration_class].items():
            video_id = str(PurePosixPath(str(row["video"])).with_suffix(""))
            by_video[video_id][duration_class].append(question_id)

    selections: dict[str, list[tuple[str, str]]] = {}
    fallbacks: list[dict[str, Any]] = []
    for video in video_manifest["videos"]:
        video_id = str(video["video_id"])
        available = by_video[video_id]
        selected: list[tuple[str, str]] = []
        selected_ids: set[str] = set()
        shortfalls: dict[str, int] = {}
        for duration_class in MCQ_CLASSES:
            ranked = sorted(
                available.get(duration_class, []),
                key=lambda question_id: stable_key("question-selection", question_id),
            )
            target = QUESTION_TARGETS[duration_class]
            chosen = ranked[:target]
            selected.extend((duration_class, question_id) for question_id in chosen)
            selected_ids.update(chosen)
            if len(chosen) < target:
                shortfalls[duration_class] = target - len(chosen)
        remaining = sorted(
            (
                (duration_class, question_id)
                for duration_class in MCQ_CLASSES
                for question_id in available.get(duration_class, [])
                if question_id not in selected_ids
            ),
            key=lambda item: stable_key("question-fallback", item[1]),
        )
        fill_count = min(5 - len(selected), len(remaining))
        selected.extend(remaining[:fill_count])
        if shortfalls or len(selected) < 5:
            fallbacks.append({
                "video_id": video_id,
                "ideal_class_shortfalls": shortfalls,
                "fallback_fill_count": fill_count,
                "selected_question_count": len(selected),
                "reason": (
                    "One or more target classes had fewer available questions; seeded remaining IDs "
                    "filled slots where possible."
                ),
            })
        selections[video_id] = selected
    return selections, fallbacks


def build_question_manifest(
    video_manifest: dict[str, Any], video_manifest_path: Path,
    metadata: Mapping[str, Mapping[str, dict[str, Any]]], data_root: Path,
) -> dict[str, Any]:
    selections, fallbacks = select_question_ids(video_manifest, metadata)
    questions: list[dict[str, Any]] = []
    for video in video_manifest["videos"]:
        video_id = str(video["video_id"])
        for duration_class, question_id in selections[video_id]:
            row = metadata[duration_class][question_id]
            options = [str(value) for value in row["options"]]
            answer = int(row["answer"])
            questions.append({
                "question_id": question_id,
                "video_id": video_id,
                "duration_class": duration_class,
                "gt_interval_sec": [float(row["start second"]), float(row["end second"])],
                "metadata_source_file": f"mcq_{duration_class}.json",
                "question": str(row["question"]),
                "options": options,
                "ground_truth_index": answer,
                "ground_truth_text": options[answer],
            })
    counts = Counter(row["duration_class"] for row in questions)
    per_video = Counter(row["video_id"] for row in questions)
    return {
        "schema_version": "egopolice-ablation-questions-v1",
        "creation_date": CREATION_DATE,
        "parent_ablation20_manifest": {
            "path": video_manifest_path.as_posix(),
            "sha256": sha256_file(ROOT / video_manifest_path),
        },
        "official_metadata_sha256": {
            f"mcq_{duration_class}.json": sha256_file(data_root / f"mcq_{duration_class}.json")
            for duration_class in MCQ_CLASSES
        },
        "selection_seed": SELECTION_SEED,
        "selection_rule": {
            "target_per_video": 5,
            "ideal_class_targets": QUESTION_TARGETS,
            "within_class_rank": "SHA256('20260722|question-selection|<question_id>'), ascending.",
            "fallback": (
                "If a class cannot meet its target, rank every remaining question ID for that video "
                "by SHA256('20260722|question-fallback|<question_id>') and fill to five; if fewer "
                "than five total questions exist, retain all available questions."
            ),
            "selection_reads_question_text": False,
            "selection_reads_options": False,
            "selection_reads_ground_truth_answer": False,
            "selection_reads_model_results": False,
            "selection_reads_gt_interval_values": False,
            "note": "Question/option/GT fields are copied only after IDs have been selected.",
        },
        "statistics": {
            "question_count": len(questions),
            "distinct_question_count": len({row["question_id"] for row in questions}),
            "distinct_video_count": len(per_video),
            "question_count_by_duration_class": {
                duration_class: counts[duration_class] for duration_class in MCQ_CLASSES
            },
            "questions_per_video": dict(per_video),
            "fallback_video_count": len(fallbacks),
        },
        "fallback_cases": fallbacks,
        "questions": questions,
    }


def _probe_duration(path: Path, ffprobe_path: str) -> float | None:
    if not path.is_file():
        return None
    completed = subprocess.run(
        [ffprobe_path, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def write_audit_csvs(
    *, parent: dict[str, Any], video_manifest: dict[str, Any],
    question_manifest: dict[str, Any], data_root: Path, output_dir: Path,
    ffprobe_path: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_ids = {row["video_id"] for row in video_manifest["videos"]}
    eligible_fields = [
        "video_id", "source_collection", "duration_lower_bound_sec", "duration_bin",
        "verified_source_duration_sec", "question_count_1s", "question_count_10s",
        "question_count_60s", "total_question_count", "downloaded", "selected_ablation20",
    ]
    with (output_dir / "egopolice_ablation20_eligible.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=eligible_fields, lineterminator="\n")
        writer.writeheader()
        for row in parent["videos"]:
            if row["video_id"] in EXCLUSIONS:
                continue
            media_path = data_root.joinpath(*PurePosixPath(row["relative_video_path"]).parts)
            writer.writerow({
                "video_id": row["video_id"],
                "source_collection": row["source_collection"],
                "duration_lower_bound_sec": row["duration_lower_bound_sec"],
                "duration_bin": row["duration_bin"],
                "verified_source_duration_sec": _probe_duration(media_path, ffprobe_path),
                "question_count_1s": row["question_count_by_type"]["1s"],
                "question_count_10s": row["question_count_by_type"]["10s"],
                "question_count_60s": row["question_count_by_type"]["60s"],
                "total_question_count": row["question_count"],
                "downloaded": media_path.is_file(),
                "selected_ablation20": row["video_id"] in selected_ids,
            })

    video_fields = [
        "selection_rank", "video_id", "source_collection", "duration_lower_bound_sec",
        "duration_bin", "question_count_1s", "question_count_10s", "question_count_60s",
        "total_question_count",
    ]
    with (output_dir / "egopolice_ablation20_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=video_fields, lineterminator="\n")
        writer.writeheader()
        for row in video_manifest["videos"]:
            writer.writerow({
                **{name: row[name] for name in video_fields[:5]},
                "question_count_1s": row["question_count_by_type"]["1s"],
                "question_count_10s": row["question_count_by_type"]["10s"],
                "question_count_60s": row["question_count_by_type"]["60s"],
                "total_question_count": row["question_count"],
            })

    question_fields = [
        "video_id", "selected_question_count", "selected_1s", "selected_10s", "selected_60s",
        "fallback_used", "fallback_reason",
    ]
    question_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in question_manifest["questions"]:
        question_counts[row["video_id"]][row["duration_class"]] += 1
    fallback_by_video = {row["video_id"]: row for row in question_manifest["fallback_cases"]}
    with (output_dir / "egopolice_ablation_questions_v1_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=question_fields, lineterminator="\n")
        writer.writeheader()
        for video in video_manifest["videos"]:
            video_id = video["video_id"]
            counts = question_counts[video_id]
            fallback = fallback_by_video.get(video_id)
            writer.writerow({
                "video_id": video_id,
                "selected_question_count": sum(counts.values()),
                "selected_1s": counts["1s"],
                "selected_10s": counts["10s"],
                "selected_60s": counts["60s"],
                "fallback_used": fallback is not None,
                "fallback_reason": fallback["reason"] if fallback else "",
            })


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze EgoPolice ablation20 v1 without model results")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--parent-manifest", type=Path,
        default=ROOT / "config/data/egopolice_50videos.json",
    )
    parser.add_argument(
        "--video-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation20_v1.json",
    )
    parser.add_argument(
        "--question-manifest", type=Path,
        default=ROOT / "config/data/egopolice_ablation_questions_v1.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/data_audit")
    parser.add_argument("--ffprobe-path", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    parent = json.loads(args.parent_manifest.read_text(encoding="utf-8"))
    parent_relative = args.parent_manifest.relative_to(ROOT)
    video_manifest = build_video_manifest(parent, parent_relative)
    args.video_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.video_manifest.write_text(
        json.dumps(video_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = load_mcq_metadata(args.data_root)
    video_relative = args.video_manifest.relative_to(ROOT)
    question_manifest = build_question_manifest(
        video_manifest, video_relative, metadata, args.data_root
    )
    args.question_manifest.write_text(
        json.dumps(question_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_audit_csvs(
        parent=parent,
        video_manifest=video_manifest,
        question_manifest=question_manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        ffprobe_path=args.ffprobe_path,
    )
    print(json.dumps({
        "video_manifest": str(args.video_manifest),
        "video_manifest_sha256": sha256_file(args.video_manifest),
        "question_manifest": str(args.question_manifest),
        "question_manifest_sha256": sha256_file(args.question_manifest),
        "video_statistics": video_manifest["statistics"],
        "question_statistics": question_manifest["statistics"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
