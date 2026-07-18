"""Build the fixed, video-diverse EgoSchema comparison mini-pilot manifest."""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


SELECTION_SEED = "egoschema-comparison-pilot-v1"
PILOT_SIZE = 25
PRIOR_LVNET_FRAME_ROOTS = (
    "frames_1fps_test",
    "frames_1fps_test2",
    "frames_1fps",
    "small_test_900frames/frames_900",
)


def question_type(question: str) -> str:
    """Return a deterministic descriptive bucket; it never uses gold text."""
    value = question.lower()
    rules = (
        (r"\bhow many\b|\bnumber of\b|\bcount(?:ing|ed|s)?\b", "counting"),
        (r"\bbefore\b|\bafter\b|\bfirst\b|\blast\b|\bthen\b|\bsequence\b|\bthroughout\b|\bchange\b|\buntil\b|\bbetween\b", "temporal_or_change"),
        (r"\bwhy\b|\bpurpose\b|\bobjective\b|\bcontribut|\bsignificance\b|\bdeduce\b|\binfer", "causal_goal_inference"),
        (r"\bwhere\b|\blocation\b", "spatial"),
        (r"\bwho\b|\bperson\b|\bcharacter\b|\bpeople\b|\binteraction", "person_interaction"),
        (r"\boverall\b|\boverarching\b|\bprimary\b|\bmain\b|\btheme\b|\bsummar", "global_summary"),
    )
    return next((label for pattern, label in rules if re.search(pattern, value)), "action_object_detail")


def selection_hash(value: str) -> str:
    """Stable cross-platform rank independent of file order and mtimes."""
    return hashlib.sha256(f"{SELECTION_SEED}|{value}".encode("utf-8")).hexdigest()


def select_cases(rows: list[dict[str, Any]], data_root: Path) -> list[dict[str, Any]]:
    """Select five cases per answer position with question-type diversity."""
    selected: list[dict[str, Any]] = []
    for label in map(str, range(5)):
        pool = [
            row
            for row in rows
            if row["answer"] == label
            and (data_root / "sample_500" / f"{row['q_uid']}.mp4").is_file()
        ]
        by_type: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in pool:
            by_type[question_type(row["question"])].append(row)
        for candidates in by_type.values():
            candidates.sort(key=lambda row: selection_hash(row["q_uid"]))
        chosen: list[dict[str, Any]] = []
        for category in sorted(by_type, key=lambda value: selection_hash(f"{label}|{value}")):
            if len(chosen) < PILOT_SIZE // 5:
                chosen.append(by_type[category][0])
        for row in sorted(pool, key=lambda item: selection_hash(item["q_uid"])):
            if len(chosen) >= PILOT_SIZE // 5:
                break
            if row not in chosen:
                chosen.append(row)
        selected.extend(chosen)
    selected.sort(key=lambda row: selection_hash(row["q_uid"]))
    if len(selected) != PILOT_SIZE or len({row["q_uid"] for row in selected}) != PILOT_SIZE:
        raise RuntimeError("Deterministic EgoSchema selection did not produce 25 unique videos")
    return selected


def probe_video(ffprobe: Path, path: Path) -> dict[str, Any]:
    """Read container metadata only; no media frames/audio are decoded."""
    completed = subprocess.run(
        [
            str(ffprobe), "-v", "error", "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,sample_rate,channels",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    streams = value.get("streams", [])
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video = [stream for stream in streams if stream.get("codec_type") == "video"]
    return {
        "duration_sec": float(value["format"]["duration"]),
        "video_codec": video[0].get("codec_name") if video else None,
        "audio_available": bool(audio),
        "audio_codec": audio[0].get("codec_name") if audio else None,
        "audio_sample_rate": int(audio[0]["sample_rate"]) if audio and audio[0].get("sample_rate") else None,
        "audio_channels": int(audio[0]["channels"]) if audio and audio[0].get("channels") else None,
    }


def frame_sets(data_root: Path, q_uid: str) -> list[dict[str, Any]]:
    """List existing extracted-frame sets without opening images."""
    output = []
    for relative in PRIOR_LVNET_FRAME_ROOTS:
        folder = data_root / relative / q_uid
        count = sum(1 for path in folder.glob("*.jpg") if not path.name.startswith("._")) if folder.is_dir() else 0
        if count:
            output.append({"path": f"{relative}/{q_uid}", "frame_count": count})
    return output


def existing_lvnet_case_ids(data_root: Path) -> set[str]:
    """Discover case IDs in saved LVNet JSONL artifacts."""
    output: set[str] = set()
    for path in data_root.rglob("*.jsonl"):
        if path.name.startswith("._") or (
            "lvnet_outputs" not in path.parts and "hks_outputs" not in path.parts
        ):
            continue
        for line in path.read_text(encoding="utf-8", errors="strict").splitlines():
            if line.strip():
                value = json.loads(line)
                if value.get("q_uid"):
                    output.add(str(value["q_uid"]))
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/manifests/egoschema_comparison_pilot.json"))
    parser.add_argument("--ffprobe", type=Path, default=Path(shutil.which("ffprobe") or "ffprobe"))
    parser.add_argument("--probe-all", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    source = data_root / "egoschema_val.json"
    rows = json.loads(source.read_text(encoding="utf-8"))
    selected = select_cases(rows, data_root)
    probe_rows = rows if args.probe_all else selected

    def probe(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        path = data_root / "sample_500" / f"{row['q_uid']}.mp4"
        return row["q_uid"], probe_video(args.ffprobe, path)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        metadata = dict(executor.map(probe, probe_rows))
    prior_lvnet = existing_lvnet_case_ids(data_root)
    cases = []
    labels: dict[str, dict[str, Any]] = {}
    for source_row in selected:
        q_uid = source_row["q_uid"]
        media = metadata[q_uid]
        options = [source_row[f"option{index}"] for index in range(5)]
        cases.append(
            {
                "case_id": q_uid,
                "video_id": q_uid,
                "google_drive_id": source_row["google_drive_id"],
                "question": source_row["question"],
                "options": options,
                "question_type": question_type(source_row["question"]),
                "video_path": f"sample_500/{q_uid}.mp4",
                **media,
                "extracted_frame_sets": frame_sets(data_root, q_uid),
                "existing_lvnet_artifact_available": q_uid in prior_lvnet,
                "selection_hash_sha256": selection_hash(q_uid),
            }
        )
        index = int(source_row["answer"])
        labels[q_uid] = {"label_index": index, "correct_option": options[index]}
    answer_counts = collections.Counter(value["label_index"] for value in labels.values())
    type_counts = collections.Counter(row["question_type"] for row in cases)
    all_audio = sum(value["audio_available"] for value in metadata.values())
    manifest = {
        "manifest_version": "egoschema-comparison-pilot-v1",
        "dataset": "EgoSchema validation subset (500 labelled questions/videos)",
        "data_root_env": "EGOSCHEMA_DATA_ROOT",
        "source_metadata_path": "egoschema_val.json",
        "source_metadata_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selection_policy": {
            "seed": SELECTION_SEED,
            "pilot_size": PILOT_SIZE,
            "distinct_video_priority": "one question per q_uid; all selected videos unique",
            "answer_position_balance": "five cases for each label index 0–4",
            "question_type_diversity": "deterministic regex buckets; one per available type before hash fill within each answer label",
            "uses_prediction_correctness": False,
            "uses_prior_lvnet_or_ours_results": False,
            "ordering": "ascending SHA-256(seed|q_uid)",
        },
        "selection_audit": {
            "source_question_count": len(rows),
            "source_distinct_video_count": len({row["q_uid"] for row in rows}),
            "raw_video_count": len(list((data_root / "sample_500").glob("*.mp4"))),
            "selected_case_count": len(cases),
            "selected_distinct_video_count": len({row["video_id"] for row in cases}),
            "answer_label_counts": {str(key): answer_counts[key] for key in range(5)},
            "question_type_counts": dict(sorted(type_counts.items())),
            "selected_with_existing_extracted_frames": sum(bool(row["extracted_frame_sets"]) for row in cases),
            "selected_with_existing_lvnet_artifacts": sum(row["existing_lvnet_artifact_available"] for row in cases),
            "videos_probed_for_audio": len(metadata),
            "videos_with_audio": all_audio,
        },
        "protocols": {
            "A": {
                "id": "protocol_a_standard_multiple_choice",
                "retrieval_visible": ["question", "options"],
                "final_selection_visible": ["question", "options", "selected_evidence"],
                "gold_visible": "posthoc_only",
            },
            "B": {
                "id": "protocol_b_question_only_retrieval",
                "retrieval_visible": ["question"],
                "final_selection_visible": ["question", "options", "selected_evidence"],
                "gold_visible": "posthoc_only",
            },
        },
        "comparison_metrics": [
            "mc_accuracy",
            "selected_frame_ids_and_timestamps",
            "selected_clip_ids_and_intervals",
            "total_model_facing_frames",
            "video_coverage_and_evidence_budget",
            "warm_online_latency_sec",
            "offline_preprocessing_latency_sec_separate",
            "api_and_model_calls",
            "final_selected_option_index_and_text",
        ],
        "cases": cases,
        "posthoc_evaluation": {
            "access_policy": "load only after raw prediction and validated prediction are saved",
            "labels_by_case_id": labels,
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["selection_audit"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
