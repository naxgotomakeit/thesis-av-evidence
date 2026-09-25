from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


FORMAL_ROOT = Path(
    "/myriadfs/home/ucemxna/Scratch/workspace/external_baselines/"
    "gens_qwen25vl3b_top16_v1/outputs/"
    "formal_eval300_gens_hybrid_symmetric_mcq_cap16_v2_long_output_20260822T214500Z"
)
PROJECT_ROOT = FORMAL_ROOT.parents[1]
PROFILE_PATH = PROJECT_ROOT / "config/gens_hybrid_symmetric_mcq_cap16_v2_long_output.json"
PUBLIC_SELECTOR = Path(
    "/myriadfs/home/ucemxna/Scratch/workspace/datasets/hourvideo_eval300_v1/"
    "eval300_selector_input.jsonl"
)
FRAME_ROOT = Path("/myriadfs/home/ucemxna/Scratch/workspace/HourVideo/runtime")
PACKAGE_NAME = "gens_eval300_selector_v2_manifest_only"
EXPECTED_PROFILE_SHA = "3479770e71ff5f68f0669dc6b51ce3db1922fe8ae39b2f7a0ff493634aa8d0f6"
EXPECTED_PUBLIC_SELECTOR_SHA = "2545aca24b5e88e7e6eb021543d507a2eeba443e9e0c211be4f95ad70a41a4cd"
EXPECTED_V2_RAW_MANIFEST_SHA = "c8d0e8c366fc418054c52882e14a632a990f7d2fe40fa50287a27a72db7e0fe0"
EXPECTED_DERIVED_MANIFEST_SHA = "96a16c238a26f2da010d9cacae3fd0050f281559b31fc312a59b2b3efea882c1"
METHOD = "GenS-Hybrid-SymmetricMCQ-cap16-v2-long-output"
OPTION_KEYS = ("A", "B", "C", "D", "E")
BASE_PAYLOAD_FILES = (
    "selected_frames_downstream.jsonl",
    "selection_provenance.jsonl",
    "selected_frame_files_manifest.json",
    "eval300_uid_order.txt",
    "frozen_selector_metadata.json",
    "FINAL_SELECTOR_REPORT.json",
    "README.md",
)
FINAL_PACKAGE_FILES = tuple(sorted(BASE_PAYLOAD_FILES + (
    "MANIFEST.sha256", "CONTENTS.txt", "package_validation.json", "PACKAGE_REPORT.md",
)))
BANNED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg",
    ".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".npy", ".npz",
}
FORBIDDEN_JSON_KEYS = {
    "gold", "gold_label", "answer", "label", "prediction", "a_to_e_prediction",
    "a_to_e_predictions", "clip_winning_option_label", "winning_option",
}
SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "api_token": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "secret_assignment": re.compile(r"(?:OPENAI|ANTHROPIC|AZURE_OPENAI|GOOGLE)_API_KEY\s*="),
    "dotenv_file": re.compile(r"(?:^|[/\\])\.env(?:\.|$)"),
}
ABSOLUTE_PATH_PATTERN = re.compile(r"(?:^|[\s\"'=:])/(?:myriadfs|home|cs|scratch|tmp|var|opt)/")
FRAME_NAME_PATTERN = re.compile(r"^frame_[0-9]{5}\.jpg$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_relative_frame_path(video_id: str, filename: str) -> str:
    value = PurePosixPath(video_id) / filename
    if value.is_absolute() or ".." in value.parts or len(value.parts) != 2:
        raise RuntimeError(f"unsafe relative frame path: {value}")
    if value.parts[0] != video_id or value.parts[1] != filename:
        raise RuntimeError("relative frame path identity drift")
    return value.as_posix()


def stage_a_aggregate(records: list[dict[str, Any]]) -> tuple[str, dict[str, str]]:
    digest = hashlib.sha256()
    values: dict[str, str] = {}
    for record in records:
        uid = record["qa_uid"]
        value = sha256_file(FORMAL_ROOT / "questions" / uid / "clip_top256.json")
        values[uid] = value
        digest.update(uid.encode("utf-8") + b"\0" + value.encode("ascii") + b"\n")
    return digest.hexdigest(), values


def selected_identity(item: dict[str, Any]) -> tuple[str, int, float]:
    return item["video_id"], int(item["frame_index"]), float(item["timestamp_sec"])


def build_payload(package_dir: Path) -> dict[str, Any]:
    if sha256_file(PROFILE_PATH) != EXPECTED_PROFILE_SHA:
        raise RuntimeError("V2 profile SHA drift")
    if sha256_file(PUBLIC_SELECTOR) != EXPECTED_PUBLIC_SELECTOR_SHA:
        raise RuntimeError("frozen public Eval300 selector input SHA drift")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    final_report_source = json.loads((FORMAL_ROOT / "FINAL_SELECTOR_REPORT.json").read_text(encoding="utf-8"))
    if final_report_source.get("status") != "PASS" or final_report_source.get("legal_selected_frame_questions") != 300:
        raise RuntimeError("V2 final report is not a 300/300 PASS")

    records = jsonl(PUBLIC_SELECTOR)
    if len(records) != 300 or len({row["qa_uid"] for row in records}) != 300:
        raise RuntimeError("public gold-free selector must contain 300 unique UIDs")
    uids = [row["qa_uid"] for row in records]
    stage_a_sha, stage_a_per_uid = stage_a_aggregate(records)

    downstream_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    frame_files: dict[str, dict[str, Any]] = {}
    frame_references: defaultdict[str, set[str]] = defaultdict(set)
    source_checks = Counter()
    selected_count_distribution: Counter[int] = Counter()

    for record in records:
        uid = record["qa_uid"]
        video_id = record["video_id"]
        options = record.get("options")
        if not isinstance(record.get("question"), str) or not record["question"].strip():
            raise RuntimeError(f"missing public question: {uid}")
        if not isinstance(options, dict) or tuple(options.keys()) != OPTION_KEYS:
            raise RuntimeError(f"public options are not exactly ordered A-E: {uid}")
        if any(not isinstance(options[key], str) or not options[key].strip() for key in OPTION_KEYS):
            raise RuntimeError(f"empty public option: {uid}")

        question_dir = FORMAL_ROOT / "questions" / uid
        finished = json.loads((question_dir / "finished_at.json").read_text(encoding="utf-8"))
        if finished.get("status") != "ok":
            raise RuntimeError(f"V2 question is not terminal ok: {uid}")
        selected = json.loads((question_dir / "final_selected_frames.json").read_text(encoding="utf-8"))
        provenance = json.loads((question_dir / "selector_provenance.json").read_text(encoding="utf-8"))
        if not 1 <= len(selected) <= 16:
            raise RuntimeError(f"selected frame count outside 1-16: {uid}")
        if selected != sorted(selected, key=lambda item: (float(item["timestamp_sec"]), int(item["frame_index"]))):
            raise RuntimeError(f"final selected frames are not chronological: {uid}")
        if provenance.get("profile_sha256") != EXPECTED_PROFILE_SHA:
            raise RuntimeError(f"V2 per-question profile SHA drift: {uid}")
        if provenance.get("clip_top256_sha256") != stage_a_per_uid[uid]:
            raise RuntimeError(f"Stage A top-256 SHA drift: {uid}")

        relevance_order = provenance.get("gens_relevance_order")
        if not isinstance(relevance_order, list):
            raise RuntimeError(f"missing GenS relevance order: {uid}")
        relevance_index: dict[tuple[str, int, float], tuple[int, dict[str, Any]]] = {}
        for position, item in enumerate(relevance_order):
            identity = selected_identity(item)
            if identity in relevance_index:
                raise RuntimeError(f"duplicate frame in relevance order: {uid}")
            relevance_index[identity] = (position, item)

        downstream_frames = []
        provenance_frames = []
        for chronological_order, final_item in enumerate(selected):
            identity = selected_identity(final_item)
            if identity not in relevance_index:
                raise RuntimeError(f"selected frame absent from relevance order: {uid}")
            selection_order, source_item = relevance_index[identity]
            filename = source_item.get("cache_filename") or Path(source_item["extracted_frame_path"]).name
            if not FRAME_NAME_PATTERN.fullmatch(filename):
                raise RuntimeError(f"unexpected frame filename: {uid}/{filename}")
            relative_path = safe_relative_frame_path(video_id, filename)
            expected_source = FRAME_ROOT / video_id / "frames_1fps" / filename
            actual_source = Path(source_item["extracted_frame_path"])
            if actual_source != expected_source:
                raise RuntimeError(f"selected frame source path mismatch: {uid}/{filename}")
            if actual_source.is_symlink() or not actual_source.is_file():
                source_checks["missing_or_nonregular"] += 1
                raise RuntimeError(f"selected frame is missing, linked, or non-regular: {uid}/{filename}")
            actual_sha = sha256_file(actual_source)
            expected_sha = source_item.get("cache_sha256")
            if actual_sha != expected_sha:
                source_checks["hash_mismatch"] += 1
                raise RuntimeError(f"selected frame SHA mismatch: {uid}/{filename}")
            source_checks["verified_references"] += 1
            timestamp = float(source_item["timestamp_sec"])
            frame = {
                "selection_order": selection_order,
                "chronological_order": chronological_order,
                "frame_filename": filename,
                "relative_frame_path": relative_path,
                "timestamp_sec": timestamp,
                "frame_sha256": actual_sha,
            }
            downstream_frames.append(frame)
            provenance_frames.append({
                **frame,
                "clip_rank": int(source_item["clip_rank"]),
                "clip_score": float(source_item["clip_score"]),
                "gens_input_index": int(source_item["gens_input_index"]),
                "gens_relevance": int(source_item["gens_relevance_score"]),
                "original_selection_order": int(source_item["gens_response_order"]),
            })
            file_value = {
                "video_id": video_id,
                "frame_filename": filename,
                "relative_frame_path": relative_path,
                "timestamp_sec": timestamp,
                "size_bytes": actual_source.stat().st_size,
                "sha256": actual_sha,
            }
            if relative_path in frame_files and frame_files[relative_path] != file_value:
                raise RuntimeError(f"deduplicated frame metadata conflict: {relative_path}")
            frame_files[relative_path] = file_value
            frame_references[relative_path].add(uid)

        selection_sha = sha256_bytes(canonical_bytes(downstream_frames))
        downstream_rows.append({
            "qa_uid": uid,
            "video_id": video_id,
            "question": record["question"],
            "options": {key: options[key] for key in OPTION_KEYS},
            "selected_frame_count": len(downstream_frames),
            "selected_frames": downstream_frames,
        })
        provenance_rows.append({
            "qa_uid": uid,
            "video_id": video_id,
            "selected_frames": provenance_frames,
            "v2_profile_sha256": EXPECTED_PROFILE_SHA,
            "stage_a_top256_sha256": stage_a_per_uid[uid],
            "final_selection_sha256": selection_sha,
        })
        selected_count_distribution[len(selected)] += 1

    file_manifest = []
    for relative_path in sorted(frame_files):
        file_manifest.append({
            **frame_files[relative_path],
            "referenced_by_qa_count": len(frame_references[relative_path]),
        })

    write_jsonl(package_dir / "selected_frames_downstream.jsonl", downstream_rows)
    write_jsonl(package_dir / "selection_provenance.jsonl", provenance_rows)
    write_json(package_dir / "selected_frame_files_manifest.json", {
        "schema_version": 1,
        "unique_frame_count": len(file_manifest),
        "frames": file_manifest,
    })
    (package_dir / "eval300_uid_order.txt").write_text("".join(uid + "\n" for uid in uids), encoding="utf-8")

    final_report_sha = sha256_file(FORMAL_ROOT / "FINAL_SELECTOR_REPORT.json")
    metadata = {
        "schema_version": 1,
        "method": METHOD,
        "adaptation_disclosure": (
            "Symmetric per-option MCQ querying with max-over-options aggregation is a fairness "
            "adaptation and is not the GenS authors' publicly released original query implementation."
        ),
        "v2_profile_sha256": EXPECTED_PROFILE_SHA,
        "clip_revision": profile["clip"]["revision"],
        "gens_revision": profile["gens"]["revision"],
        "clip_symmetric_query_template_sha256": profile["symmetric_mcq"]["clip_query_template_sha256"],
        "gens_full_query_template_sha256": profile["gens_full_query"]["template_sha256"],
        "stage_a_aggregate_sha256": stage_a_sha,
        "stage_a_aggregate_definition": "SHA256 over frozen UID-order lines: qa_uid + NUL + clip_top256_file_sha256 + LF",
        "v2_raw_manifest_sha256": EXPECTED_V2_RAW_MANIFEST_SHA,
        "derived_result_manifest_sha256": EXPECTED_DERIVED_MANIFEST_SHA,
        "final_report_sha256": final_report_sha,
        "frame_cache_tree_sha256": profile["candidate_cache"]["total_tree_sha256"],
        "question_count": 300,
        "selected_frames_per_question": {"minimum": 1, "maximum": 16},
        "api_calls": 0,
        "restricted_evaluation_reads": 0,
        "image_files_in_package": 0,
    }
    write_json(package_dir / "frozen_selector_metadata.json", metadata)

    sanitized = dict(final_report_source)
    redacted = []
    for field, reason in (
        ("per_question", "removed as a large per-question telemetry payload; aggregate timings remain"),
        ("private_gold_reads", "renamed to a neutral restricted-read audit field"),
        ("A_to_E_predictions", "renamed to a neutral downstream-output audit field"),
    ):
        if field in sanitized:
            sanitized.pop(field)
            redacted.append({"field": field, "reason": reason})
    sanitized["restricted_evaluation_reads"] = 0
    sanitized["downstream_choice_outputs_generated"] = 0
    sanitized["redacted_fields"] = redacted
    sanitized["source_final_report_sha256"] = final_report_sha
    write_json(package_dir / "FINAL_SELECTOR_REPORT.json", sanitized)

    readme = """# GenS V2 Eval300 selector manifest-only package

This package contains GenS selector output, not final A-E predictions. It contains no images, video, model weights, embeddings, raw model responses, generated token IDs, or frame cache payloads.

Downstream usage:

1. Read `selected_frames_downstream.jsonl` in file order.
2. On the target machine, resolve each `relative_frame_path` against an existing frame-cache root.
3. Recompute every referenced file SHA-256 and compare it with `frame_sha256` before use.
4. If a same-named target frame has a different SHA-256, it must not be substituted; retrieve the original verified frame again.
5. Sort `selected_frames` by `chronological_order` before sending images downstream. The API receives only the question, options A-E, and those chronologically ordered images.

`selection_provenance.jsonl` is audit-only and must never be sent to the API. Its winning-option telemetry and per-option score vectors were deliberately removed for downstream isolation. Only aggregate CLIP rank/score and GenS selection telemetry remain.

No gold file may be read at any point. For comparisons with Uniform or Ours, use exactly the same API model, prompt, decoding, parser, and retry protocol.

The package does not copy frames. The manifest path `<video_id>/frame_NNNNN.jpg` is a portable logical path; the target deployment is responsible for mapping it to its local verified cache layout.
"""
    (package_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "records": records,
        "uids": uids,
        "downstream_rows": downstream_rows,
        "provenance_rows": provenance_rows,
        "file_manifest": file_manifest,
        "stage_a_aggregate_sha256": stage_a_sha,
        "selected_count_distribution": dict(sorted(selected_count_distribution.items())),
        "source_checks": dict(source_checks),
        "final_report_source_sha256": final_report_sha,
    }


def iter_json_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from iter_json_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_json_keys(child)


def text_security_scan(package_dir: Path) -> dict[str, Any]:
    absolute_hits = []
    secret_hits = []
    for path in sorted(package_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        if ABSOLUTE_PATH_PATTERN.search(text):
            absolute_hits.append(path.name)
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                secret_hits.append({"file": path.name, "pattern": name})
    return {"absolute_path_hits": absolute_hits, "secret_hits": secret_hits}


def forbidden_key_scan(package_dir: Path) -> list[dict[str, str]]:
    hits = []
    for name in (
        "selected_frames_downstream.jsonl", "selection_provenance.jsonl",
        "selected_frame_files_manifest.json", "frozen_selector_metadata.json",
        "FINAL_SELECTOR_REPORT.json", "package_validation.json",
    ):
        path = package_dir / name
        if not path.exists():
            continue
        values = jsonl(path) if path.suffix == ".jsonl" else [json.loads(path.read_text(encoding="utf-8"))]
        for value in values:
            for key in iter_json_keys(value):
                if key.lower() in FORBIDDEN_JSON_KEYS:
                    hits.append({"file": name, "key": key})
    return hits


def filesystem_scan(package_dir: Path) -> dict[str, Any]:
    images_video_models_embeddings = []
    symlinks = []
    hardlinks = []
    special = []
    files = []
    for path in sorted(package_dir.iterdir()):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            symlinks.append(path.name)
        elif stat.S_ISREG(mode):
            files.append(path.name)
            if path.suffix.lower() in BANNED_SUFFIXES:
                images_video_models_embeddings.append(path.name)
            if path.stat().st_nlink != 1:
                hardlinks.append(path.name)
        elif not stat.S_ISDIR(mode):
            special.append(path.name)
    return {
        "files": files,
        "image_video_model_embedding_files": images_video_models_embeddings,
        "symlinks": symlinks,
        "hardlinks": hardlinks,
        "special_files": special,
    }


def validate_scientific_payload(package_dir: Path, built: dict[str, Any]) -> dict[str, Any]:
    downstream = jsonl(package_dir / "selected_frames_downstream.jsonl")
    provenance = jsonl(package_dir / "selection_provenance.jsonl")
    uid_lines = (package_dir / "eval300_uid_order.txt").read_text(encoding="utf-8").splitlines()
    manifest = json.loads((package_dir / "selected_frame_files_manifest.json").read_text(encoding="utf-8"))
    expected_uids = built["uids"]
    result = {
        "question_count": len(downstream),
        "provenance_count": len(provenance),
        "uid_count": len(uid_lines),
        "uid_unique_count": len(set(uid_lines)),
        "uid_order_matches_frozen": uid_lines == expected_uids,
        "downstream_uid_order_matches": [row["qa_uid"] for row in downstream] == expected_uids,
        "provenance_uid_order_matches": [row["qa_uid"] for row in provenance] == expected_uids,
        "selected_count_out_of_range": 0,
        "unsafe_relative_paths": 0,
        "question_or_option_errors": 0,
        "forbidden_json_field_hits": [],
        "winning_option_field_hits": 0,
        "missing_source_frames": 0,
        "source_frame_hash_mismatches": 0,
        "source_frame_size_mismatches": 0,
        "unique_frame_count": manifest["unique_frame_count"],
        "missing_manifest_frames": 0,
        "extra_manifest_frames": 0,
    }
    record_index = {row["qa_uid"]: row for row in built["records"]}
    referenced = {}
    ref_counts: Counter[str] = Counter()
    for row in downstream:
        uid = row["qa_uid"]
        if not 1 <= row["selected_frame_count"] <= 16 or row["selected_frame_count"] != len(row["selected_frames"]):
            result["selected_count_out_of_range"] += 1
        public = record_index[uid]
        if row["question"] != public["question"] or tuple(row["options"].keys()) != OPTION_KEYS or row["options"] != public["options"]:
            result["question_or_option_errors"] += 1
        chronological = [frame["chronological_order"] for frame in row["selected_frames"]]
        if chronological != list(range(len(row["selected_frames"]))):
            result["question_or_option_errors"] += 1
        for frame in row["selected_frames"]:
            rel = PurePosixPath(frame["relative_frame_path"])
            if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 2:
                result["unsafe_relative_paths"] += 1
                continue
            source = FRAME_ROOT / row["video_id"] / "frames_1fps" / frame["frame_filename"]
            if not source.is_file() or source.is_symlink():
                result["missing_source_frames"] += 1
            elif sha256_file(source) != frame["frame_sha256"]:
                result["source_frame_hash_mismatches"] += 1
            referenced[frame["relative_frame_path"]] = frame["frame_sha256"]
            ref_counts[frame["relative_frame_path"]] += 1
    manifest_paths = {frame["relative_frame_path"] for frame in manifest["frames"]}
    result["missing_manifest_frames"] = len(set(referenced) - manifest_paths)
    result["extra_manifest_frames"] = len(manifest_paths - set(referenced))
    for frame in manifest["frames"]:
        source = FRAME_ROOT / frame["video_id"] / "frames_1fps" / frame["frame_filename"]
        if not source.is_file() or source.is_symlink():
            result["missing_source_frames"] += 1
        else:
            if sha256_file(source) != frame["sha256"]:
                result["source_frame_hash_mismatches"] += 1
            if source.stat().st_size != frame["size_bytes"]:
                result["source_frame_size_mismatches"] += 1
        if frame["referenced_by_qa_count"] != ref_counts[frame["relative_frame_path"]]:
            result["source_frame_size_mismatches"] += 1
    result["forbidden_json_field_hits"] = forbidden_key_scan(package_dir)
    result["winning_option_field_hits"] = sum(
        1 for item in result["forbidden_json_field_hits"] if "winning" in item["key"].lower()
    )
    fs = filesystem_scan(package_dir)
    security = text_security_scan(package_dir)
    result.update({
        "image_video_model_embedding_files": fs["image_video_model_embedding_files"],
        "symlinks": fs["symlinks"], "hardlinks": fs["hardlinks"], "special_files": fs["special_files"],
        "absolute_path_hits": security["absolute_path_hits"], "secret_hits": security["secret_hits"],
    })
    failures = (
        len(downstream) != 300 or len(provenance) != 300 or len(uid_lines) != 300
        or len(set(uid_lines)) != 300 or not result["uid_order_matches_frozen"]
        or not result["downstream_uid_order_matches"] or not result["provenance_uid_order_matches"]
        or any(result[key] for key in (
            "selected_count_out_of_range", "unsafe_relative_paths", "question_or_option_errors",
            "winning_option_field_hits", "missing_source_frames", "source_frame_hash_mismatches",
            "source_frame_size_mismatches", "missing_manifest_frames", "extra_manifest_frames",
        ))
        or result["forbidden_json_field_hits"] or result["image_video_model_embedding_files"]
        or result["symlinks"] or result["hardlinks"] or result["special_files"]
        or result["absolute_path_hits"] or result["secret_hits"]
    )
    result["status"] = "PASS" if not failures else "FAIL"
    return result


def write_contents(package_dir: Path) -> None:
    descriptions = {
        "selected_frames_downstream.jsonl": "300 gold-free downstream selector records",
        "selection_provenance.jsonl": "300 audit-only selection provenance records",
        "selected_frame_files_manifest.json": "deduplicated referenced-frame metadata; no images",
        "eval300_uid_order.txt": "frozen Eval300 UID order",
        "frozen_selector_metadata.json": "frozen method and artifact identities",
        "FINAL_SELECTOR_REPORT.json": "sanitized aggregate V2 final report",
        "README.md": "downstream usage and safety instructions",
        "MANIFEST.sha256": "SHA-256 manifest for every other package file",
        "CONTENTS.txt": "this file inventory",
        "package_validation.json": "machine-readable validation report",
        "PACKAGE_REPORT.md": "human-readable package validation summary",
    }
    text = "GenS V2 Eval300 manifest-only package contents\n\n"
    for name in FINAL_PACKAGE_FILES:
        text += f"{name}\t{descriptions[name]}\n"
    (package_dir / "CONTENTS.txt").write_text(text, encoding="utf-8")


def write_manifest(package_dir: Path) -> None:
    lines = []
    for path in sorted(package_dir.iterdir(), key=lambda item: item.name):
        if path.name == "MANIFEST.sha256":
            continue
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"manifest encountered non-regular file: {path.name}")
        lines.append(f"{sha256_file(path)}  {path.name}\n")
    (package_dir / "MANIFEST.sha256").write_text("".join(lines), encoding="utf-8")


def verify_manifest(package_dir: Path) -> dict[str, Any]:
    entries = {}
    for line in (package_dir / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    expected = set(FINAL_PACKAGE_FILES) - {"MANIFEST.sha256"}
    actual = {path.name for path in package_dir.iterdir() if path.is_file()} - {"MANIFEST.sha256"}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    hash_mismatch = sorted(name for name in expected & actual if entries.get(name) != sha256_file(package_dir / name))
    manifest_missing = sorted(expected - set(entries))
    manifest_extra = sorted(set(entries) - expected)
    return {
        "missing": missing, "extra": extra, "hash_mismatch": hash_mismatch,
        "manifest_missing": manifest_missing, "manifest_extra": manifest_extra,
        "status": "PASS" if not (missing or extra or hash_mismatch or manifest_missing or manifest_extra) else "FAIL",
    }


def create_archive(package_dir: Path, archive: Path) -> None:
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as handle:
        handle.add(package_dir, arcname=PACKAGE_NAME, recursive=True)


def archive_revalidation(archive: Path) -> dict[str, Any]:
    unsafe = []
    disallowed_types = []
    with tempfile.TemporaryDirectory(prefix="gens_eval300_selector_v2_verify_") as temporary:
        root = Path(temporary).resolve()
        with tarfile.open(archive, "r:gz") as handle:
            members = handle.getmembers()
            for member in members:
                pure = PurePosixPath(member.name)
                if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != PACKAGE_NAME:
                    unsafe.append(member.name)
                if not (member.isdir() or member.isreg()):
                    disallowed_types.append(member.name)
            if unsafe or disallowed_types:
                return {"status": "FAIL", "unsafe_members": unsafe, "disallowed_types": disallowed_types}
            for member in members:
                target = root.joinpath(*PurePosixPath(member.name).parts)
                if root not in target.resolve(strict=False).parents and target.resolve(strict=False) != root:
                    unsafe.append(member.name)
                    continue
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = handle.extractfile(member)
                    if source is None:
                        unsafe.append(member.name)
                        continue
                    with target.open("wb") as output:
                        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                            output.write(block)
        extracted = root / PACKAGE_NAME
        manifest_result = verify_manifest(extracted)
        files = {path.name for path in extracted.iterdir() if path.is_file()}
        expected_files = set(FINAL_PACKAGE_FILES)
        fs = filesystem_scan(extracted)
        security = text_security_scan(extracted)
        forbidden = forbidden_key_scan(extracted)
        result = {
            "unsafe_members": unsafe,
            "disallowed_types": disallowed_types,
            "missing": sorted(expected_files - files),
            "extra": sorted(files - expected_files),
            "hash_mismatch": manifest_result["hash_mismatch"],
            "manifest_missing": manifest_result["manifest_missing"],
            "manifest_extra": manifest_result["manifest_extra"],
            "image_video_model_embedding_files": fs["image_video_model_embedding_files"],
            "symlinks": fs["symlinks"], "hardlinks": fs["hardlinks"], "special_files": fs["special_files"],
            "absolute_path_hits": security["absolute_path_hits"], "secret_hits": security["secret_hits"],
            "forbidden_json_field_hits": forbidden,
        }
        result["status"] = "PASS" if not any(result[key] for key in result if key != "status") else "FAIL"
        return result


def package_report_text(validation: dict[str, Any], built: dict[str, Any], archive_check: dict[str, Any]) -> str:
    return f"""# Package validation report

Status: **{validation['status']}**

- Downstream records: {validation['question_count']}/300
- Provenance records: {validation['provenance_count']}/300
- Frozen-order unique UIDs: {validation['uid_unique_count']}/300
- Unique referenced frames: {validation['unique_frame_count']}
- Selected-frame distribution: {json.dumps(built['selected_count_distribution'], sort_keys=True)}
- Source frame missing: {validation['missing_source_frames']}
- Source frame hash mismatch: {validation['source_frame_hash_mismatches']}
- Unsafe relative paths: {validation['unsafe_relative_paths']}
- Forbidden result fields: {len(validation['forbidden_json_field_hits'])}
- Winning-option fields: {validation['winning_option_field_hits']}
- Image/video/model/embedding files: {len(validation['image_video_model_embedding_files'])}
- Absolute paths: {len(validation['absolute_path_hits'])}
- Symlinks/hardlinks/special files: {len(validation['symlinks'])}/{len(validation['hardlinks'])}/{len(validation['special_files'])}
- Secret scan hits: {len(validation['secret_hits'])}
- Provisional fresh-extraction validation: {archive_check['status']}

The package is manifest-only. No image content is included. Winning-option and per-option CLIP telemetry were removed from the portable provenance records.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    package_dir = args.package_dir.resolve()
    archive = args.archive.resolve()
    if package_dir.exists() or archive.exists():
        raise RuntimeError("refusing to overwrite an existing package directory or archive")
    package_dir.mkdir(parents=True)

    built = build_payload(package_dir)
    initial_validation = validate_scientific_payload(package_dir, built)
    if initial_validation["status"] != "PASS":
        raise RuntimeError(f"scientific payload validation failed: {initial_validation}")

    provisional_archive_check = {
        "status": "PASS",
        "unsafe_members": [], "disallowed_types": [], "missing": [], "extra": [],
        "hash_mismatch": [], "image_video_model_embedding_files": [],
        "restricted_payload_hits": [], "secret_hits": [],
    }
    validation_document = {
        "schema_version": 1,
        "status": "PASS",
        "scientific_payload": initial_validation,
        "archive_revalidation": provisional_archive_check,
        "assertions": {
            "stage_a_reruns": 0, "model_runs": 0, "gpu_runs": 0, "api_calls": 0,
            "restricted_evaluation_reads": 0, "raw_responses_in_package": 0,
            "generated_token_ids_in_package": 0, "image_files_in_package": 0,
        },
    }
    write_json(package_dir / "package_validation.json", validation_document)
    (package_dir / "PACKAGE_REPORT.md").write_text(
        package_report_text(initial_validation, built, provisional_archive_check), encoding="utf-8"
    )
    write_contents(package_dir)
    write_manifest(package_dir)

    final_fs = filesystem_scan(package_dir)
    if set(final_fs["files"]) != set(FINAL_PACKAGE_FILES):
        raise RuntimeError("final package file set mismatch")
    if final_fs["symlinks"] or final_fs["hardlinks"] or final_fs["special_files"]:
        raise RuntimeError("final package contains links or special files")
    manifest_result = verify_manifest(package_dir)
    if manifest_result["status"] != "PASS":
        raise RuntimeError(f"final MANIFEST verification failed: {manifest_result}")
    final_security = text_security_scan(package_dir)
    if final_security["absolute_path_hits"] or final_security["secret_hits"]:
        raise RuntimeError(f"final security scan failed: {final_security}")
    final_forbidden = forbidden_key_scan(package_dir)
    if final_forbidden:
        raise RuntimeError(f"final forbidden JSON field scan failed: {final_forbidden}")

    create_archive(package_dir, archive)
    archive_check = archive_revalidation(archive)
    if archive_check["status"] != "PASS":
        raise RuntimeError(f"fresh archive revalidation failed: {archive_check}")

    result = {
        "status": "PASS",
        "package_file_count": len(FINAL_PACKAGE_FILES),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "question_count": len(built["downstream_rows"]),
        "unique_frame_count": len(built["file_manifest"]),
        "directory_manifest_validation": manifest_result,
        "fresh_archive_revalidation": archive_check,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
