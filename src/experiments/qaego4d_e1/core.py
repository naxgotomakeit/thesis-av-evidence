from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


CONDITIONS = ("blind", "uniform_8", "uniform_32", "oracle_leq8")
TASKS = ("open", "closed")
CONTEXT_SCOPE = "canonical_clip"


class E1ProtocolError(RuntimeError):
    """Raised when a frozen E1 contract would be violated."""


@dataclass(frozen=True)
class CanonicalCase:
    task: str
    question_id: str
    clip_uid: str
    video_uid: str
    clip_start_sec: float
    clip_end_sec: float
    parent_video_path: Path
    question: str
    answer: str
    wrong_answers: tuple[str, ...]
    evidence_start_sec: float
    evidence_end_sec: float

    @property
    def clip_duration_sec(self) -> float:
        return self.clip_end_sec - self.clip_start_sec


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _annotation_path(annotation_root: Path, task: str) -> Path:
    return annotation_root / (
        "annotations.QaEgo4D_test.json" if task == "open" else "annotations.QaEgo4D_test_close.json"
    )


def _manifest_rows(path: Path) -> list[str]:
    payload = load_json(path)
    ids = payload.get("question_ids") if isinstance(payload, dict) else None
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise E1ProtocolError(f"Invalid eval manifest: {path}")
    return ids


def _mapping_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    if not isinstance(rows, list):
        raise E1ProtocolError(f"Invalid canonical mapping: {path}")
    result = {str(row.get("sample_id")): row for row in rows if isinstance(row, dict)}
    if len(result) != len(rows):
        raise E1ProtocolError(f"Duplicate/missing sample IDs in mapping: {path}")
    return result


def load_cases(
    *, task: str, annotation_root: Path, manifest_path: Path, mapping_path: Path,
) -> tuple[list[CanonicalCase], str]:
    if task not in TASKS:
        raise E1ProtocolError(f"Unsupported task: {task}")
    manifest_ids = _manifest_rows(manifest_path)
    annotations = load_json(_annotation_path(annotation_root, task))
    if not isinstance(annotations, list):
        raise E1ProtocolError("QaEgo4D annotations must be a list")
    rows = {str(row.get("sample_id")): row for row in annotations if isinstance(row, dict)}
    mapping = _mapping_rows(mapping_path)
    if len(rows) != len(annotations):
        raise E1ProtocolError(f"Duplicate/missing sample IDs in {task} annotations")
    missing_annotations = [sample_id for sample_id in manifest_ids if sample_id not in rows]
    missing_mapping = [sample_id for sample_id in manifest_ids if sample_id not in mapping]
    if missing_annotations or missing_mapping:
        raise E1ProtocolError(
            f"Manifest mapping failure task={task} annotations={len(missing_annotations)} mapping={len(missing_mapping)}"
        )
    cases: list[CanonicalCase] = []
    for sample_id in manifest_ids:
        row, canonical = rows[sample_id], mapping[sample_id]
        clip_start, clip_end = float(canonical["clip_start_sec"]), float(canonical["clip_end_sec"])
        evidence_start = float(row["video_start_sec"])
        evidence_end = float(row["video_end_sec"])
        if (
            str(row.get("video_uid")) != str(canonical.get("video_uid"))
            or str(row.get("video_id")) != str(canonical.get("clip_uid"))
            or str(canonical.get("context_scope")) != CONTEXT_SCOPE
            or clip_end <= clip_start
        ):
            raise E1ProtocolError(f"Canonical mapping contract mismatch: {sample_id}")
        path = Path(str(canonical["parent_video_path"]))
        if not path.is_file() or path.stat().st_size <= 0:
            raise E1ProtocolError(f"Missing parent MP4 for {sample_id}: {path}")
        wrong_answers = tuple(str(value) for value in row.get("wrong_answers", []))
        if task == "closed" and len(wrong_answers) != 3:
            raise E1ProtocolError(f"Closed task needs exactly three distractors: {sample_id}")
        cases.append(CanonicalCase(
            task=task,
            question_id=sample_id,
            clip_uid=str(canonical["clip_uid"]),
            video_uid=str(canonical["video_uid"]),
            clip_start_sec=clip_start,
            clip_end_sec=clip_end,
            parent_video_path=path,
            question=str(row["question"]),
            answer=str(row["answer"]),
            wrong_answers=wrong_answers,
            evidence_start_sec=evidence_start,
            evidence_end_sec=evidence_end,
        ))
    return cases, sha256_file(manifest_path)


def deterministic_closed_options(case: CanonicalCase) -> tuple[list[str], int]:
    if case.task != "closed":
        raise E1ProtocolError("Closed options requested for non-closed case")
    options = [case.answer, *case.wrong_answers]
    digest = hashlib.sha256(case.question_id.encode("utf-8")).digest()
    cursor = 0
    for right in range(len(options) - 1, 0, -1):
        left = digest[cursor % len(digest)] % (right + 1)
        cursor += 1
        options[left], options[right] = options[right], options[left]
    return options, options.index(case.answer)


def build_prompt(case: CanonicalCase, *, open_template: str, closed_template: str) -> tuple[str, list[str] | None, int | None]:
    if case.task == "open":
        return open_template.format(question=case.question), None, None
    options, answer_index = deterministic_closed_options(case)
    option_lines = "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))
    return closed_template.format(question=case.question, options=option_lines), options, answer_index


def uniform_requested_timestamps(start_sec: float, end_sec: float, *, budget: int, nominal_fps: float) -> list[float]:
    if budget <= 0 or nominal_fps <= 0 or end_sec <= start_sec:
        raise E1ProtocolError("Invalid uniform sampling inputs")
    count = min(budget, max(1, math.floor((end_sec - start_sec) * nominal_fps)))
    width = (end_sec - start_sec) / count
    return [start_sec + (index + 0.5) * width for index in range(count)]


def oracle_requested_timestamps(case: CanonicalCase, *, budget: int, nominal_fps: float) -> tuple[list[float], str, tuple[float, float]]:
    if case.evidence_end_sec < case.evidence_start_sec:
        raise E1ProtocolError(f"Negative evidence interval: {case.question_id}")
    if math.isclose(case.evidence_start_sec, case.evidence_end_sec, abs_tol=1e-9):
        point = min(max(case.evidence_start_sec, case.clip_start_sec), math.nextafter(case.clip_end_sec, case.clip_start_sec))
        # A point has no non-empty temporal interval.  The approved provisional
        # rule is nearest *within the canonical clip*, never a point-sized seek.
        return [point], "point_nearest_decodable_frame_provisional", (case.clip_start_sec, case.clip_end_sec)
    start = max(case.evidence_start_sec, case.clip_start_sec)
    end = min(case.evidence_end_sec, case.clip_end_sec)
    if end <= start:
        raise E1ProtocolError(f"Evidence interval does not intersect canonical clip: {case.question_id}")
    return uniform_requested_timestamps(start, end, budget=budget, nominal_fps=nominal_fps), "interval_uniform_within_gt", (start, end)


def _frame_index(frame: Any, stream: Any) -> int:
    rate = float(stream.average_rate)
    time_base = float(stream.time_base)
    ticks_per_frame = 1.0 / (rate * time_base)
    return int(round((int(frame.pts) - int(stream.start_time or 0)) / ticks_per_frame))


def _decode_nearest_frames_with_seeks(
    *, video_path: Path, requested_timestamps: Iterable[float], allowed_start_sec: float, allowed_end_sec: float,
) -> tuple[list[Image.Image], list[dict[str, Any]], int]:
    """Resolve requested timestamps with indexed backward seeks, never outside the permitted range."""
    import av

    targets = [float(value) for value in requested_timestamps]
    if not targets:
        return [], [], 0
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        images: list[Image.Image] = []
        records: list[dict[str, Any]] = []
        seen_pts: set[int] = set()
        decoded_frame_count = 0
        for target in targets:
            container.seek(int(target / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
            previous: Any | None = None
            current: Any | None = None
            for frame in container.decode(stream):
                decoded_frame_count += 1
                if frame.time is None:
                    continue
                timestamp = float(frame.time)
                if timestamp < allowed_start_sec:
                    continue
                if timestamp >= allowed_end_sec:
                    break
                if timestamp < target:
                    previous = frame
                    continue
                current = frame
                break
            candidate = current or previous
            if candidate is None:
                raise E1ProtocolError(
                    f"No decodable frame in allowed range [{allowed_start_sec}, {allowed_end_sec}) for {video_path}"
                )
            if current is not None and previous is not None:
                if abs(float(previous.time) - target) <= abs(float(current.time) - target):
                    candidate = previous
            source_pts = int(candidate.pts)
            if source_pts in seen_pts:
                continue
            seen_pts.add(source_pts)
            images.append(candidate.to_image().convert("RGB"))
            records.append({
                "requested_timestamp_sec": target,
                "actual_timestamp_sec": float(candidate.time),
                "source_frame_index": _frame_index(candidate, stream),
                "source_pts": source_pts,
                "nominal_fps": float(stream.average_rate),
            })
        if not images:
            raise E1ProtocolError(f"No unique decodable frames for {video_path}")
        return images, records, decoded_frame_count
    finally:
        container.close()


def _interval_distance(timestamp_sec: float, start_sec: float, end_sec: float) -> float:
    if start_sec <= timestamp_sec < end_sec:
        return 0.0
    return min(abs(timestamp_sec - start_sec), abs(timestamp_sec - end_sec))


def _select_nearest_interval_candidate(candidates: Iterable[Any], *, start_sec: float, end_sec: float) -> Any:
    """Choose the closest candidate; equal distances resolve to earlier time."""
    values = list(candidates)
    if not values:
        raise E1ProtocolError("No candidate frame for interval fallback")
    return min(
        values,
        key=lambda frame: (_interval_distance(float(frame.time), start_sec, end_sec), float(frame.time)),
    )


def decode_zero_decodable_interval_nearest_frame(
    *, case: CanonicalCase,
) -> tuple[list[Image.Image], list[dict[str, Any]], float, int]:
    """Amendment #2 fallback for a non-empty interval containing no frame.

    This function intentionally returns one frame only.  It never expands the
    GT interval: the canonical clip is the safety boundary for the nearest
    decodable-frame resolution required by Amendment #2.
    """
    start_sec, end_sec = case.evidence_start_sec, case.evidence_end_sec
    if not start_sec < end_sec:
        raise E1ProtocolError("Zero-decodable interval fallback requires start < end")
    started = time.perf_counter()
    import av

    container = av.open(str(case.parent_video_path))
    try:
        stream = container.streams.video[0]
        container.seek(
            int(start_sec / float(stream.time_base)),
            stream=stream,
            any_frame=False,
            backward=True,
        )
        previous: Any | None = None
        following: Any | None = None
        decoded_frame_count = 0
        for frame in container.decode(stream):
            decoded_frame_count += 1
            if frame.time is None:
                continue
            timestamp = float(frame.time)
            if timestamp < case.clip_start_sec:
                continue
            if timestamp >= case.clip_end_sec:
                break
            if timestamp < start_sec:
                previous = frame
                continue
            if timestamp < end_sec:
                raise E1ProtocolError("Zero-decodable fallback invoked despite an in-window frame")
            following = frame
            break
        candidates = [frame for frame in (previous, following) if frame is not None]
        candidate = _select_nearest_interval_candidate(candidates, start_sec=start_sec, end_sec=end_sec)
        timestamp = float(candidate.time)
        if not case.clip_start_sec <= timestamp < case.clip_end_sec:
            raise E1ProtocolError("Zero-decodable fallback escaped canonical clip")
        record = {
            "requested_timestamp_sec": (start_sec + end_sec) / 2.0,
            "actual_timestamp_sec": timestamp,
            "source_frame_index": _frame_index(candidate, stream),
            "source_pts": int(candidate.pts),
            "nominal_fps": float(stream.average_rate),
        }
        return [candidate.to_image().convert("RGB")], [record], time.perf_counter() - started, decoded_frame_count
    finally:
        container.close()


def decode_requested_frames(
    *, case: CanonicalCase, requested_timestamps: Iterable[float], allowed_start_sec: float, allowed_end_sec: float,
) -> tuple[list[Image.Image], list[dict[str, Any]], float, int]:
    if allowed_start_sec < case.clip_start_sec or allowed_end_sec > case.clip_end_sec or allowed_end_sec <= allowed_start_sec:
        raise E1ProtocolError("Frame decode range must be wholly inside canonical clip")
    started = time.perf_counter()
    try:
        images, records, decoded_frame_count = _decode_nearest_frames_with_seeks(
            video_path=case.parent_video_path, requested_timestamps=requested_timestamps,
            allowed_start_sec=allowed_start_sec, allowed_end_sec=allowed_end_sec,
        )
        for image, metadata in zip(images, records):
            actual = float(metadata["actual_timestamp_sec"])
            if not allowed_start_sec <= actual < allowed_end_sec:
                image.close()
                raise E1ProtocolError("Decoder returned a frame outside the allowed canonical range")
        if not images:
            raise E1ProtocolError("Sampling resolved to zero unique frames")
        return images, records, time.perf_counter() - started, decoded_frame_count
    except Exception:
        for image in locals().get("images", []):
            image.close()
        raise


def valid_result(record: Any, *, config_hash: str, manifest_hash: str) -> bool:
    required = {
        "clip_uid", "video_uid", "context_scope", "video_decode_time_s", "frames_decoded_online",
        "frames_shown", "answer_model_time_s", "total_latency_s", "model_calls", "config_hash", "manifest_hash",
        "success", "cuda_memory",
    }
    return (
        isinstance(record, dict)
        and required <= set(record)
        and record.get("config_hash") == config_hash
        and record.get("manifest_hash") == manifest_hash
        and record.get("context_scope") == CONTEXT_SCOPE
    )
