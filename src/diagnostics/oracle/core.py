from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


SUPPORTED_DURATION_CLASSES = ("1s", "10s", "60s")
ORACLE_PROMPT_VERSION = "egopolice-oracle-short-clip-mcq-v1"


class OracleInputError(RuntimeError):
    """Raised when an Oracle diagnostic input violates its contract."""


@dataclass(frozen=True)
class OracleCase:
    question_id: str
    duration_class: str
    source_video_relative_path: str
    source_video_path: Path
    start_sec: float
    end_sec: float
    question: str
    options: tuple[str, ...]
    ground_truth_index: int

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def ground_truth_text(self) -> str:
        return self.options[self.ground_truth_index]


def _metadata_path(data_root: Path, duration_class: str) -> Path:
    if duration_class not in SUPPORTED_DURATION_CLASSES:
        raise OracleInputError(f"Unsupported duration class: {duration_class}")
    return data_root / f"mcq_{duration_class}.json"


def _load_rows(data_root: Path, duration_class: str) -> list[dict[str, Any]]:
    path = _metadata_path(data_root, duration_class)
    if not path.is_file():
        raise OracleInputError(f"Missing MCQ metadata: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise OracleInputError(f"MCQ metadata must be a JSON list: {path}")
    return rows


def _source_path(data_root: Path, value: Any) -> tuple[str, Path]:
    relative = str(value or "").replace("\\", "/")
    parts = PurePosixPath(relative).parts
    if not relative or PurePosixPath(relative).is_absolute() or ".." in parts:
        raise OracleInputError(f"Invalid source video path: {relative!r}")
    return relative, data_root.joinpath("videos", *parts)


def load_oracle_case(data_root: Path, duration_class: str, question_id: str) -> OracleCase:
    matches = [row for row in _load_rows(data_root, duration_class) if str(row.get("id")) == question_id]
    if len(matches) != 1:
        raise OracleInputError(
            f"Expected one {duration_class} row for {question_id}, found {len(matches)}"
        )
    row = matches[0]
    relative, source_path = _source_path(data_root, row.get("video"))
    options = tuple(str(value) for value in row.get("options") or ())
    answer = row.get("answer")
    start_sec = float(row.get("start second"))
    end_sec = float(row.get("end second"))
    if not source_path.is_file():
        raise OracleInputError(f"Source video is unavailable: {source_path}")
    if not str(row.get("question") or "").strip() or len(options) != 5:
        raise OracleInputError(f"Invalid question/options for {question_id}")
    if not isinstance(answer, int) or not 0 <= answer < 5:
        raise OracleInputError(f"Invalid answer index for {question_id}")
    if start_sec < 0 or end_sec <= start_sec:
        raise OracleInputError(f"Invalid Oracle interval for {question_id}: [{start_sec}, {end_sec})")
    expected_duration = float(duration_class.removesuffix("s"))
    if abs((end_sec - start_sec) - expected_duration) > 1e-6:
        raise OracleInputError(
            f"Interval duration does not match {duration_class} for {question_id}"
        )
    return OracleCase(
        question_id=question_id,
        duration_class=duration_class,
        source_video_relative_path=relative,
        source_video_path=source_path,
        start_sec=start_sec,
        end_sec=end_sec,
        question=str(row["question"]),
        options=options,
        ground_truth_index=answer,
    )


def select_materializable_question_ids(
    data_root: Path, duration_plan: Sequence[str],
) -> list[tuple[str, str]]:
    """Select the first materializable row per requested class in metadata order.

    Selection uses only question identity, interval validity, and source-file
    availability. It never reads options, answers, or model outputs.
    """
    selected: list[tuple[str, str]] = []
    used_ids: set[str] = set()
    next_offsets = {duration_class: 0 for duration_class in SUPPORTED_DURATION_CLASSES}
    row_cache = {
        duration_class: _load_rows(data_root, duration_class)
        for duration_class in set(duration_plan)
    }
    for duration_class in duration_plan:
        rows = row_cache[duration_class]
        found: tuple[str, str] | None = None
        for index in range(next_offsets[duration_class], len(rows)):
            row = rows[index]
            next_offsets[duration_class] = index + 1
            question_id = str(row.get("id") or "")
            if not question_id or question_id in used_ids:
                continue
            try:
                _, source_path = _source_path(data_root, row.get("video"))
                start_sec = float(row.get("start second"))
                end_sec = float(row.get("end second"))
            except (OracleInputError, TypeError, ValueError):
                continue
            expected_duration = float(duration_class.removesuffix("s"))
            if source_path.is_file() and start_sec >= 0 and abs(end_sec - start_sec - expected_duration) <= 1e-6:
                found = (duration_class, question_id)
                break
        if found is None:
            raise OracleInputError(f"No materializable {duration_class} question remains")
        selected.append(found)
        used_ids.add(found[1])
    return selected


def build_oracle_prompt(case: OracleCase) -> str:
    option_lines = "\n".join(f"{index}. {text}" for index, text in enumerate(case.options))
    return (
        f"You are given the ground-truth {case.duration_sec:g}-second video clip for one "
        "multiple-choice question. Answer using only this clip.\n\n"
        f"Question: {case.question}\n\nOptions:\n{option_lines}\n\n"
        "Return exactly one option index: 0, 1, 2, 3, or 4. "
        "Do not provide an explanation."
    )


def build_oracle_messages(case: OracleCase, *, max_pixels: int = 262144) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(case.source_video_path),
                    "video_start": case.start_sec,
                    "video_end": case.end_sec,
                    "fps": 1.0,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": build_oracle_prompt(case)},
            ],
        }
    ]


def normalize_single_video_fps(
    video_inputs: Sequence[Any], video_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Adapt qwen-vl-utils 0.0.14 output to Transformers 5.14.1."""
    normalized = dict(video_kwargs)
    fps = normalized.get("fps")
    if isinstance(fps, (list, tuple)):
        if len(video_inputs) != 1 or len(fps) != 1:
            raise OracleInputError("Oracle diagnostics require exactly one video and one FPS value")
        normalized["fps"] = fps[0]
    return normalized


def parse_oracle_prediction(raw_text: str) -> int:
    text = raw_text.strip()
    numeric = re.fullmatch(r"(?:option\s*)?([0-4])\.?", text, flags=re.IGNORECASE)
    if numeric:
        return int(numeric.group(1))
    letter = re.fullmatch(r"(?:option\s*)?([A-E])\.?", text, flags=re.IGNORECASE)
    if letter:
        return ord(letter.group(1).upper()) - ord("A")
    raise OracleInputError(f"Unparseable Oracle MCQ prediction: {raw_text!r}")
