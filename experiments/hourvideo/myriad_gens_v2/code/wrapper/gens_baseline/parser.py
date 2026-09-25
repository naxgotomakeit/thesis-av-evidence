from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence


KEY_PATTERN = re.compile(r"^(?P<start>[1-9]\d*)(?:-(?P<end>[1-9]\d*))?$")


class ParserFailure(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class ParsedFrame:
    gens_input_index: int
    gens_relevance_score: int
    gens_span: str
    gens_response_order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParserFailure("duplicate_json_key", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_gens_response(raw_response: str, frame_count: int) -> list[ParsedFrame]:
    if not isinstance(raw_response, str):
        raise ParserFailure("non_string_response", "GenS response must be a string")
    if frame_count < 0 or frame_count > 256:
        raise ValueError("frame_count must be between 0 and 256")
    try:
        payload = json.loads(raw_response.strip(), object_pairs_hook=_unique_object)
    except ParserFailure:
        raise
    except json.JSONDecodeError as exc:
        raise ParserFailure("invalid_json", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ParserFailure("non_object_json", "GenS response must be one JSON object")

    parsed: list[ParsedFrame] = []
    claimed: set[int] = set()
    for response_order, (span, score) in enumerate(payload.items()):
        match = KEY_PATTERN.fullmatch(span)
        if not match:
            raise ParserFailure("invalid_key", f"invalid frame/span key: {span!r}")
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ParserFailure("invalid_score", f"score for {span!r} must be integer 1-5")
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if start > end:
            raise ParserFailure("reversed_span", f"span start exceeds end: {span}")
        if start < 1 or end > frame_count:
            raise ParserFailure(
                "out_of_range", f"span {span} outside available frame range 1-{frame_count}"
            )
        members = set(range(start, end + 1))
        overlap = claimed.intersection(members)
        if overlap:
            raise ParserFailure(
                "overlapping_span", f"span {span} repeats frame indices {sorted(overlap)}"
            )
        claimed.update(members)
        for frame_number in range(start, end + 1):
            parsed.append(
                ParsedFrame(
                    gens_input_index=frame_number,
                    gens_relevance_score=score,
                    gens_span=span,
                    gens_response_order=response_order,
                )
            )
    return parsed


def map_parsed_frames(
    parsed: Sequence[ParsedFrame], chronological_candidates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for item in parsed:
        index = item.gens_input_index - 1
        if index < 0 or index >= len(chronological_candidates):
            raise ParserFailure("out_of_range", "parsed frame index cannot be mapped")
        candidate = dict(chronological_candidates[index])
        candidate.update(item.to_dict())
        mapped.append(candidate)
    return mapped
