#!/usr/bin/env python3
"""Select a small, diverse manual-annotation set from the existing candidate pool."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "dataset_inspection" / "dev_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "dataset_inspection" / "dev_cases_10.json"


def suspected_evidence_type(case: dict[str, Any]) -> str:
    qtype = str(case.get("question_type") or "").lower()
    question = str(case.get("question") or "").lower()
    context = str(case.get("provided_context") or case.get("annotation_context") or "").lower()
    text = f"{question} {context}"
    visual_terms = ("visually", "visible", "camera shows", "seen", "look", "display", "what card", "what object")
    speech_terms = ("say", "said", "says", "spoken", "voice", "asks", "reply", "conversation", "utter", "phrase")
    sound_terms = ("sound", "heard", "clink", "thud", "rustl", "scrap", "tap", "music", "gurg", "swish", "footstep")
    visual = any(term in text for term in visual_terms)
    speech = any(term in text for term in speech_terms)
    sound = any(term in text for term in sound_terms)
    if "cross-modal" in qtype or (visual and (speech or sound)):
        return "audio_visual"
    if speech:
        return "speech"
    if sound or any(term in qtype for term in ("sound", "spatial location", "temporal", "counting")):
        return "non_speech_sound"
    if visual:
        return "visual"
    return "unknown"


def select_cases(pool: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    valid = [
        case for case in pool
        if case.get("media_valid") is True
        and case.get("video_path")
        and case.get("audio_path")
        and case.get("video_duration")
        and case.get("audio_duration")
    ]
    if len({case.get("video_id") for case in valid}) < count:
        raise ValueError(f"Need {count} distinct valid videos, but only found {len({case.get('video_id') for case in valid})}")

    enriched = []
    for case in valid:
        item = dict(case)
        item["suspected_evidence_type"] = suspected_evidence_type(item)
        activity = item.get("signal_active_fraction")
        item["_activity"] = float(activity) if isinstance(activity, (int, float)) else -1.0
        enriched.append(item)

    # Two examples from each heuristic evidence family when available. Question-type
    # diversity is the next priority, followed by signal activity and stable case ID.
    targets = ["audio_visual", "speech", "non_speech_sound", "visual", "unknown"] * 2
    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    used_question_types: Counter[str] = Counter()
    used_evidence_types: Counter[str] = Counter()

    for target in targets:
        candidates = [c for c in enriched if c["suspected_evidence_type"] == target and c.get("video_id") not in used_videos]
        if not candidates:
            continue
        candidates.sort(key=lambda c: (used_question_types[str(c.get("question_type"))], -c["_activity"], str(c.get("case_id"))))
        chosen = candidates[0]
        selected.append(chosen)
        used_videos.add(str(chosen["video_id"]))
        used_question_types[str(chosen.get("question_type"))] += 1
        used_evidence_types[target] += 1

    while len(selected) < count:
        remaining = [c for c in enriched if c.get("video_id") not in used_videos]
        if not remaining:
            raise ValueError(f"Could only select {len(selected)} cases")
        remaining.sort(key=lambda c: (
            used_evidence_types[c["suspected_evidence_type"]],
            used_question_types[str(c.get("question_type"))],
            -c["_activity"],
            str(c.get("case_id")),
        ))
        chosen = remaining[0]
        selected.append(chosen)
        used_videos.add(str(chosen["video_id"]))
        used_question_types[str(chosen.get("question_type"))] += 1
        used_evidence_types[chosen["suspected_evidence_type"]] += 1

    output = []
    for chosen in selected[:count]:
        item = {k: v for k, v in chosen.items() if not k.startswith("_")}
        item["evidence_start"] = None
        item["evidence_end"] = None
        item["needs_manual_evidence_annotation"] = True
        evidence = item["suspected_evidence_type"]
        item["selection_reason"] = (
            f"Selected from the validated candidate pool with distinct video_id; "
            f"adds question-type diversity ({item.get('question_type')}) and is a heuristic "
            f"{evidence} evidence candidate; external audio signal-active fraction="
            f"{item.get('signal_active_fraction')}. Evidence type and timestamps require manual verification."
        )
        output.append(item)
    return output


def compact(value: Any, width: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text if len(text) <= width else text[: width - 1] + "…"


def print_table(cases: list[dict[str, Any]]) -> None:
    columns = [
        ("case_id", 10), ("video_id", 8), ("question", 48), ("answer", 36),
        ("suspected_evidence_type", 22), ("selection_reason", 68),
    ]
    header = " | ".join(name.ljust(width) for name, width in columns)
    print(header)
    print("-+-".join("-" * width for _, width in columns))
    for case in cases:
        print(" | ".join(compact(case.get(name), width).ljust(width) for name, width in columns))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pool = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(pool, list):
        raise SystemExit("Input candidate pool must be a JSON array")
    selected = select_cases(pool, 10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(selected)
    print(f"\nWrote exactly {len(selected)} cases to {args.output}")


if __name__ == "__main__":
    main()
