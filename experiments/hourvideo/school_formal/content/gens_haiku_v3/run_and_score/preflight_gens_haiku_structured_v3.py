#!/usr/bin/env python3
"""Full 300-question no-API/no-gold preflight for structured-answer v3."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from direct_api_v1.anthropic_provider import direct_action_tools  # noqa: E402
from gens_haiku_eval300.runtime import atomic_json, canonical_sha, sha256_file  # noqa: E402
from gens_haiku_eval300.structured_v3 import (  # noqa: E402
    StructuredV3Config,
    build_structured_request,
    runtime_fingerprint,
    tool_schema,
    user_question_text,
)

DEFAULT_RUNNER = ROOT / "scripts/run_gens_haiku_structured_v3.py"


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/gens_haiku_structured_v3.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gens_haiku_structured_v3/preflight_v3")
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    args = parser.parse_args()
    config = StructuredV3Config.load(args.config)
    rows = load_rows(Path(config.selector_path))
    ids = [row["qa_uid"] for row in rows]
    canonical = json.loads(Path(config.canonical_population_manifest).read_text(encoding="utf-8"))["eval300_question_ids"]
    uid_order = Path(config.uid_order_path).read_text(encoding="utf-8").splitlines()
    failures: list[dict] = []
    image_refs = 0
    unique_paths: set[str] = set()
    upload_bytes = 0
    allowed_payload = {"model", "max_tokens", "temperature", "system", "messages", "tools", "tool_choice"}
    direct_final = direct_action_tools(0)
    if direct_final != [tool_schema()]:
        failures.append({"error": "Direct final_answer schema differs from frozen v3 schema"})
    for index, row in enumerate(rows):
        try:
            payload, frames = build_structured_request(row, config, encode_images=True)
            if set(payload) != allowed_payload:
                raise AssertionError(f"request fields differ: {sorted(payload)}")
            if payload["tools"] != [tool_schema()] or payload["tool_choice"] != {"type": "any"}:
                raise AssertionError("tool schema/choice mismatch")
            if payload["system"] != config.system_prompt:
                raise AssertionError("system prompt mismatch")
            if len(payload["messages"]) != 1 or payload["messages"][0]["role"] != "user":
                raise AssertionError("message layout mismatch")
            blocks = payload["messages"][0]["content"]
            if len(blocks) != len(frames) + 1 or blocks[-1] != {"type": "text", "text": user_question_text(row)}:
                raise AssertionError("image/text block layout mismatch")
            if any(block["type"] != "image" or block["source"]["media_type"] != "image/jpeg" for block in blocks[:-1]):
                raise AssertionError("image transport mismatch")
            if [frame["timestamp_sec"] for frame in frames] != sorted(frame["timestamp_sec"] for frame in frames):
                raise AssertionError("non-chronological frames")
            for frame in frames:
                path = Path(frame["path"])
                with Image.open(path) as image:
                    image.verify()
                if frame["expected_sha256"] != frame["observed_sha256"]:
                    raise AssertionError("frame SHA mismatch")
                unique_paths.add(str(path))
                upload_bytes += path.stat().st_size
            image_refs += len(frames)
        except Exception as error:
            failures.append({
                "index": index,
                "question_id": row.get("qa_uid"),
                "error": f"{type(error).__name__}: {error}",
            })

    prompt_spec = {
        "system_prompt": config.system_prompt,
        "user_message": "chronological original JPEG image blocks, then Question/Options A-E/Return the final_answer action",
        "tool": tool_schema(),
        "tool_choice": {"type": "any"},
        "parser": "exactly one final_answer tool_use and no non-tool blocks; Direct-effective str conversion; nonblank reason with no local upper bound; extra tool-input fields ignored; A-E only; no text extraction",
        "design_note": "Direct-v1.2 final-answer rule adapted to frozen GenS frames; map/navigation/inspection removed.",
    }
    fingerprints = {
        "config_sha256": sha256_file(args.config),
        "runtime_sha256": sha256_file(ROOT / "src/gens_haiku_eval300/structured_v3.py"),
        "shared_runtime_sha256": sha256_file(ROOT / "src/gens_haiku_eval300/runtime.py"),
        "provider_base_sha256": sha256_file(ROOT / "src/gens_haiku_eval300/structured_v2.py"),
        "system_prompt_sha256": sha256_file(Path(config.system_prompt_path)),
        "runner_sha256": sha256_file(args.runner),
        "preflight_sha256": sha256_file(Path(__file__)),
        "selector_sha256": sha256_file(Path(config.selector_path)),
        "uid_order_sha256": sha256_file(Path(config.uid_order_path)),
        "canonical_population_manifest_sha256": sha256_file(Path(config.canonical_population_manifest)),
        "tool_schema_sha256": canonical_sha(tool_schema()),
        "prompt_spec_sha256": canonical_sha(prompt_spec),
    }
    historical_anchors = {}
    for label, path in {
        "gens_v1_score": ROOT / "outputs/gens_haiku_eval300/formal_v1_evaluation/posthoc_score.json",
        "gens_v1_posthoc": ROOT / "outputs/gens_haiku_eval300/formal_v1_posthoc_answer_extraction_v1/posthoc_sensitivity_score.json",
        "gens_v2_score": ROOT / "outputs/gens_haiku_structured_v2/formal_v2_evaluation/score.json",
        "direct_accuracy": ROOT / "outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/canonical_summary_v1/accuracy.json",
    }.items():
        historical_anchors[label] = {"path": str(path), "sha256": sha256_file(path)}
    manifest = {
        "schema_version": "gens_haiku_structured_answer_v3_manifest",
        "experiment_name": "GenS + Haiku structured-answer v3",
        "experiment_id": config.experiment_id,
        "question_count": len(rows),
        "question_ids": ids,
        "routes": [
            {
                "route_index": index,
                "question_id": row["qa_uid"],
                "video_id": row["video_id"],
                "input_sha256": canonical_sha(row),
                "selected_frame_count": row["selected_frame_count"],
            }
            for index, row in enumerate(rows)
        ],
        "config": json.loads(args.config.read_text(encoding="utf-8")),
        "fingerprints": fingerprints,
        "prompt_spec": prompt_spec,
        "historical_read_only_anchors": historical_anchors,
        "v2_to_v3_differences": {
            "answer_policy": "null abstention removed; best-evidence A-E selection required",
            "system_prompt": "fixed user-approved Direct-derived v3 system prompt",
            "tool_schema": "Direct-v1.2 final_answer copied exactly",
            "tool_choice": {"type": "any"},
            "parser": "Direct effective final-answer parser: str conversion, nonblank reason, no local 240-character enforcement, extra input fields ignored; no natural-language extraction",
            "parser_alignment_only": True,
        },
        "gold_loaded": False,
        "api_calls_made": 0,
    }
    counts = [row["selected_frame_count"] for row in rows]
    status = not failures and len(rows) == len(set(ids)) == 300 and ids == canonical == uid_order
    report = {
        "status": "PASS" if status else "FAIL",
        "gold_loaded": False,
        "api_calls": 0,
        "questions": len(rows),
        "unique_question_ids": len(set(ids)),
        "canonical_order_exact": ids == canonical,
        "selector_order_exact": ids == uid_order,
        "payloads_checked": len(rows) - sum(1 for row in failures if "index" in row),
        "failures": failures,
        "image_references": image_refs,
        "unique_physical_images": len(unique_paths),
        "upload_bytes": upload_bytes,
        "frame_count": {
            "min": min(counts),
            "mean": statistics.mean(counts),
            "median": statistics.median(counts),
            "max": max(counts),
            "distribution": dict(sorted(Counter(counts).items())),
        },
        "request_field_whitelist": sorted(allowed_payload),
        "map_sent": False,
        "gold_sent": False,
        "old_answer_sent": False,
        "selection_provenance_sent": False,
        "direct_final_schema_exact": direct_final == [tool_schema()],
        "legacy_outputs_read_only": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "formal_manifest.json", manifest)
    atomic_json(args.output / "preflight_report.json", report)
    atomic_json(args.output / "smoke_manifest.json", {
        "question_ids": [
            "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31",
            "6fd90f8d-7a4d-425d-a812-3268db0b0342_8_9",
        ],
        "frame_counts": [1, 16],
        "selection_rule": "user-frozen prior frame-count smoke identities; no gold",
        "gold_loaded": False,
    })
    run_fp = runtime_fingerprint(args.config, manifest, args.runner, Path(__file__))
    atomic_json(args.output / "freeze_identity.json", {
        "manifest_sha256": sha256_file(args.output / "formal_manifest.json"),
        "run_fingerprint": run_fp,
        "fingerprints": fingerprints,
        "scientific_configuration_frozen": True,
        "gold_loaded": False,
    })
    print(json.dumps({**report, "manifest": str(args.output / "formal_manifest.json"), "run_fingerprint": run_fp}, indent=2))
    if not status:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
