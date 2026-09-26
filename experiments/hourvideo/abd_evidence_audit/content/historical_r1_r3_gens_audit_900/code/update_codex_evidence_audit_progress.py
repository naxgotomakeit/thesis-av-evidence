#!/usr/bin/env python3
"""Validate completed Codex evidence-audit batches and atomically save progress.

This script only reads frozen review packages/batch metadata and writes derived
audit progress.  It never reads method identities, gold, or prior scores.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SUPPORT = {
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "unreviewable",
    "no_final_answer",
}
REASON = {
    "corresponds_to_input",
    "reasonable_unverified_inference",
    "common_sense_option_wording_or_guess",
    "unsupported_factual_assertion",
    "contradicts_input",
    "input_evidence_cannot_verify_claim",
}
SOURCE = {"map", "images", "map_and_images", "none", "not_applicable"}
CONFIDENCE = {"high", "medium", "low"}
REQUIRED = {
    "neutral_id",
    "option_support",
    "reason_basis",
    "direct_evidence_source",
    "audit_note",
    "evidence_refs",
    "missing_key_evidence",
    "confidence",
    "needs_further_review",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_root", type=Path)
    args = parser.parse_args()
    root = args.audit_root.resolve()
    batches = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))["batches"]

    completed: list[str] = []
    reviewed_ids: list[str] = []
    issues: list[str] = []
    for batch in batches:
        batch_id = batch["batch_id"]
        path = root / "reviews" / f"{batch_id}.jsonl"
        if not path.exists():
            continue
        rows = read_jsonl(path)
        expected = batch["neutral_ids"]
        actual = [row.get("neutral_id") for row in rows]
        if actual != expected:
            issues.append(f"{batch_id}: IDs/order mismatch: expected={expected}, actual={actual}")
            continue
        for row in rows:
            missing = REQUIRED - set(row)
            if missing:
                issues.append(f"{batch_id}/{row.get('neutral_id')}: missing {sorted(missing)}")
            if row.get("option_support") not in SUPPORT:
                issues.append(f"{batch_id}/{row.get('neutral_id')}: invalid option_support")
            if not isinstance(row.get("reason_basis"), list) or not set(row.get("reason_basis", [])) <= REASON:
                issues.append(f"{batch_id}/{row.get('neutral_id')}: invalid reason_basis")
            if row.get("direct_evidence_source") not in SOURCE:
                issues.append(f"{batch_id}/{row.get('neutral_id')}: invalid evidence source")
            if row.get("confidence") not in CONFIDENCE:
                issues.append(f"{batch_id}/{row.get('neutral_id')}: invalid confidence")
        if not any(issue.startswith(batch_id + "/") or issue.startswith(batch_id + ":") for issue in issues):
            completed.append(batch_id)
            reviewed_ids.extend(actual)

    if len(reviewed_ids) != len(set(reviewed_ids)):
        issues.append("duplicate neutral IDs across completed batches")
    payload = {
        "audit_label": "Codex-assisted evidence audit / Codex辅助证据审计",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "gold_loaded": False,
        "total_batches": len(batches),
        "completed_batches": completed,
        "completed_batch_count": len(completed),
        "reviewed_route_count": len(reviewed_ids),
        "remaining_route_count": 900 - len(reviewed_ids),
        "validation_issues": issues,
    }
    atomic_json(root / "progress.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
