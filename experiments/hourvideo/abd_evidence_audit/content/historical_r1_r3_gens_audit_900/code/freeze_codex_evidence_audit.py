#!/usr/bin/env python3
"""Freeze the completed neutral-ID evidence audit before score linkage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_root", type=Path)
    args = parser.parse_args()
    root = args.audit_root.resolve()
    progress = json.loads((root / "progress.json").read_text(encoding="utf-8"))
    if progress["reviewed_route_count"] != 900 or progress["validation_issues"]:
        raise SystemExit("refusing to freeze: audit is incomplete or invalid")

    manifest = json.loads((root / "batch_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict] = []
    batch_hashes: dict[str, str] = {}
    for batch in manifest["batches"]:
        path = root / "reviews" / f"{batch['batch_id']}.jsonl"
        raw = path.read_bytes()
        batch_hashes[batch["batch_id"]] = sha256_bytes(raw)
        parsed = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if [row["neutral_id"] for row in parsed] != batch["neutral_ids"]:
            raise SystemExit(f"batch mismatch: {batch['batch_id']}")
        rows.extend(parsed)
    expected = [f"E{index:04d}" for index in range(1, 901)]
    if [row["neutral_id"] for row in rows] != expected:
        raise SystemExit("neutral IDs are not exactly E0001..E0900")

    combined = b"".join(
        (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    combined_path = root / "evidence_audit_frozen.jsonl"
    atomic_write(combined_path, combined)
    payload = {
        "audit_label": "Codex-assisted evidence audit / Codex辅助证据审计",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "route_count": 900,
        "batch_count": len(manifest["batches"]),
        "neutral_id_first": "E0001",
        "neutral_id_last": "E0900",
        "evidence_audit_path": str(combined_path),
        "evidence_audit_sha256": sha256_bytes(combined),
        "batch_file_sha256": batch_hashes,
        "criteria_sha256": sha256_bytes((root / "AUDIT_CRITERIA_FROZEN.md").read_bytes()),
        "package_freeze_sha256": sha256_bytes((root / "package_freeze.json").read_bytes()),
        "classification_complete_before_score_linkage": True,
        "records_must_not_be_modified_after_freeze": True,
    }
    freeze_path = root / "evidence_audit_freeze.json"
    atomic_write(
        freeze_path,
        (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
