from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import HASHES_PATH, MANIFEST_PATH, ROOT


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, Any]:
    return load_json(MANIFEST_PATH)


def verify_artifact_hashes() -> list[dict[str, Any]]:
    records = load_json(HASHES_PATH)["artifacts"]
    results = []
    for record in records:
        path = ROOT / record["path"]
        actual = sha256_file(path) if path.is_file() else None
        results.append({
            **record,
            "exists": path.is_file(),
            "actual_sha256": actual,
            "matches": actual == record["sha256"],
        })
    return results
