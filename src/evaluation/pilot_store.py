"""Resume-safe per-case checkpoints for future pilot live execution."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SAFE_CASE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class CaseCheckpointStore:
    """Persist each completed case atomically without overwriting by default."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, case_id: str) -> Path:
        if not SAFE_CASE_ID.fullmatch(case_id):
            raise ValueError("Unsafe case_id for checkpoint path")
        return self.root / f"{case_id}.json"

    def save(self, case_id: str, payload: dict[str, Any], *, overwrite: bool = False) -> Path:
        path = self.path_for(case_id)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Checkpoint already exists: {case_id}")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def load(self, case_id: str) -> dict[str, Any] | None:
        path = self.path_for(case_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    def completed_case_ids(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.json"))

