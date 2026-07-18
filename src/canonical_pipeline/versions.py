"""Explicit canonical version map; never inferred from filenames or mtimes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanonicalConfig:
    """Validated canonical versions, source lineage, models and project paths."""

    project_root: Path
    raw: dict[str, Any]

    @property
    def versions(self) -> dict[str, str]:
        return dict(self.raw["canonical_versions"])

    def path(self, name: str) -> Path:
        return self.project_root / self.raw["paths"][name]


def load_canonical_config(project_root: Path, config_path: Path | None = None) -> CanonicalConfig:
    """Load the checked-in source-of-truth mapping without version inference."""
    path = config_path or project_root / "config/canonical_pipeline.json"
    if not path.is_file():
        raise FileNotFoundError(f"Canonical pipeline configuration is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {"task5a": "v2", "task5b": "v1.1", "task5c": "v1.2", "task6": "v1.2", "task7a": "v1", "task7b": "v3"}
    if raw.get("canonical_versions") != expected:
        raise ValueError("Canonical version map differs from the frozen research baseline")
    return CanonicalConfig(project_root=project_root, raw=raw)

