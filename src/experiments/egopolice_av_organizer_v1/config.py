from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FREEZE_ID = "egopolice_av_organizer_v1_freeze"
FREEZE_ROOT = ROOT / "outputs" / "experiments" / FREEZE_ID
MANIFEST_PATH = FREEZE_ROOT / "freeze_manifest.json"
HASHES_PATH = FREEZE_ROOT / "artifact_hashes.json"
CONFIG_PATH = FREEZE_ROOT / "frozen_config.json"
PHASES_PATH = FREEZE_ROOT / "frozen_phase_boundaries.json"
SCHEMA_PATH = FREEZE_ROOT / "frozen_schema.json"
PROMPTS_PATH = FREEZE_ROOT / "frozen_prompts.md"
VIDEO_ID = "540772226"
PHASE_IDS = [f"P{i:02d}" for i in range(1, 10)]
