import json
import os
import subprocess
import sys

from src.experiments.egopolice_av_organizer_v1.artifact_manifest import (
    load_manifest, verify_artifact_hashes,
)
from src.experiments.egopolice_av_organizer_v1.config import PHASE_IDS, ROOT
from src.experiments.egopolice_av_organizer_v1.validation import (
    validate_evidence_map, validate_presenter,
)


def test_manifest_paths_and_hashes():
    manifest = load_manifest()
    for path in manifest["canonical_artifact_paths"]:
        assert (ROOT / path).exists(), path
    assert all(x["matches"] for x in verify_artifact_hashes())


def test_nine_ordered_phases_and_full_duration():
    result = validate_evidence_map()
    assert result["valid"], result
    phases = json.loads((ROOT / "outputs/experiments/egopolice_av_organizer_v1_freeze/frozen_phase_boundaries.json").read_text())["phases"]
    assert [x["phase_id"] for x in phases] == PHASE_IDS
    assert sum(x["end_sec"] - x["start_sec"] for x in phases) == 1235.307


def test_exact_copy_atoms_and_typed_links():
    result = validate_evidence_map()
    assert result["exact_copy_visual_atoms"]
    assert result["exact_copy_transcript_atoms"]
    assert result["visual_atoms"] == 18
    assert result["transcript_atoms"] == 33


def test_presenter_continuity_and_required_content():
    result = validate_presenter()
    assert result["valid"], result
    assert result["phase_count"] == 9


def test_no_api_key_literal_in_freeze_files():
    roots = [
        ROOT / "src/experiments/egopolice_av_organizer_v1",
        ROOT / "scripts/experiments/run_egopolice_av_organizer_v1.py",
        ROOT / "outputs/experiments/egopolice_av_organizer_v1_freeze",
        ROOT / "docs/experiments/egopolice_av_organizer_v1_freeze.md",
    ]
    secret = os.environ.get("ANTHROPIC_API_KEY", "")
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore")
                assert "sk-ant-" not in text
                if secret:
                    assert secret not in text


def test_verify_only_has_zero_api_calls():
    command = [
        sys.executable,
        str(ROOT / "scripts/experiments/run_egopolice_av_organizer_v1.py"),
        "--video-id", "540772226",
        "--reuse-existing-artifacts", "--verify-only",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["api_calls"] == 0


def test_fresh_run_requires_explicit_api_flag():
    command = [
        sys.executable,
        str(ROOT / "scripts/experiments/run_egopolice_av_organizer_v1.py"),
        "--fresh-run",
    ]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "--allow-api-calls" in completed.stderr
