"""Deterministic, secret-free fingerprints for comparable canonical runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from src.final_qa.canonical_schema import (
    CANONICAL_OUTPUT_SCHEMA_VERSION,
    canonical_output_schema_hash,
)
from src.final_qa.task7b_gemini import SYSTEM_INSTRUCTION

from .contracts import EVIDENCE_CONTRACT_VERSION
from .versions import CanonicalConfig


FINGERPRINT_VERSION = "canonical-run-fingerprint-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={project_root.as_posix()}", *arguments],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("A Git commit is required for canonical run fingerprinting") from exc
    return {"commit_sha": commit, "tracked_worktree_dirty": dirty}


def _index_files(project_root: Path, video_id: str) -> dict[str, tuple[Path, Path]]:
    return {
        "visual": (
            project_root / "outputs/visual_index" / video_id / "visual_state_regions.json",
            project_root / "outputs/visual_index" / video_id / "region_embeddings.npy",
        ),
        "speech": (
            project_root / "outputs/audio_index" / video_id / "transcript_embedding_index.json",
            project_root / "outputs/audio_index" / video_id / "transcript_embeddings.npy",
        ),
        "acoustic": (
            project_root / "outputs/audio_index" / video_id / "acoustic_embedding_index.json",
            project_root / "outputs/audio_index" / video_id / "acoustic_embeddings.npy",
        ),
    }


def index_fingerprint(project_root: Path, video_ids: Iterable[str]) -> dict[str, Any]:
    """Fingerprint metadata and embedding content for each reusable video index."""
    videos: dict[str, Any] = {}
    for video_id in sorted(set(map(str, video_ids))):
        modalities: dict[str, Any] = {}
        for modality, (metadata_path, embedding_path) in _index_files(project_root, video_id).items():
            missing = [path.as_posix() for path in (metadata_path, embedding_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Cannot fingerprint missing {modality} index for {video_id}: {missing}")
            modalities[modality] = {
                "metadata_path": metadata_path.relative_to(project_root).as_posix(),
                "metadata_sha256": _sha256_file(metadata_path),
                "embedding_path": embedding_path.relative_to(project_root).as_posix(),
                "embedding_sha256": _sha256_file(embedding_path),
                "embedding_bytes": embedding_path.stat().st_size,
            }
        videos[video_id] = modalities
    return {"videos": videos, "sha256": _canonical_hash(videos)}


def build_run_fingerprint(
    project_root: Path,
    config: CanonicalConfig,
    manifest_path: Path,
    video_ids: Iterable[str],
    *,
    planner_model: str | None,
    final_model: str = "gemini-3.5-flash",
    thinking_level: str = "low",
    store: bool = False,
) -> dict[str, Any]:
    """Build a complete future-run compatibility fingerprint without secrets."""
    canonical_config_path = project_root / "config/canonical_pipeline.json"
    retrieval_config_path = project_root / "configs/retrieval_mvp.yaml"
    task6_budget_path = config.path("task6_budget")
    planner_source = project_root / "src/question_planner/task5a.py"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run manifest is missing: {manifest_path}")
    indexes = index_fingerprint(project_root, video_ids)
    components = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "git": _git_state(project_root),
        "canonical_versions": config.versions,
        "config": {
            "canonical_pipeline_sha256": _sha256_file(canonical_config_path),
            "retrieval_mvp_sha256": _sha256_file(retrieval_config_path),
            "task6_budget_sha256": _sha256_file(task6_budget_path),
        },
        "planner": {
            "model": planner_model,
            "prompt_and_schema_source_sha256": _sha256_file(planner_source),
        },
        "final_qa": {
            "model": final_model,
            "thinking_level": thinking_level,
            "store": store,
            "system_instruction_sha256": _sha256_bytes(SYSTEM_INSTRUCTION.encode("utf-8")),
        },
        "schema": {
            "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
            "output_schema_version": CANONICAL_OUTPUT_SCHEMA_VERSION,
            "output_schema_sha256": canonical_output_schema_hash(),
        },
        "manifest": {
            "path": manifest_path.resolve().relative_to(project_root.resolve()).as_posix(),
            "sha256": _sha256_file(manifest_path),
        },
        "indexes": indexes,
    }
    return {**components, "fingerprint_sha256": _canonical_hash(components)}


def fingerprints_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two results were produced under the same full contract."""
    left_value = left.get("fingerprint_sha256")
    right_value = right.get("fingerprint_sha256")
    return bool(left_value and right_value and left_value == right_value)
