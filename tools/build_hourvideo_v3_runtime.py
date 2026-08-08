"""Create a self-contained, source-identical HourVideo V3 runtime snapshot.

The snapshot is intentionally a copy rather than a refactor: it keeps the
historical import paths and entry points unchanged, while excluding V4/V5,
generated artifacts, local data, and secrets.  This makes it suitable for a
separate school-server checkout without mutating the research lineage.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "hourvideo_v3_runtime"

# This is the complete static runtime closure for:
# selection -> no-API question projection -> V1 offline/live R1_AV + R3_2
# -> V3 question-symmetric visual review.  Do not add V4/V5 experiments here.
SOURCE_PACKAGES = (
    "claim_level_av_sufficiency_v2",
    "claim_level_av_sufficiency_v3_1_requirement_centric",
    "claim_level_av_sufficiency_v3_minimal_contract",
    "egopolice_r1_av_dual_channel_retrieval_smoke_v1",
    "fine_reranking",
    "hourvideo_r1_av_r3_2_frozen_gate_cache_replay_v1",
    "hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1",
    "hourvideo_r1_av_r3_2_question_batched_visual_review_v2",
    "hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3",
    "hourvideo_r1_av_r3_2_single_video_smoke",
    "hourvideo_r1_av_r3_2_ten_video_pilot_v1",
    "hourvideo_ten_video_pilot_selection_v1",
    "medium_semantic_abstraction",
    "planner_medium_retrieval",
    "r1_av_r3_2_claim_guided_gemini_closed_loop_v3",
    "r1_av_r3_2_requirement_centric_pipeline_canary",
    "r1_av_r3_2_review_cache_integration_canary_v1",
    "r1_r3_v2_map_aware_all_medium_retrieval_pair",
    "reviewed_visual_evidence_cache_v1",
    "reviewed_visual_evidence_cache_v1_gemini_canary",
    "shared_question_scope_review_gate_v1",
    "shared_sufficiency_v3_2_1_temporal_anchor",
    "shared_sufficiency_v3_2_contract",
)

SOURCE_FILES = (
    "src/__init__.py",
    "src/experiments/__init__.py",
    "scripts/experiments/run_hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.py",
    "scripts/experiments/run_hourvideo_ten_video_pilot_selection_v1.py",
    "scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py",
    "scripts/experiments/run_hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.py",
    "configs/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.json",
    "configs/experiments/hourvideo_ten_video_pilot_selection_v1.json",
    "configs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1.json",
    "configs/experiments/hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.json",
    "docs/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.md",
    "docs/experiments/hourvideo_ten_video_pilot_selection_v1.md",
    "docs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1.md",
    "docs/experiments/hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.md",
    "tests/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1/test_contract.py",
    "tests/experiments/hourvideo_ten_video_pilot_selection_v1/test_selection.py",
    "tests/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1/test_live_contract.py",
    "tests/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1/test_offline.py",
    "tests/experiments/hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3/test_contract.py",
)

GENERATED_FILES = {
    ".env.example": "ANTHROPIC_API_KEY=\nGEMINI_API_KEY=\n",
    "requirements.txt": """# Install a CUDA-compatible torch build first when using a GPU.\nnumpy\nanthropic\npillow\npydantic\nsoundfile\ntransformers\nultralytics\nopenai-whisper\n# torch is intentionally not pinned: choose the build matching the school GPU/CUDA.\n""",
    ".gitignore": """.env\n__pycache__/\n.pytest_cache/\noutputs/\ndata/\n*.mp4\n*.avi\n*.mov\n*.mkv\n*.jpg\n*.jpeg\n*.png\n*.wav\n*.npy\n*.npz\n*.safetensors\n*.pt\n*.pth\n""",
    "README.md": """# HourVideo V3 runtime\n\nA source-identical runtime snapshot of the HourVideo ten-video V3 baseline.\nIt contains the complete code path from the no-API question projection and\nten-video selection through hierarchy construction, ASR, detector/caption\nstages, R1_AV/R3_2 map-assisted retrieval, and V3 question-symmetric visual\nreview.\n\nIt deliberately excludes V4/V5 experiments, generated outputs, datasets,\nmodel weights and credentials. Configure the four JSON files under\n`configs/experiments/` for the school filesystem before running.\n\n## Stages\n\n```powershell\npython scripts/experiments/run_hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.py\npython scripts/experiments/run_hourvideo_ten_video_pilot_selection_v1.py\npython scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage prepare\npython scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage audio\npython scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage detector\npython scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage captions\npython scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage preflight\npython scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage live\npython scripts/experiments/run_hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.py --stage preflight\npython scripts/experiments/run_hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3.py --stage live\n```\n\n`runtime_manifest.json` records SHA-256 identities for every copied source\nfile, so this snapshot can be checked against the original V3 source.\n""",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    paths = [ROOT / relative for relative in SOURCE_FILES]
    for package in SOURCE_PACKAGES:
        paths.extend((ROOT / "src" / "experiments" / package).glob("*.py"))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing V3 runtime source: " + ", ".join(str(path) for path in missing))
    return sorted(set(paths))


def build() -> None:
    if DESTINATION.resolve().parent != ROOT.resolve() or DESTINATION.name != "hourvideo_v3_runtime":
        raise RuntimeError(f"Refusing to replace an unsafe destination: {DESTINATION}")
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    records = []
    for source in source_paths():
        relative = source.relative_to(ROOT)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append({"path": relative.as_posix(), "sha256": sha256(source), "bytes": source.stat().st_size})
    for relative, content in GENERATED_FILES.items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    manifest = {
        "snapshot": "hourvideo_v3_runtime",
        "source_identity": "V1 R1_AV/R3_2 ten-video pilot + V3 question-symmetric review",
        "copied_file_count": len(records),
        "copied_files": records,
        "excluded": ["V4", "V5", "datasets", "outputs", "model weights", "credentials"],
    }
    (DESTINATION / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"destination": str(DESTINATION), "copied_file_count": len(records)}, indent=2))


if __name__ == "__main__":
    build()
