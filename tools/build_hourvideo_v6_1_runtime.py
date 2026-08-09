"""Create a self-contained, source-identical HourVideo V6.1 runtime snapshot.

Unlike build_hourvideo_v3_runtime.py, this tool does NOT read the working
tree. Every copied file is read as a git blob at a pinned revision, so the
snapshot is provably the frozen V6.1 lineage and cannot be contaminated by
whatever happens to be staged/untracked in the working copy at build time
(as of this writing: staged V6.2 files, ~90 untracked V4/V5 experiment
directories).

Two revisions are used, because V6.1's actual source closure spans a gap
in this repo's history:

  CORE_REVISION (749091f...): where V6/V6.1 (hourvideo_r1_av_r3_2_
  coarse_locked_claim_loop_v6) and its transitive `experiments.*` import
  closure were first committed. Verified by statically resolving every
  `from experiments.X import ...` / `import experiments.X` reachable from
  the three pipeline entry scripts (selection is NOT one of them -- see
  below) -- not by hand-copying the commit message's file list.

  SELECTION_REVISION (3a5eea4...): hourvideo_ten_video_pilot_selection_v1
  was never committed to the canonical src/experiments/ tree at any
  revision -- it only exists, as a copy, inside the hourvideo_v3_runtime/
  school-deployment snapshot committed here. That copy is byte-identical
  to the current working-tree copy (diffed at build-tool authoring time),
  so it is the only traceable source for this module.

Selection itself is NOT run as part of building this snapshot and no
selected-video list is baked in: the operator picks and validates videos
on the school machine, against whatever HourVideo files are actually
present there (see README.md in the destination).

Excluded on purpose: V6.2 (unfrozen, still under test), V4/V5 experiments,
generated outputs, datasets, model weights, credentials.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "hourvideo_v6_1_runtime"

CORE_REVISION = "749091f636978bd4012ccd71a5debd5fca7280b0"
SELECTION_REVISION = "3a5eea47da92fe97b3b578401e7091eed65d62ef"

# Verified by static import resolution (both `from experiments.X import` and
# `from src.experiments.X import` spellings -- both are used in this tree)
# from:
#   scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py
#   scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py
#   scripts/experiments/run_hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.py
# at CORE_REVISION. Every entry below was actually imported, transitively;
# none were added by guessing from names or the original commit message.
#
# Deliberately NOT included despite matching the same grep: claim_guided_
# gemini_review_loop_v2, claim_handoff_routing_v2_2, claim_reliability_
# gate_v2_1. These are only imported lazily inside claim_level_av_
# sufficiency_v3_minimal_contract.core.compatibility_report(), a function
# nothing on the V6.1 path calls (confirmed by grep: its only caller is
# inside that same module's own standalone report-generation code, which
# nothing here reaches). They are also absent from CORE_REVISION's tree.
CORE_SOURCE_PACKAGES = (
    "claim_level_av_sufficiency_v2",
    "claim_level_av_sufficiency_v3_1_requirement_centric",
    "claim_level_av_sufficiency_v3_minimal_contract",
    "egopolice_r1_av_dual_channel_retrieval_smoke_v1",
    "fine_reranking",
    "hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6",
    "hourvideo_r1_av_r3_2_frozen_gate_cache_replay_v1",
    "hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1",
    "hourvideo_r1_av_r3_2_single_video_smoke",
    "hourvideo_r1_av_r3_2_ten_video_pilot_v1",
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

CORE_SOURCE_FILES = (
    "src/__init__.py",
    "src/experiments/__init__.py",
    "scripts/experiments/run_hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.py",
    "scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py",
    "scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py",
    "configs/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.json",
    "configs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1.json",
    "configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json",
    "docs/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.md",
    "docs/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1.md",
    "docs/experiments/hourvideo_v6_coarse_locked_claim_audit_loop_plan.md",
    "tests/experiments/claim_level_av_sufficiency_v2/test_claim_level_av_sufficiency_v2.py",
    "tests/experiments/claim_level_av_sufficiency_v3_minimal_contract/test_contract.py",
    "tests/experiments/hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1/test_contract.py",
    "tests/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1/test_live_contract.py",
    "tests/experiments/hourvideo_r1_av_r3_2_ten_video_pilot_v1/test_offline.py",
    "tests/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6/test_contract.py",
)
# hourvideo_r1_av_r3_2_single_video_smoke/test_contract.py and
# hourvideo_r1_av_r3_2_frozen_gate_cache_replay_v1/test_replay.py are
# deliberately excluded: both assert against outputs/experiments/
# hourvideo_r1_av_r3_2_single_video_smoke_v1/*.json, a generated fixture
# from one past run on this machine, not portable source -- they cannot
# pass in a fresh checkout regardless of code correctness.

# Sourced from SELECTION_REVISION, under its hourvideo_v3_runtime/ prefix
# (see module docstring for why). (source_relative, dest_relative) pairs.
SELECTION_FILES = (
    ("hourvideo_v3_runtime/src/experiments/hourvideo_ten_video_pilot_selection_v1/__init__.py",
     "src/experiments/hourvideo_ten_video_pilot_selection_v1/__init__.py"),
    ("hourvideo_v3_runtime/src/experiments/hourvideo_ten_video_pilot_selection_v1/core.py",
     "src/experiments/hourvideo_ten_video_pilot_selection_v1/core.py"),
    ("hourvideo_v3_runtime/scripts/experiments/run_hourvideo_ten_video_pilot_selection_v1.py",
     "scripts/experiments/run_hourvideo_ten_video_pilot_selection_v1.py"),
    ("hourvideo_v3_runtime/configs/experiments/hourvideo_ten_video_pilot_selection_v1.json",
     "configs/experiments/hourvideo_ten_video_pilot_selection_v1.json"),
    ("hourvideo_v3_runtime/tests/experiments/hourvideo_ten_video_pilot_selection_v1/test_selection.py",
     "tests/experiments/hourvideo_ten_video_pilot_selection_v1/test_selection.py"),
)

GENERATED_FILES = {
    ".env.example": "ANTHROPIC_API_KEY=\nGEMINI_API_KEY=\n",
    "pytest.ini": (
        "[pytest]\n"
        "addopts = --import-mode=importlib\n"
        "pythonpath = . src\n"
    ),
    "requirements.txt": (
        "# Install a CUDA-compatible torch build first when using a GPU.\n"
        "numpy\nanthropic\ngoogle-genai\npillow\npydantic\nsoundfile\ntransformers\n"
        "ultralytics\nopenai-whisper\n"
        "# torch is intentionally not pinned: choose the build matching the school GPU/CUDA.\n"
    ),
    ".gitignore": (
        ".env\n__pycache__/\n.pytest_cache/\noutputs/\ndata/\n*.mp4\n*.avi\n*.mov\n*.mkv\n"
        "*.jpg\n*.jpeg\n*.png\n*.wav\n*.npy\n*.npz\n*.safetensors\n*.pt\n*.pth\n"
    ),
    "README.md": """# HourVideo V6.1 runtime

A source-identical runtime snapshot of the frozen V6.1 coarse-locked
claim/audit loop, built from git blobs (not the working tree) at two
pinned revisions -- see `runtime_manifest.json` for exactly which
revision each file came from and its SHA-256.

It deliberately excludes V6.2 (still under test, not frozen), V4/V5
experiments, generated outputs, datasets, model weights and credentials.

## Before running anything

Every JSON file under `configs/experiments/` has machine-specific paths
baked in (D:/Thesisdata/HourVideo/..., this machine's venv/model-cache
paths, etc.) -- rewrite them for the school filesystem first.

`configs/experiments/hourvideo_ten_video_pilot_selection_v1.json` also
has a `selected_cases` list from the original 10-video pilot on this
machine. Selection was intentionally NOT run as part of building this
snapshot, and no video list is authoritative here -- replace
`selected_cases` (and `hourvideo_root`, `frame_audit_path`,
`video_audit_path`) with your own choices, validated against whatever
HourVideo video files actually exist on the school machine, before
running the selection stage.

## Stages

```powershell
python scripts/experiments/run_hourvideo_r1_av_r3_2_no_api_interface_validation_v1_1.py
python scripts/experiments/run_hourvideo_ten_video_pilot_selection_v1.py
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage prepare
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage audio
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage detector
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage captions
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage preflight
python scripts/experiments/run_hourvideo_r1_av_r3_2_ten_video_pilot_v1.py --stage live
python scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py --config configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json --stage preflight
python scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py --config configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json --stage live
python scripts/experiments/run_hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6.py --config configs/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6_1.json --stage evaluate
```

`runtime_manifest.json` records the source revision and SHA-256 for every
copied file, so this snapshot can be checked against the original source.
""",
}


def git_show(revision: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def core_source_paths() -> list[str]:
    paths = list(CORE_SOURCE_FILES)
    for package in CORE_SOURCE_PACKAGES:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", CORE_REVISION, "--",
             f"src/experiments/{package}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout.decode("utf-8")
        paths.extend(line for line in listing.splitlines() if line.endswith(".py"))
    return sorted(set(paths))


def build() -> None:
    if DESTINATION.resolve().parent != ROOT.resolve() or DESTINATION.name != "hourvideo_v6_1_runtime":
        raise RuntimeError(f"Refusing to replace an unsafe destination: {DESTINATION}")
    if DESTINATION.exists():
        import shutil
        shutil.rmtree(DESTINATION)

    records = []

    for relative in core_source_paths():
        data = git_show(CORE_REVISION, relative)
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({
            "path": relative, "revision": CORE_REVISION,
            "sha256": sha256_bytes(data), "bytes": len(data),
        })

    for source_relative, dest_relative in SELECTION_FILES:
        data = git_show(SELECTION_REVISION, source_relative)
        target = DESTINATION / dest_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        records.append({
            "path": dest_relative, "revision": SELECTION_REVISION,
            "source_path_in_revision": source_relative,
            "sha256": sha256_bytes(data), "bytes": len(data),
        })

    for relative, content in GENERATED_FILES.items():
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")

    manifest = {
        "snapshot": "hourvideo_v6_1_runtime",
        "source_identity": "V6.1 coarse-locked claim/audit loop, full pipeline "
                            "(selection through live claim/audit)",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "core_revision": CORE_REVISION,
        "selection_revision": SELECTION_REVISION,
        "selection_revision_note": (
            "hourvideo_ten_video_pilot_selection_v1 was never committed to "
            "src/experiments/ directly; sourced from its only tracked copy, "
            "inside the hourvideo_v3_runtime/ snapshot at this revision."
        ),
        "copied_file_count": len(records),
        "copied_files": records,
        "excluded": [
            "V6.2 (unfrozen, still under test)", "V4", "V5", "video selection results",
            "datasets", "outputs", "model weights", "credentials",
        ],
    }
    (DESTINATION / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"destination": str(DESTINATION), "copied_file_count": len(records)}, indent=2))


if __name__ == "__main__":
    build()
