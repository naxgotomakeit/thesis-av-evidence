from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE.parents[2]
AUDIT_ROOT = (
    RUNTIME_ROOT
    / "outputs/experiments/hourvideo_v7_3_videoseal_eval300_runtime_aligned_paired_v1/pre_eval300_audit"
)
PACKAGE_NAME = "hourvideo_v7_3_videoseal_eval300_runtime_aligned_paired_v1_source_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    archive = AUDIT_ROOT / f"{PACKAGE_NAME}.tar.gz"
    listing = AUDIT_ROOT / f"{PACKAGE_NAME}.contents.json"
    staging_parent = AUDIT_ROOT / "transfer_package_staging"
    staging = staging_parent / PACKAGE_NAME
    for target in (archive, listing, staging_parent):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing package target: {target}")

    sources: dict[str, Path] = {
        "README.md": HERE / "README.md",
        "DATA_MACHINE_RUNBOOK.md": HERE / "DATA_MACHINE_RUNBOOK.md",
        "DIFF_FROM_REFERENCE_48.md": HERE / "DIFF_FROM_REFERENCE_48.md",
        "config.json": HERE / "config.json",
        "run.py": HERE / "run.py",
        "paired_eval_statistics.py": HERE / "paired_eval_statistics.py",
        "pre_eval300_audit.py": HERE / "pre_eval300_audit.py",
        "test_contract.py": HERE / "test_contract.py",
        "three_question_uids.txt": HERE / "three_question_uids.txt",
        "reference_overlay/videoseal/tools/tool_map.py": HERE / "reference_runtime/videoseal/tools/tool_map.py",
        "reference_overlay/videoseal/tools/visual_tools.py": HERE / "reference_runtime/videoseal/tools/visual_tools.py",
        "reference_overlay/videoseal/tools/retrieval_adapter.py": HERE / "reference_runtime/videoseal/tools/retrieval_adapter.py",
        "runtime_support/src/experiments/hourvideo_v7_1_videoseal_frozen_downstream_smoke_v1/__init__.py": RUNTIME_ROOT / "src/experiments/hourvideo_v7_1_videoseal_frozen_downstream_smoke_v1/__init__.py",
        "runtime_support/src/experiments/hourvideo_v7_1_videoseal_frozen_downstream_smoke_v1/core.py": RUNTIME_ROOT / "src/experiments/hourvideo_v7_1_videoseal_frozen_downstream_smoke_v1/core.py",
        "runtime_support/src/experiments/planner_medium_retrieval/__init__.py": RUNTIME_ROOT / "src/experiments/planner_medium_retrieval/__init__.py",
        "runtime_support/src/experiments/planner_medium_retrieval/core.py": RUNTIME_ROOT / "src/experiments/planner_medium_retrieval/core.py",
        "runtime_support/src/experiments/planner_medium_retrieval/schemas.py": RUNTIME_ROOT / "src/experiments/planner_medium_retrieval/schemas.py",
        "runtime_support/src/experiments/planner_medium_retrieval/validation.py": RUNTIME_ROOT / "src/experiments/planner_medium_retrieval/validation.py",
        "runtime_support/src/experiments/fine_reranking/__init__.py": RUNTIME_ROOT / "src/experiments/fine_reranking/__init__.py",
        "runtime_support/src/experiments/fine_reranking/core.py": RUNTIME_ROOT / "src/experiments/fine_reranking/core.py",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing package sources: {missing}")

    staging.mkdir(parents=True)
    records = []
    for relative, source in sorted(sources.items()):
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records.append(
            {
                "path": relative,
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )

    forbidden_names = {".env", "predictions", "trajectory", "metrics", "cache", "indexes", "models", "videos"}
    for record in records:
        parts = {part.lower() for part in Path(record["path"]).parts}
        if parts & forbidden_names:
            raise RuntimeError(f"forbidden package path: {record['path']}")
        content = (staging / record["path"]).read_text(encoding="utf-8", errors="ignore")
        credential_patterns = (
            r"\bsk-[A-Za-z0-9_-]{20,}",
            r"(?:OPENAI|EMBEDDING)_API_KEY\s*=\s*['\"]?(?!local\b|\$\{|os\.getenv)[^\s'\"]{12,}",
        )
        if any(re.search(pattern, content) for pattern in credential_patterns):
            raise RuntimeError(f"possible credential in package source: {record['path']}")

    manifest = {
        "package": PACKAGE_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reference_bundle_sha256": "ffcb174919b40dc3d278ce325e054c83716353c61de1c496600fbd960a51fe84",
        "experiment_type": "accuracy-efficiency comparison",
        "formal_backends": [
            "videoseal_flat_top30",
            "hierarchical_fixed_width_3_3_6_lexical_coarse",
        ],
        "payload_file_count": len(records),
        "payload_files": records,
        "exclusions": [
            "credentials/.env",
            "models",
            "videos",
            "indexes",
            "predictions",
            "trajectories",
            "metrics",
            "caches",
        ],
    }
    manifest_path = staging / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contents_path = staging / "CONTENTS.sha256"
    contents_path.write_text(
        "".join(f"{record['sha256']}  {record['path']}\n" for record in records),
        encoding="utf-8",
    )

    with tarfile.open(archive, "w:gz") as handle:
        handle.add(staging, arcname=PACKAGE_NAME, recursive=True)
    archive_record = {
        "archive": str(archive.resolve()),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "archive_file_count": len(records) + 2,
        "payload_file_count": len(records),
        "manifest": manifest,
    }
    listing.write_text(json.dumps(archive_record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(archive_record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
