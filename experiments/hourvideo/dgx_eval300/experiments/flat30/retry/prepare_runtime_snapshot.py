#!/usr/bin/env python3
"""Freeze the local Python import closure and DGX launch files used by Eval300."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REPO = Path("/home/naxucl/projects/VideoSEAL")
OUT = Path("/home/naxucl/data/HourVideo/videoseal_original/eval300_timeout_retry_20260819T084545Z/.runtime_snapshot")
ENTRY_MODULES = ["videoseal.cli.run_from_parquet", "videoseal.runner.per_question_runner"]
EXPLICIT_FILES = [
    "pyproject.toml",
    "uv.lock",
    "scripts/dgx/README.md",
    "scripts/dgx/common.env",
    "scripts/dgx/start_model.sh",
    "scripts/dgx/run_eval300.sh",
    "scripts/dgx/smoke_one.sh",
    "scripts/dgx/memory_logger.sh",
]


def module_path(name: str) -> Path | None:
    base = REPO.joinpath(*name.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def imports_for(module: str, path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    package = module.split(".")[:-1] if path.name != "__init__.py" else module.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names if alias.name.startswith("videoseal"))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                base = ".".join(prefix + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base.startswith("videoseal"):
                found.add(base)
                for alias in node.names:
                    found.add(f"{base}.{alias.name}")
    return found


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing existing runtime snapshot: {OUT}")
    queue = list(ENTRY_MODULES)
    modules: dict[str, Path] = {}
    while queue:
        module = queue.pop()
        path = module_path(module)
        if path is None:
            continue
        canonical = str(path.relative_to(REPO))
        if canonical in {str(p.relative_to(REPO)) for p in modules.values()}:
            continue
        modules[module] = path
        queue.extend(sorted(imports_for(module, path)))

    files = {path.relative_to(REPO) for path in modules.values()}
    files.update(Path(item) for item in EXPLICIT_FILES)
    for relative in list(files):
        parent = relative.parent
        while str(parent) not in (".", ""):
            init = parent / "__init__.py"
            if (REPO / init).is_file():
                files.add(init)
            parent = parent.parent

    source = OUT / "source"
    for relative in sorted(files):
        src = REPO / relative
        if not src.is_file():
            raise FileNotFoundError(src)
        dst = source / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    metadata = OUT / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "git_head.txt").write_text(git("rev-parse", "HEAD"), encoding="utf-8")
    (metadata / "git_branch.txt").write_text(git("branch", "--show-current"), encoding="utf-8")
    (metadata / "git_status.txt").write_text(git("status", "--short"), encoding="utf-8")
    diff = subprocess.run(
        ["git", "-C", str(REPO), "diff", "--binary", "HEAD", "--", *[str(p) for p in sorted(files)]],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    (metadata / "runtime_dirty.patch").write_bytes(diff)
    manifest = [
        {"path": str(relative), "sha256": sha256(source / relative), "size": (source / relative).stat().st_size}
        for relative in sorted(files)
    ]
    (metadata / "files.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (metadata / "snapshot.json").write_text(
        json.dumps(
            {
                "source_repository": str(REPO),
                "branch": git("branch", "--show-current").strip(),
                "head": git("rev-parse", "HEAD").strip(),
                "file_count": len(manifest),
                "entry_modules": ENTRY_MODULES,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"snapshot": str(OUT), "files": len(manifest)}, sort_keys=True))


if __name__ == "__main__":
    main()
