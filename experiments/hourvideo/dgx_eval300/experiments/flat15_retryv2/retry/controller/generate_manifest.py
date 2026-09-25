#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for path in sorted(p for p in root.rglob("*") if p.is_file()):
    rel = path.relative_to(root)
    if rel.as_posix() == "MANIFEST.sha256" or "__pycache__" in rel.parts or rel.suffix == ".pyc":
        continue
    rows.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {rel.as_posix()}")
(root / "MANIFEST.sha256").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(f"manifest_entries={len(rows)}")
