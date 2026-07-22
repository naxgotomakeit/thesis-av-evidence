"""Formal frozen EgoPolice Blind-98 entry point."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines.egopolice_formal.runner import main


if __name__ == "__main__":
    raise SystemExit(main(["--condition", "blind", *sys.argv[1:]]))
