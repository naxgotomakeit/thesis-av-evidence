import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_ten_video_pilot_selection_v1.core import canonical


def test_canonical_is_deterministic() -> None:
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})


def test_canonical_has_newline() -> None:
    assert canonical({"a": 1}).endswith(b"\n")
