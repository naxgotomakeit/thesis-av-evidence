import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_ten_video_pilot_v1.offline import case_configs


def test_case_configs_is_callable() -> None:
    assert callable(case_configs)
