from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "outputs/experiments/hourvideo_r1_av_r3_2_single_video_smoke_v1"


class ReplaySourceTests(unittest.TestCase):
    def test_source_has_two_initial_sufficiency_calls(self) -> None:
        rows = json.loads((SOURCE / "live_cost_progress.json").read_text(encoding="utf-8"))
        sides = {row["side"] for row in rows if row.get("stage") == "sufficiency_initial"}
        self.assertEqual(sides, {"r1_av", "r3_2"})

    def test_source_has_no_gold_in_question(self) -> None:
        question = json.loads((SOURCE / "question_input.json").read_text(encoding="utf-8"))
        self.assertNotIn("gold", question)
        self.assertNotIn("correct", question)

    def test_maps_remain_navigation_only(self) -> None:
        for name in ("r1_av_navigation_map.json", "r3_2_navigation_map.json"):
            value = json.loads((SOURCE / name).read_text(encoding="utf-8"))
            self.assertFalse(value["hard_filtering_allowed"])


if __name__ == "__main__":
    unittest.main()
