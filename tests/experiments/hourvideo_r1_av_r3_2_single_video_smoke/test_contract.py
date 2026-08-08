from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.hourvideo_r1_av_r3_2_single_video_smoke.live_runner import REQUIREMENTS, _build_r1_map, _parent_map


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.out = ROOT / "outputs/experiments/hourvideo_r1_av_r3_2_single_video_smoke_v1"
        cls.hierarchy = json.loads((cls.out / "shared_hierarchy.json").read_text(encoding="utf-8"))
        cls.projections = json.loads((cls.out / "r1_medium_projection.json").read_text(encoding="utf-8"))
        cls.audio = json.loads((cls.out / "audio_asr.json").read_text(encoding="utf-8"))["segments"]
        cls.embeddings = np.load(cls.out / "medium_siglip.float32.npy", allow_pickle=False)

    def test_shared_structure(self) -> None:
        self.assertEqual(len(self.hierarchy["medium_nodes"]), 36)
        self.assertEqual(len(self.hierarchy["fine_nodes"]), 107)
        self.assertEqual(len(self.projections), 36)
        self.assertEqual(self.embeddings.shape, (36, 768))

    def test_r1_map_complete_and_navigation_only(self) -> None:
        doc = _build_r1_map(self.hierarchy, self.projections, self.audio, self.embeddings)
        self.assertFalse(doc["semantic_fields_available"])
        self.assertFalse(doc["hard_filtering_allowed"])
        parent = _parent_map(doc)
        self.assertEqual(set(parent), {r["medium_id"] for r in self.hierarchy["medium_nodes"]})

    def test_r1_caption_leakage_zero(self) -> None:
        forbidden = ('"qwen_caption"', '"vlm_caption":', '"visual_caption"')
        text = json.dumps(self.projections, ensure_ascii=False).lower()
        self.assertFalse(any(term in text for term in forbidden))

    def test_question_has_five_safe_options_and_no_gold(self) -> None:
        question = json.loads((self.out / "question_input.json").read_text(encoding="utf-8"))
        self.assertEqual([r["option_id"] for r in question["answer_options"]], list("ABCDE"))
        self.assertFalse(any(key in question for key in ("answer", "label", "correct_option", "ground_truth")))

    def test_requirements_are_shared_and_unique(self) -> None:
        ids = [row["requirement_id"] for row in REQUIREMENTS]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()
