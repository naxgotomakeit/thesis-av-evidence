import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.visual.micro_clips import build_micro_clips, micro_clip_ranges, temporal_overlap


class MicroClipTests(unittest.TestCase):
    def test_ranges_include_short_tail(self):
        self.assertEqual(micro_clip_ranges(7), [(0, 4), (2, 6), (4, 7)])
        self.assertEqual(micro_clip_ranges(0), [])
        self.assertEqual(micro_clip_ranges(8, duration=8.2), [(0, 4), (2, 6), (4, 8), (6, 8)])

    def test_pool_representative_and_motion(self):
        paths = sorted(Path("outputs/visual_index/00002/frames_1fps").glob("frame_*.jpg"))[:3]
        self.assertEqual(len(paths), 3)
        emb = np.array([[1, 0], [.9, .1], [0, 1]], np.float32)
        rows, pooled = build_micro_clips(emb, paths, 3.0)
        self.assertEqual(pooled.shape, (1, 2))
        self.assertEqual(rows[0]["representative_frame_index"], 1)
        self.assertGreaterEqual(rows[0]["motion_magnitude"], 0)

    def test_half_open_overlap(self):
        item = {"start_time": 2, "end_time": 6}
        self.assertTrue(temporal_overlap(item, 5, 7))
        self.assertFalse(temporal_overlap(item, 6, 7))

    def test_coverage_assignment_and_metadata_alignment(self):
        ranges = micro_clip_ranges(9)
        self.assertEqual(ranges, [(0, 4), (2, 6), (4, 8), (6, 9)])
        self.assertEqual(set(i for a, b in ranges for i in range(a, b)), set(range(9)))

    def test_index_construction_interface_has_no_qa_inputs(self):
        import inspect
        parameters = set(inspect.signature(build_micro_clips).parameters)
        forbidden = {"question", "answer", "answer_options", "context", "reference_timestamp"}
        self.assertFalse(parameters & forbidden)

    def test_deterministic_descending_cosine_retrieval(self):
        from src.visual.micro_clips import rank_cosine
        embeddings = np.array([[0, 1], [1, 0], [.7, .7]], np.float32)
        rows = [{"embedding_row_index": i} for i in range(3)]
        first = rank_cosine(np.array([1, 0], np.float32), embeddings, rows, 3)
        second = rank_cosine(np.array([1, 0], np.float32), embeddings, rows, 3)
        self.assertEqual([x["embedding_row_index"] for x in first], [1, 2, 0])
        self.assertEqual([x["embedding_row_index"] for x in first], [x["embedding_row_index"] for x in second])
        self.assertEqual([x["rank"] for x in first], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
