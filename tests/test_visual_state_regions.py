import unittest

import numpy as np

from src.visual.state_regions import (frame_timestamps, representative_frame_index,
                                      segment_frame_indices, validate_region_schema)


class VisualStateRegionTests(unittest.TestCase):
    def test_frame_timestamp_extraction(self):
        self.assertEqual(frame_timestamps(4, 1.0, 3.2), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(frame_timestamps(3, 2.0), [0.0, 0.5, 1.0])

    def test_visual_state_region_schema(self):
        region = {"region_id": "region_0000", "start_time": 0.0, "end_time": 2.0,
                  "frame_timestamps": [0.0, 1.0], "frame_indices": [0, 1],
                  "region_pooled_embedding_path": "region_embeddings.npy", "region_embedding_index": 0,
                  "representative_keyframe_path": "keyframes/a.jpg", "representative_frame_timestamp": 1.0,
                  "mean_change_score": 0.1}
        self.assertEqual(validate_region_schema(region), [])

    def test_representative_keyframe_selection(self):
        embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
        self.assertEqual(representative_frame_index(embeddings), 1)
        self.assertEqual(representative_frame_index(embeddings[:2], global_start_index=5), 5)

    def test_very_short_video(self):
        self.assertEqual(segment_frame_indices([0.0], np.empty(0), 0.18, 2.0), [(0, 0)])
        self.assertEqual(segment_frame_indices([], np.empty(0), 0.18, 2.0), [])


if __name__ == "__main__":
    unittest.main()
