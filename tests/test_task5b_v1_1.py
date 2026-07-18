import inspect
import unittest

from scripts import correct_task5b_v1_1
from src.retrieval.task5b import canonical_visual_frames, package_micro_frames


def micro_candidate(start, end, candidate_id="micro_a"):
    return {
        "candidate_id": candidate_id,
        "candidate_type": "micro_window",
        "start_time": start,
        "end_time": end,
        "frame_paths": ["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"],
        "warnings": [],
    }


SOURCE = {
    "frame_paths": ["f0.jpg", "f1.jpg", "f2.jpg", "f3.jpg"],
    "frame_timestamps": [0, 1, 2, 3],
    "frame_indices": [0, 1, 2, 3],
}


class Task5BV11Tests(unittest.TestCase):
    def test_a_half_open_start_interval(self):
        packaged = package_micro_frames(micro_candidate(0, 2), SOURCE)
        self.assertEqual(packaged["selected_frame_timestamps"], [0.0, 1.0])
        self.assertEqual(packaged["frame_paths"], packaged["selected_frame_paths"])
        self.assertEqual(packaged["source_frame_paths"], SOURCE["frame_paths"])

    def test_b_half_open_later_interval(self):
        packaged = package_micro_frames(micro_candidate(2, 4), SOURCE)
        self.assertEqual(packaged["selected_frame_timestamps"], [2.0, 3.0])

    def test_c_overlapping_micro_windows_are_deduplicated(self):
        left = package_micro_frames(micro_candidate(0, 3, "left"), SOURCE)
        right = package_micro_frames(micro_candidate(1, 4, "right"), SOURCE)
        merged = canonical_visual_frames([left, right], [], video_id="video_a")
        self.assertEqual([x["timestamp"] for x in merged], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(merged), 4)
        self.assertEqual({x["video_id"] for x in merged}, {"video_a"})

    def test_d_dense_frame_is_canonical_on_duplicate_timestamp(self):
        micro = package_micro_frames(micro_candidate(0, 2), SOURCE)
        merged = canonical_visual_frames([micro], [{"dense_frame_id": "d0", "timestamp": 1.0, "frame_path": "dense_1.jpg"}], video_id="video_a")
        one = next(x for x in merged if x["timestamp"] == 1.0)
        self.assertEqual(one["canonical_frame_path"], "dense_1.jpg")
        self.assertEqual(one["provenance"], "both")
        self.assertEqual(len(merged), 2)

    def test_deduplication_key_includes_video_id(self):
        left = package_micro_frames(micro_candidate(0, 1, "left"), SOURCE)
        right = package_micro_frames(micro_candidate(0, 1, "right"), SOURCE)
        left["video_id"] = "video_a"
        right["video_id"] = "video_b"
        self.assertEqual(len(canonical_visual_frames([left, right], [])), 2)

    def test_e_exact_end_frame_is_excluded(self):
        packaged = package_micro_frames(micro_candidate(0, 2), SOURCE)
        self.assertNotIn(2.0, packaged["selected_frame_timestamps"])

    def test_f_no_case_or_answer_hardcoding_in_correction(self):
        source = inspect.getsource(correct_task5b_v1_1)
        self.assertNotRegex(source, r"000\d")
        self.assertNotIn("provided_context", source)
        self.assertNotIn('["answer"]', source)

    def test_g_correction_has_zero_llm_calls(self):
        source = inspect.getsource(correct_task5b_v1_1)
        self.assertNotIn("anthropic", source)
        self.assertNotIn("messages.create", source)
        self.assertIn('"task5b_llm_api_calls":0', source)

    def test_h_correction_does_not_access_reference_timestamps(self):
        source = inspect.getsource(correct_task5b_v1_1)
        self.assertNotIn("provided_timestamp", source)
        self.assertIn("posthoc_reference_evaluation", source)

    def test_timestamp_fallback_requires_declared_fps(self):
        source = {"frame_paths": ["a.jpg", "b.jpg"], "frame_indices": [0, 1]}
        unknown = package_micro_frames(micro_candidate(0, 2), source)
        self.assertEqual(unknown["selected_frame_paths"], [])
        self.assertIn("unable_to_resolve_micro_frame_timestamps", unknown["warnings"])
        resolved = package_micro_frames(micro_candidate(0, 1), source, fps=2)
        self.assertEqual(resolved["selected_frame_timestamps"], [0.0, 0.5])


if __name__ == "__main__":
    unittest.main()
