import math
import unittest
from unittest.mock import patch

from videoseal.agents.tool_agent import _full_video_fallback_request
from videoseal.utils.video.time import srt_timestamp_to_seconds
from videoseal.utils.video.tooling import normalize_window_min_width


class FullVideoFallbackWindowTest(unittest.TestCase):
    def _assert_full_video_request(self, duration: float) -> None:
        span, fps = _full_video_fallback_request(duration, 64)
        start = srt_timestamp_to_seconds(span["start_time"])
        end = srt_timestamp_to_seconds(span["end_time"])

        self.assertEqual(start, 0.0)
        self.assertLessEqual(end, duration)
        self.assertLess(duration - end, 1e-6)

        frame_count = min(int((end - start) * fps), 64)
        self.assertEqual(frame_count, 64)
        timestamps = [start + i / fps for i in range(frame_count)]
        self.assertEqual(len(timestamps), 64)
        self.assertTrue(all(timestamps[i] < timestamps[i + 1] for i in range(63)))
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(63)]
        self.assertLess(max(intervals) - min(intervals), 1e-9)
        self.assertLess(timestamps[-1], duration)

    def test_integer_duration_keeps_full_video_and_64_uniform_timestamps(self) -> None:
        self._assert_full_video_request(1800.0)

    def test_fractional_duration_keeps_full_video_and_64_uniform_timestamps(self) -> None:
        self._assert_full_video_request(4396.83)

    def test_ordinary_15_second_tail_clamp_is_unchanged(self) -> None:
        with patch("videoseal.utils.video.tooling.get_video_duration", return_value=100.25):
            start, end = normalize_window_min_width("unused.mp4", 99.0, 101.0)
        self.assertTrue(math.isclose(start, 85.25))
        self.assertTrue(math.isclose(end, 100.25))
        self.assertTrue(math.isclose(end - start, 15.0))


if __name__ == "__main__":
    unittest.main()
