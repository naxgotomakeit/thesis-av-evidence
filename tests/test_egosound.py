import json
import unittest
from pathlib import Path

from src.data.egosound import load_annotations, match_media, parse_provided_timestamp, validate_manifest_row, video_id_from_annotation
from scripts.select_dev_cases_10 import select_cases


class EgoSoundTests(unittest.TestCase):
    def test_annotation_parsing(self):
        path = Path(".test_annotations.json")
        try:
            path.write_text(json.dumps([{"question_id": "00001_1", "question": "Q"}]), encoding="utf-8")
            self.assertEqual(load_annotations(path)[0]["question_id"], "00001_1")
        finally:
            path.unlink(missing_ok=True)

    def test_id_and_path_matching(self):
        entry = {"video_path": "Ego4d/videos/00001.mp4", "question_id": "00001_3"}
        video, audio = Path("v/00001.mp4"), Path("a/00001.wav")
        self.assertEqual(video_id_from_annotation(entry), "00001")
        self.assertEqual(match_media(entry, {"00001": video}, {"00001": audio})[:3], ("00001", video, audio))

    def test_missing_file_handling(self):
        result = match_media({"question_id": "00002_1"}, {}, {})
        self.assertIn("missing_video", result[3])
        self.assertIn("missing_external_audio", result[3])

    def test_manifest_schema(self):
        row = {k: None for k in ("case_id", "source_subset", "video_id", "video_path", "audio_path", "question", "answer", "question_type", "provided_timestamp_start", "provided_timestamp_end", "provided_context", "evidence_source", "video_duration", "audio_duration", "video_has_embedded_audio")}
        row.update(media_valid=True, warnings=[])
        self.assertEqual(validate_manifest_row(row), [])

    def test_provided_timestamp_parsing(self):
        self.assertEqual(parse_provided_timestamp("01:02 - 01:05"), (62.0, 65.0, []))
        self.assertEqual(parse_provided_timestamp(None), (None, None, []))

    def test_small_dev_selection(self):
        pool = []
        for index in range(12):
            pool.append({
                "case_id": f"{index:05d}_1", "video_id": f"{index:05d}",
                "video_path": f"videos/{index:05d}.mp4", "audio_path": f"audios/{index:05d}.wav",
                "video_duration": 10.0, "audio_duration": 10.0, "media_valid": True,
                "question": "What sound was heard?", "answer": "A clink.",
                "question_type": "Sound Source Identification", "signal_active_fraction": 1.0,
                "evidence_start": None, "evidence_end": None,
            })
        selected = select_cases(pool, 10)
        self.assertEqual(len(selected), 10)
        self.assertEqual(len({x["video_id"] for x in selected}), 10)
        self.assertTrue(all(x["needs_manual_evidence_annotation"] for x in selected))
        self.assertTrue(all(x["evidence_start"] is None and x["evidence_end"] is None for x in selected))


if __name__ == "__main__":
    unittest.main()
