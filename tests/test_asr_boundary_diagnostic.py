import inspect
import unittest

from scripts import diagnose_task5c_asr_boundary as diagnostic


class ASRBoundaryDiagnosticTests(unittest.TestCase):
    def test_extracts_only_named_fallback_object(self):
        row = '{"case_id":"x","fallback":{"segments":[{"text":"ok"}]},"posthoc_weak_reference_evaluation":{"secret":"unused"}}'
        self.assertEqual(diagnostic.json_object_after_key(row, '"fallback"'), {"segments": [{"text": "ok"}]})

    def test_vad_gaps_are_deterministic(self):
        regions = [{"start_time": 2.0, "end_time": 3.0}, {"start_time": 5.0, "end_time": 6.0}]
        self.assertEqual(diagnostic.vad_input_gaps(regions, 1.0, 7.0), [{"start_time": 1.0, "end_time": 2.0}, {"start_time": 3.0, "end_time": 5.0}, {"start_time": 6.0, "end_time": 7.0}])

    def test_diagnostic_does_not_load_reference_or_answer_fields(self):
        source = inspect.getsource(diagnostic)
        self.assertNotIn("provided_timestamp", source)
        self.assertNotIn('["answer"]', source)
        self.assertIn("production_pipeline_modified", source)


if __name__ == "__main__":
    unittest.main()
