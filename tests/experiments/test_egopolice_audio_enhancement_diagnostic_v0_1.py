import json
import tempfile
import unittest
from pathlib import Path

from src.experiments.egopolice_audio_enhancement_diagnostic.core import needs_fallback, safe_id, target_scores


class AudioEnhancementDiagnosticTest(unittest.TestCase):
    def test_safe_id(self):
        self.assertEqual(safe_id('copa/2021-1076/540772226'), 'copa__2021-1076__540772226')

    def test_fallback_for_empty_and_low_confidence(self):
        self.assertTrue(needs_fallback({'raw_transcript': '', 'detected_language': 'en', 'warnings': []}))
        self.assertTrue(needs_fallback({'raw_transcript': 'hello', 'detected_language': 'en', 'avg_logprob': -1.2, 'warnings': []}))

    def test_target_scores_keep_alternatives(self):
        result = target_scores(['Gunshot', 'Explosion', 'Impact'], [0.8, 0.4, 0.3], {'gunshot_like_candidate':['gunshot'], 'explosion':['explosion'], 'impact':['impact']})
        self.assertEqual(result, {'gunshot_like_candidate': 0.8, 'explosion': 0.4, 'impact': 0.3})


if __name__ == '__main__':
    unittest.main()
