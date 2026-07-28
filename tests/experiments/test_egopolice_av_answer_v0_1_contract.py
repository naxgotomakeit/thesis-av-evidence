import copy
import unittest

from scripts.experiments.run_egopolice_av_answer_v0_1 import validate_answer


VIDEO_ID = "copa/2023-0001622/822714139"
CRITICAL = [
    "weapon_visible",
    "gunshot_or_firearm_discharge",
    "physical_confrontation",
    "force_or_restraint",
    "arrest_or_handcuffing",
    "injury_or_medical_assistance",
]


def valid_answer():
    return {
        "video_id": VIDEO_ID,
        "incident_summary": "Officers respond to a reported incident.",
        "timeline": [{
            "time_range": "18.5-21.2 sec",
            "event": "Radio traffic references shots.",
            "visual_evidence": "Officers are visible at the scene.",
            "speech_evidence": "ASR records a radio reference to shots.",
            "temporal_relation": "overlap",
            "confidence": "medium",
        }],
        "officer_actions": [],
        "civilian_actions": [],
        "important_speech": [],
        "critical_events": {key: "unclear | unknown | unknown | speech" for key in CRITICAL},
        "uncertainty_or_missing_information": ["ASR is uncertain."],
        "evidence_conflicts": [],
    }


class V0ContractTest(unittest.TestCase):
    def test_valid_v0_contract(self):
        self.assertEqual(validate_answer(valid_answer(), VIDEO_ID), [])

    def test_missing_contract_field_is_rejected(self):
        answer = valid_answer()
        del answer["critical_events"]["weapon_visible"]
        del answer["timeline"][0]["speech_evidence"]
        errors = validate_answer(answer, VIDEO_ID)
        self.assertIn("critical_event_missing_or_not_string:weapon_visible", errors)
        self.assertIn("timeline_0_missing:speech_evidence", errors)

    def test_invalid_critical_status_is_rejected(self):
        answer = copy.deepcopy(valid_answer())
        answer["critical_events"]["force_or_restraint"] = "maybe | 10.0 | unclear evidence | visual"
        self.assertIn("invalid_status:force_or_restraint", validate_answer(answer, VIDEO_ID))


if __name__ == "__main__":
    unittest.main()
