from __future__ import annotations

import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from experiments.shared_question_scope_review_gate_v1.core import (
    build_route_payload,
    canonical_review_scope,
    select_local_fines,
    validate_route_decision,
)


class GateContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "question_id": "q1",
            "question_text": "What happens?",
            "answer_options": [{"option_id": "A", "text": "a"}, {"option_id": "B", "text": "b"}],
        }
        self.assessments = [{"requirement_id": "req_later_activity", "status": "uncertain", "direct_support": False}]
        self.payload = build_route_payload(
            question=self.question,
            assessments=self.assessments,
            candidate_option_ids=["A", "B"],
            evidence_ranges=[{"range": [10, 20], "source": "e1"}],
            navigation_ranges=[{"range": [20, 30], "source": "C02"}],
        )

    def test_answer_now_has_no_review(self) -> None:
        decision = {"question_id": "q1", "decision": "answer_now", "reason": "resolved", "review_request": None}
        self.assertEqual(validate_route_decision(decision, self.payload, video_duration_sec=100), [])

    def test_uncertainty_does_not_force_review_in_payload(self) -> None:
        self.assertTrue(self.payload["policy"]["uncertainty_alone_does_not_trigger_review"])

    def test_review_must_be_local(self) -> None:
        decision = {"question_id": "q1", "decision": "request_local_review", "reason": "x", "review_request": {"review_goal": "x", "target_requirement_ids": ["req_later_activity"], "target_time_range": [0, 60], "target_kind": "dynamic"}}
        self.assertIn("whole-video/broad review rejected", validate_route_decision(decision, self.payload, video_duration_sec=100))

    def test_navigation_range_can_localize_but_is_not_evidence(self) -> None:
        nav = [row for row in self.payload["localization"]["available_local_ranges"] if row["source"] == "C02"]
        self.assertEqual(nav[0]["source_kind"], "navigation_only")

    def test_dynamic_selection_uses_only_local_range(self) -> None:
        decision = {"review_request": {"target_time_range": [10, 30], "target_kind": "dynamic"}}
        fine = [{"fine_id": "F1", "timestamp_sec": 5}, {"fine_id": "F2", "timestamp_sec": 15}, {"fine_id": "F3", "timestamp_sec": 25}, {"fine_id": "F4", "timestamp_sec": 35}]
        self.assertEqual([row["fine_id"] for row in select_local_fines(decision=decision, fine_nodes=fine)], ["F2", "F3"])

    def test_scope_is_canonical(self) -> None:
        self.assertEqual(canonical_review_scope("req_later_activity"), "later_activity")


if __name__ == "__main__":
    unittest.main()
