# shared_sufficiency_v3_2_1_temporal_anchor_canary_v1

This isolated canary adds typed temporal anchors only to V3.2 requirements that ask for chronological or physical event order. Non-temporal claims retain the exact V3.2 fields and validator behavior.

Supported `before`/`after` requires two packet-backed `event_occurrence` anchors, exact source intervals, and a mathematically valid non-overlapping relation. State confirmation, plans/commands, and unclear mentions remain insufficient. Invalid model output is preserved and rejected rather than repaired.

The live scope is exactly two calls: R1 and Dense/R3 `q_handcuff_before_medical`. Planner, Retrieval, packet construction, Gemini, visual review, final QA, and the full 6+6 regression are out of scope.
