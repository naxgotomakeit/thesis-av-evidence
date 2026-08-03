# R1_AV / R3_2 claim-guided Gemini closed loop v3

This isolated smoke keeps the frozen upstream R1_AV and R3_2 maps, Planner,
Retrieval, Sufficiency and reliability outputs unchanged. It replaces only the
downstream Gemini orchestration with the validated claim-guided pattern:

1. text-only routing per question;
2. at most one local visual review when an answer-critical unresolved claim is
   concretely localizable;
3. no fixed image-count cap and no full-video scan;
4. one text-only final batch per representation.

Local image selection reuses only canonical Fine frames and already-existing
exact-timestamp follow-ups. It performs no new video decoding. The final stage
cannot re-run retrieval or alter deterministic answerability.
