# R1 / R3_1 / R3_2 map-aware all-Medium retrieval pair v1

This isolated six-question experiment compares the frozen R1 structural map, the R3_1 noise-retaining AV map, and the manually accepted R3_2 global clean-phase map under one Planner and retrieval contract.

Planner map suggestions are soft hints only. Every path ranks all 30 Mediums with zero Coarse prior and no Fine reranking. R3_1 and R3_2 use the same exact canonical-caption lexical corpus, so their ranking differences arise from map-conditioned query formulation rather than different stored Medium text.

The experiment reports Planner input/output tokens and latency separately. It does not run Organizer, Sufficiency, temporal review, Final Gemini, QA, or HourVideo.
