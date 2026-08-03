# R1_AV / R3_2 cached visual-review full closed loop v1

This isolated experiment connects the immutable reviewed-visual cache to the
current requirement-scoped R1_AV/R3_2 handoff and the frozen text-only Final
Gemini contract.

The experiment replays Planner, Retrieval, Sufficiency and question-scope Gate
artifacts read-only. Every previously reviewed image/scope key must be a cache
hit. A missing key fails before Final Gemini; this version never silently
re-reviews an image. Final Gemini receives resolved requirement assessments,
not raw images or map text.

The intended online behavior is therefore:

1. look up reviewed visual facts by image hash, review-contract version and
   fact scope;
2. reuse immutable findings and update only the target requirement;
3. preserve accepted temporal handoff for the R3_2 temporal diagnostic;
4. make one text-only batched Final Gemini call per representation.

This is a video-226 integration diagnostic. It does not freeze correctness,
HourVideo performance or dataset-level behavior.

At the user-facing serialization boundary, Unicode dash/minus characters are
normalized to the ASCII hyphen. This avoids Windows terminal/rendering
mojibake while preserving the immutable raw Gemini response and all source
artifacts. The normalization is punctuation-only and produces an explicit
audit.
