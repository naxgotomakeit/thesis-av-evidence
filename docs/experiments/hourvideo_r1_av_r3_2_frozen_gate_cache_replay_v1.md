# HourVideo R1_AV/R3_2 frozen Gate/cache replay v1

This isolated replay reconnects the frozen question-scope visual-review policy
to the first HourVideo single-video smoke. It reuses the saved maps, Planner
outputs, retrieval packets and first-pass Sufficiency assessments. It does not
rerun index construction, Organizer, Planner, retrieval or initial Sufficiency.

The frozen routing prompt decides `answer_now` versus a single targeted local
review. Uncertainty alone is not a review trigger. A local review is permitted
only for an answer-critical unresolved requirement with a concrete range. Every
image lookup uses the immutable `(image_sha256, review_contract_version,
fact_scope)` cache key before a Gemini visual call. Final Gemini remains
text-only.
