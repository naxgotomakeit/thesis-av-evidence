# Fine Reranking v1

This offline experiment consumes the frozen `structured_organizer_v1` index and the existing
`planner_medium_retrieval_v1` outputs for video 226.

For every selected Medium, it expands existing child Fine nodes, encodes each frozen Planner
search-unit query with `google/siglip-base-patch16-224`, ranks the existing 768-dimensional
Fine representative-frame embeddings, and selects at most two temporally diverse Fine nodes.

It does not call Haiku, rerun Medium retrieval, regenerate embeddings or captions, sample new
video frames, judge sufficiency, or generate answers.

Default selection configuration:

- at most two Fine nodes per selected Medium;
- second representative timestamp at least two seconds away, or a distinct non-identical
  Fine interval;
- a single Fine is valid when no diverse second exists.

The unified handoff for the next stage is
`outputs/experiments/fine_reranking_v1/226/evidence_candidates.json`.
