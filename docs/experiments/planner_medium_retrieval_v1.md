# Planner + Medium Retrieval v1

This is a question-conditioned diagnostic over the read-only
`structured_organizer_v1/226/hierarchical_index_v1.json`.

It runs one Claude Haiku Planner call per diagnostic question, validates the plan against
the index capabilities, routes through storyline/Coarse nodes, and ranks Medium nodes using
frozen SigLIP and caption signals. It does not rerank Fine frames and does not generate an
answer.

## Frozen configuration

- Planner: `claude-haiku-4-5-20251001`, temperature 0, max output 1800 tokens.
- At most one JSON/contract repair call per question.
- SigLIP: `google/siglip-base-patch16-224`, 768-dimensional normalized embeddings.
- Score: `0.60 * normalized_visual + 0.30 * lexical + 0.10 * coarse_prior`.
- Targeted: 2–4 Coarse nodes, at most 8 Medium nodes.
- Global coverage: one Medium per every Coarse node.
- Multi-target compare: independent unit rankings, at most 4 Medium nodes per unit and 10
  after deterministic merge/deduplication.

Every score component, matched lexical term, routing decision, prompt, raw response, parsed
plan, usage record, and frozen input hash is retained under
`outputs/experiments/planner_medium_retrieval_v1/226/`.

## Run

```powershell
conda run -n thesis_av python scripts/experiments/run_planner_medium_retrieval_v1.py
```

The API key is read from `ANTHROPIC_API_KEY`; it is never serialized.
