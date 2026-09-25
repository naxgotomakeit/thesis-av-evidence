# Dense Semantic Beam-B H-15 — Final Acceptance Amendment

## Status

`PASS — TOOL SCHEMA PARITY RESTORED`

`READY_FOR_FORMAL_H15`

```text
Recovered protocol events: 1
Unrecovered protocol failures: 0
```

This amendment supersedes only the acceptance interpretation of the single H-15 smoke recovery event. It does not modify, delete, or overwrite the earlier `BLOCKED` report or any frozen runtime, protocol, Retriever, Embedding, Flat result, or readiness artifact.

## Reclassified event

UID `115774b6-534d-444f-b7aa-d1b834eb0ee7_17_7`, attempt `20260827T221425Z-ljl8`, is classified as:

`SEMANTIC_REPLAN_WITH_NEW_INSPECTION`

- The first plain-text `A. White` was not a legal `<final>...</final>` response and is not a prediction.
- Runtime applied the same frozen Parser and tool-required recovery protocol used by formal Flat.
- Planner then issued one additional `visual_retrieve` and two additional `visual_inspect` calls.
- Both new Inspector outputs returned `C. Yellow` with confidence `1.00`.
- The final legal response was `<final>C</final>`.
- All retrieval, Inspector, recovery and elapsed costs from the attempt remain part of its measured execution cost.
- The event is not Dense-H-specific and does not impair Flat/H fairness.
- No additional smoke and no Flat rerun are required.

## Accepted runtime

The accepted child runtime is the immutable overlay at:

`../dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z/runtime_overlay`

The accepted `retrieval_adapter.py` SHA-256 is:

`008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0`

Formal H-15 must use `RETRIEVAL_BACKEND=dense_semantic_beam_b`, `DENSE_BEAM_B=15`, the frozen native Qwen3-8B Planner, frozen Qwen2.5-VL-7B Inspector, and concurrency 1.
