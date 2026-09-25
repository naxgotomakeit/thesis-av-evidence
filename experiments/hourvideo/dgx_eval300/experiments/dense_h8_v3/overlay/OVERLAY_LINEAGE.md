# H-8 Budget Overlay v3 Lineage

Algorithmic parent: `dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z/runtime_overlay`, used by formal new Dense Semantic Beam-B H-15/H-30.

Immediate parent: `dense_semantic_beam_b_h8_budget_overlay_v2_20260829T155639Z/runtime_overlay`.

The immediate parent already contains `allowed_b=[8,15,30]` and the matching provider config SHA. The sole runtime-source change in v3 fixes the missed outer entry validation in `videoseal/tools/retrieval_adapter.py`:

```text
(6, 15, 30) -> (8, 15, 30)
```

The matching error text changes accordingly. This permits preregistered B=8 to reach the unchanged Retriever; it does not modify scoring, ranking, evidence materialization, prompts, schemas, models, timeout, parser, retry, fallback, or scoring.

Dense Retriever SHA remains `47b8dcfea459f9edacf650e9975043714635c8c3b5fd054f78715535f99cd46e`.
