# New Dense Semantic Beam-B H-8 Superseding Amendment v3

Status: `PASS — B=8 OUTER ADAPTER ENTRY GAP FIXED`

## Parent proof

- Formal new H-15: `outputs/dense_semantic_beam_b_h15_eval300_formal_v1_20260827T224330Z`
- Formal new H-30: `outputs/dense_semantic_beam_b_h30_eval300_formal_v2_20260829T155639Z`
- Backend: `dense_semantic_beam_b`
- Dense Retriever SHA: `47b8dcfea459f9edacf650e9975043714635c8c3b5fd054f78715535f99cd46e`
- Tools schema SHA: `71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863`
- Old lexical H used: false
- Old global Fine SigLIP H used: false

## Failure diagnosis and sole behavior change

The v2 formal process imported the intended H-8 v2 overlay first on `PYTHONPATH`; its actual `retrieval_adapter.py` SHA was `008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0`, and lines 364–365 retained `(6,15,30)`. The H-8 config and Retriever accepted `[8,15,30]`, but the outer adapter rejected B=8 before Planner or retrieval execution.

The v2 300 first-pass and 300 retry metrics are infrastructure diagnostics only, with zero Planner/retrieval/Inspector/prediction/trajectory activity. They are excluded from resume, retry selection, and formal statistics.

```diff
- if self.b not in (6, 15, 30):
-     raise RuntimeError(f"DENSE_BEAM_B must be 6, 15, or 30; got {self.b}")
+ if self.b not in (8, 15, 30):
+     raise RuntimeError(f"DENSE_BEAM_B must be 8, 15, or 30; got {self.b}")
```

Old adapter SHA: `008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0`.

New adapter SHA: `477e331b0636ced674a3db7861181d7d3038eb55027213e66a9b818b05ff8264`.

All other runtime files are byte-identical to H-8 v2. Actual-import trace, B=8/15/30 construction, B=6/7 rejection, unchanged tools schema, and 13/13 Dense Retriever contracts pass.
