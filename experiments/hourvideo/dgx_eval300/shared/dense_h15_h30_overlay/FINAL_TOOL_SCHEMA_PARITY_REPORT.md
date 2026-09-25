# Dense-H Tool Schema Parity — Final Report

## Status

`BLOCKED`

The requested code correction and all offline gates passed. The only authorized H-15 live smoke completed successfully and used the correct Dense Beam-B retrieval path, but it recorded one parser/protocol event. Because the acceptance requirements explicitly require no parser error and prohibit expanding the smoke, this version is not marked ready for formal H-15.

## Only runtime-code change

Parent overlay: `dense_semantic_beam_b_live_preflight_v1_20260827T202200Z/runtime_overlay`

New overlay: `dense_semantic_beam_b_tool_schema_parity_v1_20260827T220706Z/runtime_overlay`

Only `videoseal/tools/retrieval_adapter.py` differs. `RETRIEVE_DESCRIPTION` changed from:

```text
Retrieve candidate temporal spans from the video's visual semantic index. Use a natural-language visual query; inspect returned spans before answering.
```

to the exact formal Flat text:

```text
Fine-grained retrieval over the visual (LVBench) semantic index; optionally summarizes top hits and returns useful spans.
```

Source SHA changed from `f2ebf69d95f792f099d30f10bf4fcd32ccf89d9f567e23f6eeaf3535f15975cf` to `008efa173913759a7713209b649ab3947e99349a2385afd017995778460fc4b0`. Recursive parent/new overlay comparison found no other difference.

## Offline validation

- Complete Flat/H tools schema field equality: PASS.
- Complete serialized schema byte equality: PASS.
- Formal Flat schema SHA: `71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863`.
- Corrected H schema SHA: `71346effd5acd8d567e16921316e0f6cb0e94b422f42c4d55edb24a86d4a1863`.
- Existing Dense Beam-B offline contract: 13/13 PASS.
- Backend remains `dense_semantic_beam_b`; B configuration and query passing are unchanged.
- Flat caption-segment return format, H-30 Medium gate, telemetry, hard-failure behavior, and the prohibition on lexical/BM25/global-SigLIP/silent fallback remain covered by the unchanged contract suite.

## Single H-15 live smoke

- UID: `115774b6-534d-444f-b7aa-d1b834eb0ee7_17_7` (previously used smoke fixture).
- Attempt: `20260827T221425Z-ljl8`.
- B=15; concurrency=1; native 8B Planner; corrected V7.4 plus the new overlay.
- Trajectory schema SHA equals formal Flat: PASS.
- Dense Beam-B backend and hierarchy complete: PASS on both retrieval calls.
- Selected Coarse/Medium/segments are each 15: PASS.
- One query embedding and one SigLIP query encoding per retrieval: PASS (`[1,1]` for both counters).
- Flat caption-segment schema and Medium gate: PASS.
- Global Fine, caption-score, representative-frame and silent fallback flags: all false.
- Timeout/provider/asset/backend errors: none.
- Final answer: `C`, completed; ground truth `C`.
- Parser/protocol gate: FAIL. After the first Inspector result, Planner emitted `A. White` without a `<final>` wrapper. Runtime recorded `invalid response; reminded model to use a tool`, recovered, and eventually completed. This is one parser/protocol event and therefore fails the requested zero-error acceptance gate.

No second smoke was attempted.

## Shutdown

Planner and Inspector were terminated by the smoke script cleanup trap. Ports 18082 and 18083 are released. Formal Eval300 was not started.

## Flat fallback lineage

`FLAT_RESULTS_UNAFFECTED`; `FLAT_AFFECTED_UIDS=[]`; no Flat rerun is required.
