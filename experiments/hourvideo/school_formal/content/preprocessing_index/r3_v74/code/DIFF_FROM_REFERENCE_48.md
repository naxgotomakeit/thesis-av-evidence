# Source delta from the verified 48-file reference runtime

Reference bundle SHA-256:
`ffcb174919b40dc3d278ce325e054c83716353c61de1c496600fbd960a51fe84`.

Only the following reference-runtime paths are overlaid:

- `videoseal/tools/tool_map.py`: selects the frozen flat or hierarchical
  implementation behind the unchanged `visual_retrieve(query)` schema.
- `videoseal/tools/visual_tools.py`: adds structured audit metadata for flat raw
  candidates and summarizer-selected spans. It does not change retrieval order,
  summarizer calls, tool output, or Planner-visible content.
- `videoseal/tools/retrieval_adapter.py`: new backend seam. Flat delegates to the
  reference embedding retriever. Hierarchical delegates to the existing
  Coarse→Medium→Fine implementation, applies the same reference summarizer and
  stores traversal/provenance only in metadata.

The inherited hierarchical implementation and its minimal import dependencies
are under `runtime_support/src/experiments/`. They preserve lexical Coarse hard
gating, Medium scoring `0.6 * min-max(SigLIP) + 0.3 * lexical`, and fixed
top-3 Coarse → top-3 Medium → two Fine per Medium selection.

No reference Planner, Inspector, prompt, parser, runner, sampler, timeout,
retry, step limit, termination, or fallback file is replaced. `config.json` is
the audited freeze record; data/index/model paths must be resolved at deployment
without changing the experiment values.
