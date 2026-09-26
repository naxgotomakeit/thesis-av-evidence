# A/B/D eval300 preparation draft

This directory is review material only. It is not a smoke run, formal preflight, frozen prompt, or formal experiment.

Deliverables:

- `ORIGINAL_GENS_V3_PROMPT_AND_CONTRACT.md`: verbatim original prompt, schema, settings, parser, and SHA anchors.
- `COMMON_ABD_PROMPT_AND_EXACT_DIFF.md` and `.json`: complete shared prompt, unchanged user template, exact historical diff, and applied ABD-draft payload contract.
- `OUTPUT_CONTRACT_COMPARISON.md`: Direct/GenS acceptance and scoring comparison.
- `INPUT_CONSISTENCY_REPORT.md` and `.json`: 300/300/300 input and SHA checks.
- `REPRESENTATIVE_REQUEST_PREVIEWS.json`: actual provider-shaped model-visible previews (JPEG bytes only replaced with a validated placeholder), clearly separated from backend provenance metadata.
- `COST_ESTIMATE.md`, `.json`, and per-question `.csv`: offline planning estimate.
- `inputs/A.jsonl`, `B.jsonl`, `D.jsonl`: no answer, reason, provider response, or gold fields.
- `MODIFIED_FILES_THIS_ROUND.md`: source/config and regenerated-deliverable inventory.
- `formal_candidate/`: guarded runtime manifest, offline preflight report, per-task capacity estimates, and recovery/freeze notes.

Comparison boundary: new B/D contain explicit resolved-time labels; historical GenS did not. Therefore the new wrapper is not claimed to be identical to historical GenS. Within ABD, D minus its map block is model-visible identical to B.

Current preparation status: PASS. Primary 900-task base estimate: $22.452 (range $21.023–$24.145).

Real API calls: **0**. Smoke launches: **0**. Formal preflight launches: **0**. Formal experiment launches: **0**.
