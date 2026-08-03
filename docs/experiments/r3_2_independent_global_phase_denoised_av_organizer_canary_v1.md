# R3_2 independent global-phase denoised AV Organizer canary v1

This isolated canary tests an R3_2 semantic navigation map without supplying any
historical phase count, boundary, label, summary, or event conclusion to the
model.

The frozen inputs are the 30 canonical repaired Qwen captions and 107 canonical
timestamped ASR records. The pipeline is:

1. one global call discovers contiguous semantic phases;
2. one local call per discovered phase selects a bounded evidence subset;
3. one batched call writes navigation summaries from selected evidence only;
4. code reconstructs all IDs, boundaries, source mappings, and audio overlap.

Historical nine-phase output is loaded only after the candidate is complete and
is used solely for post-hoc descriptive comparison. It is not an acceptance
target.

The map has no Storyline and cannot hard-filter or score retrieval candidates.
All Medium nodes remain eligible. Raw captions and ASR remain in the map sidecar;
the Planner view exposes only the denoised summary and selected source evidence.
