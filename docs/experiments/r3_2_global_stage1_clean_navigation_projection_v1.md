# R3_2 global Stage-1 clean navigation projection v1

This no-API projection uses only the independently generated and validated global
Stage 1 result from the exact-interleaved frozen-method canary. Stage 1 saw all
current canonical repaired captions and ASR, but no historical phase output.

The Planner-visible navigation summary is the global phase label. Boundary reasons,
raw captions, raw ASR, and failed local Stage 2 text remain in audit sidecars and
are not exposed to the Planner. This preserves global semantic navigation without
allowing local caption noise to re-enter the map.
