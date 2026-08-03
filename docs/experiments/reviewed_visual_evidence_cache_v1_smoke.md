# Reviewed visual evidence cache v1 smoke

This isolated smoke implements deterministic cache lookup, immutable record
storage, hit/partial-hit/miss routing, a fixed reviewed-visual record contract,
target-only requirement updates, and deterministic gate-status recomputation.

It does not connect to R1_AV or R3_2, call Gemini, or implement evidence-conflict
resolution. A synthetic record exercises the miss-to-write-to-hit path without
claiming a real image observation.
