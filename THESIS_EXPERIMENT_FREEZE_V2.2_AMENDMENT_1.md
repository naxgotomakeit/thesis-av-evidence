# Protocol Amendment #1 — Answer-model inference dtype

**Date:** 2026-07-23  
**Base protocol:** `THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md`

## Change

For all subsequent formal E1/E2+ answer-model inference, use
Qwen2.5-VL-7B-Instruct through HuggingFace in **FP16**, unquantized, replacing
the prior BF16 formal candidate.

## Reason and scope

The target Quadro RTX 6000 / Turing sm_75 runs FP16 substantially faster.
Matched Open validation showed numerical stability but some output divergence;
FP16 is not claimed to be behavior-identical to BF16.  The Closed-50 paired
gate found identical U8/Oracle accuracy and identical Oracle-U8 headroom, with
no systematic harmful direction.

All formal E1/E2+ comparisons use the same HF FP16 configuration. No formal
statistic may mix BF16 and FP16 outputs. Previous BF16 outputs are preserved as
exploratory/reference-only artifacts and excluded from formal statistics.

## Unchanged

- Qwen2.5-VL-7B-Instruct checkpoint and HuggingFace backend
- unquantized model; `max_pixels=262144`
- prompts, canonical clip scope, frame rules/budgets, greedy decoding, manifests
- B0/B1/B2 and E0-E6 definitions

The active protocol is V2.2 FINAL plus this additive amendment.
