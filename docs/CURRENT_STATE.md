# Current project state

- Branch: `exp/dull-baseline-qwen7b`
- Controlled model: `Qwen/Qwen2.5-VL-7B-Instruct`
- Local model path:
  `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/Qwen2.5-VL-7B-Instruct`
- Canonical EgoPolice data root:
  `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0`
- Repository output root: `outputs/`

The frozen visual pipeline and `config/data/egopolice_50videos.json` are
unchanged. B0 remains: full source video → eight equal temporal bins → exact
midpoint frame from each bin → question plus five options and eight frames →
exactly one Qwen call. B0 does not use GT timestamps, short clips, retrieval,
audio, hierarchy, Planner, checking, or fallback.

Completed diagnostics:

- Oracle-5: 3/5, BF16 unquantized, one load/five calls.
- B0-5: 0/5; GT Interval Hit@8 was 2/5; BF16 unquantized, one
  load/five calls.

Formal set freeze completed on 2026-07-22:

- `config/data/egopolice_ablation20_v1.json`: 20 videos selected without model
  results from the parent frozen 50, excluding `pasadena/bMMuC`.
- `config/data/egopolice_ablation_questions_v1.json`: 98 paired questions over
  all 20 videos.

No formal B0 run, B1 work, or new video download was started during the freeze.
The next execution task is not authorized in this handoff; do not run formal B0
or begin B1 without an explicit instruction.
