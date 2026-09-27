# Historical thesis protocol V2.2

English presentation summary, not a replacement experimental record. The linked frozen originals remain authoritative for byte-level provenance. Numerical values and artifact identities reproduced here are copied without alteration; this summary does not introduce new results or revise frozen labels. See the [source register](../ENGLISH_READING_GUIDE.md) for original-file hashes and checksum membership.

The [original protocol](../../THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md) was revised on 2026-07-23. Its original ACTIVE status describes that planning stage, not the authority of every final dissertation result. It specifies planned experiments and controls; it does not prove that every planned experiment was completed. Later formal QaEgo4D records are linked in the [thesis index](../../THESIS_ARTIFACT_INDEX.md#qaego4d-formal-frozen-experiments).

## Method definitions

B0 samples uniform-8 frames without a reusable index. B1 builds a flat event map and uses fixed retrieval to select at most eight representative frames. B2 derives Medium events through Fine → Safe-Merge → Fluid Loose and applies the same retrieval policy and model-facing budget. The intended contrast is the hierarchical event-derivation package, not hierarchy isolated from all other effects.

E1/E2 use the official QaEgo4D canonical clip. Mapping evidence timestamps to a parent video does not establish that the original question remains well-posed throughout that video. Parent-window evaluation requires an explicit contamination audit.

Option A uses events with one representative frame per retrieved event. B1/B2 must share backbone, scorer, aggregation, Top-K policy and representative-frame rule. Event count, mean duration and duration standard deviation must be reported. The protocol requires an event-count-matched control if counts differ by more than ±25%. Flat segmentation and hierarchy parameters must be frozen before evaluation.

## Planned experiments

| ID | Purpose and boundary |
|---|---|
| E0 | Audit question validity under parent-video context expansion; define a common clean subset. |
| E1 | Test answer-model headroom with Blind, Uniform-8, Uniform-32 and Oracle conditions. Closed Blind receives question/options; Open Blind receives question only. |
| E2 | Compare B0/B1/B2 on the same frozen QaEgo4D population using paired tests. |
| E3 | Evaluate 2/5/10/20/40-minute parent windows on a common audited subset, retaining the same evidence-position regime across lengths. |
| E4a | Derive canonical-clip amortization from E2 logs, with no independent evaluation subset or extra inference. |
| E4b | Measure parent-video deployment amortization using actual parent-level index cost and separately validated quality where representation changes. |
| E5 | Explore audio and Planner branches after appropriate dataset-specific B0/B2 anchors. |
| E6 | Compare against external frame-selection methods on a separately selected comparable benchmark. |

Point evidence (`start == end`) is provisionally retained rather than automatically discarded; instantaneous evidence semantics and tolerance remain protocol checks. Answer generation and repeated LLM judging have separate frozen configurations. The protocol distinguishes statistical power/MDE from observed gain and requires paired analysis.

## Cost, data and governance

Offline preprocessing and per-query costs are logged separately. Include decode/seek/I/O for B0 and any online frame access by B1/B2. Report GPU time, wall time, money, tokens, calls and storage separately; do not add incompatible units. Clip-level logs cannot establish parent-level index-once cost.

Three manifests distinguish Open, Closed and the E3 common subset. The candidate answer model is frozen after the headroom gate. The small engineering model is excluded from formal results. Dataset audits cover annotation semantics, well-posedness, UID mapping, question types, length coverage and headroom.

The early EgoPolice full-source variant had temporal ambiguity and distractor contamination risks; its diagnostic results must not be compared directly with the official clip protocol. A separately constructed body-camera case study was proposed. The protocol also records retired numbering schemes, unresolved segmentation/position/tolerance choices, benchmark attribution and summarization-evaluation design. These historical open items are not assertions about the final repository.
