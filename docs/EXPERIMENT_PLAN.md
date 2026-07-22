# EgoPolice experiment plan

## Formal controlled ablation

The main ablation uses the frozen 20-video, 98-question paired subset in
`config/data/egopolice_ablation20_v1.json` and
`config/data/egopolice_ablation_questions_v1.json`. Every stage must evaluate
the same questions. The parent frozen 50-video pool is reserved for later
final-scale validation and must not be used for threshold tuning.

The stages are incremental and nested:

- **B0:** full source video → eight uniform midpoint-bin frames → one Qwen call.
- **B1:** reusable visual Medium map → simple flat Top-K visual retrieval → the
  same final Qwen model.
- **B2-V / B2-AV:** controlled visual-only versus fixed audiovisual retrieval;
  B2 adds audio evidence without Planner routing.
- **B3:** the same hierarchical audiovisual map with simple fixed routing.
- **B4:** the same map and evidence budget with Planner routing.

The purpose is to add one mechanism at a time and measure accuracy, temporal
grounding diagnostics, online latency, offline preprocessing, model calls,
model-facing frames/tokens, peak GPU memory, index size, and amortized cost.

Controlled variables include the paired question pool, Qwen2.5-VL-7B final
answer model, MCQ prompt semantics, decoding policy, evaluation protocol, and a
maximum final visual evidence budget of **no more than eight frames**. Any
stage-specific extra calls or reusable preprocessing must be logged rather than
hidden.

The short-clip Oracle is a diagnostic upper-context condition: it supplies the
annotated short interval directly to Qwen. It is not B0 and is not a deployable
long-video baseline.

## Temporal exposure terminology

**GT Interval Hit@8** means that at least one already-selected B0 timestamp lies
inside the annotated ground-truth interval. It is a coarse temporal exposure
diagnostic, not true evidence recall. An interval can contain only a brief
informative action, so a sampled frame inside a 10-second interval may still
miss the decisive visual event. GT intervals must never influence B0 sampling.

The original B0-5 smoke JSON is preserved byte-for-byte and therefore retains
the legacy field name `gt_hit_at_8`; in that artifact it has exactly the GT
Interval Hit@8 meaning above. Future diagnostic schema uses
`gt_interval_hit_at_8`.
