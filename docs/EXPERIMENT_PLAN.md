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

## Frozen evaluation dimensions

Every formal B0–B4 result must use two separate grouping dimensions:

1. Question/annotation duration class: `1s`, `10s`, `60s`.
2. Full source-video duration class, computed only from validated ffprobe media
   duration:
   - `le_300`: duration <= 300 seconds
   - `301_600`: 300 < duration <= 600 seconds
   - `601_1200`: 600 < duration <= 1200 seconds
   - `gt_1200`: duration > 1200 seconds

The source-video bins must never be derived from the 1s/10s/60s annotation
window or from the parent manifest's annotation-end lower bound. The formal
video-to-bin assignment is frozen in the readiness audit before inference.

Every formal report must include correct/total, accuracy, question count, and
distinct-video count overall, by question-duration class, and by source-video
duration class. Where counts permit, also report the source-length × question-
duration cross-tab. Never report percentages without sample counts.

For B0, each source-length bin must additionally aggregate GT Interval Hit@8,
mean nearest timestamp distance to the GT interval, mean/median online latency,
frame extraction and inference latency, text/visual tokens, frames/question,
and model calls/question. B1–B4 must reuse the same bins and aggregation logic;
where applicable they additionally report GT Interval Hit@K or retrieval
Recall@K. This protocol enables later paired tests of whether uniform sampling
degrades with video length and whether retrieval, hierarchy, or Planner changes
that pattern. No such conclusion is assumed in advance.

## Formal experiment log contract

After every formal run, append one concise entry to `docs/EXPERIMENT_LOG.md`
with run ID, date, branch, git commit, both manifest hashes, model/checkpoint,
configuration, exact change and reason relative to the previous stage, overall
metrics, both duration-grouped metric sets, efficiency metrics, artifact paths,
conclusion, limitations/anomalies, and next step. Raw machine-readable results
remain under `outputs/`; documentation is the handoff layer, not a substitute.
Update `docs/CURRENT_STATE.md` after each meaningful stage.
