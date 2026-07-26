# Protocol Amendment #6 — Formal E2 minimum-prefix Medium routing

**Date:** 2026-07-24  
**Base protocol:** THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md  
**Prior amendments:** #1, #2, #3, #4, and #5

## 1. Reason and pre-quality status

Amendment #5 incorrectly froze B2 Stage-1 routing as an initial Top-3 Medium
prefix followed by expansion only when that prefix exposed fewer than the
target Fine count. This does not match the intended hierarchical retrieval
design: B2 must accumulate the minimum ranked Medium prefix needed to expose
enough Fine candidates.

This is a pre-quality clarification of the intended B2 routing algorithm, not
result-driven tuning. Amendment #5 implementation completed only two-clip
structural validation; full retrieval dryrun20 did not resume, and no B1/B2
recall, 7B QA, or QA-quality result has been inspected. Those historical
implementation artifacts remain preserved.

## 2. Shared target

For every canonical clip, define:

~~~
K = min(8, total number of Fine events)
~~~

B1 remains: search all Fine events with the same frozen Fine scorer,
deterministically rank them, then select Top-K Fine events and their K Fine
representative frames. When total Fine count is below eight, B1 uses all Fine
events.

## 3. Revised B2 Stage-1 routing

Formal B2 Stage 1 is:

1. score and deterministically rank all Medium events with the frozen
   question-to-Medium scorer;
2. begin with the single highest-ranked Medium, Top-1;
3. construct its deduplicated descendant Fine-ID union;
4. if that union has fewer than K Fine events, add the next-ranked Medium,
   Top-2, and recompute the deduplicated union;
5. continue one Medium at a time in exact rank order, Top-3, Top-4, and so
   on; and
6. stop immediately when the union size is at least K or all Medium events
   have been exhausted.

Thus Stage 1 uses the **minimum prefix** of the ranked Medium list required to
expose at least K Fine candidates. There is no fixed minimum of three Mediums.
The selected Medium count depends only on Medium rank and descendant Fine
candidate-count sufficiency relative to K; GT, answers, options, captions,
recall, and QA quality never enter this decision.

## 4. B2 Stage-2 and controlled comparison

After Stage 1, B2 takes the complete deduplicated union of descendant Fine
events from the selected Mediums, applies exactly the same question-to-Fine
scorer used by B1, deterministically ranks the candidates, and selects Top-K
Fine events and K unique Fine representative frames. If candidate Fine count
exceeds K, use Top-K; if it equals K, use all K. There is no per-Medium quota,
equal allocation, duplicate padding, or unrelated padding.

The controlled comparison is therefore:

- **B1:** question → search all Fine → Top-K Fine.
- **B2:** question → rank Medium → accumulate the smallest ranked Medium
  prefix with at least K descendants → search only those Fine → Top-K Fine.

Both methods share Fine segmentation, Fine representatives, Fine C-RADIO
vectors, question encoder, Fine scorer, final K-frame evidence budget, and the
frozen 7B answer model. The intended variable is global Fine-flat retrieval
versus Medium-guided restriction of the Fine search space.

## 5. Edge cases

- **Top-1 sufficient:** if the highest-ranked Medium already has at least K
  descendant Fine events, Stage 1 selects exactly one Medium.
- **Several Mediums required:** for cumulative counts 3, 5, 7, and 10 after
  Top-1 through Top-4, select Top-1 through Top-4 and stop, then rank those
  ten Fine candidates globally.
- **Total Fine below eight:** K equals the total Fine count; B2 accumulates
  until its union reaches K or all Mediums are exhausted.
- **Duplicate descendants:** deduplicate by Fine event ID before evaluating
  union size.
- **Exhaustion:** if all Mediums are exhausted before the union reaches K,
  record a hierarchy coverage/integrity failure and do not pad with unrelated
  frames.

## 6. Logging update

`stage1_selected_medium_count` is fully dynamic. Per query, record:

- `stage1_total_medium_count`
- `stage1_selected_medium_count`
- `stage1_expansion_steps`
- `stage1_selected_medium_ids_in_rank_order`
- `stage1_cumulative_fine_count_after_each_medium`
- `stage2_fine_candidate_count_before_topk`
- `target_final_fine_count_K`
- `final_selected_fine_count`

Historical Amendment #4 and #5 logs remain preserved. No semantic assumption
that three Mediums is a default or minimum remains active.

## 7. Exact supersession and unchanged scope

Amendment #6 supersedes only Amendment #5 provisions requiring:

- initial Top-3 Medium routing;
- a minimum routing count of three; and
- expansion beginning only after Top-3.

Amendment #5 remains historically preserved. Everything else from Amendments
#4 and #5 remains unchanged: B1 Fine-flat retrieval; `K = min(8, N_Fine)`;
Fine DINO-medoid representatives; Fine-only C-RADIO encoding; Medium
representation; Stage-2 global Top-K Fine; equal final evidence budget;
caption diagnostic-only status; retired B1-prime; hierarchy parameters; answer
model; canonical scope; sequential decode optimization; and cost/logging
separation. B0, E1, and Formal Closed/Open manifests are unchanged.

## Active protocol

The active protocol is V2.2 FINAL plus Amendments #1–#6. Amendment #6
supersedes only the affected Amendment #5 Stage-1 routing minimum-prefix and
associated logging semantics.
