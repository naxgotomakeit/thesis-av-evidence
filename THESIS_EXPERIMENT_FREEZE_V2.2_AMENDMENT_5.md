# Protocol Amendment #5 — Formal E2 shared final Fine evidence count

**Date:** 2026-07-24  
**Base protocol:** THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md  
**Prior amendments:** #1, #2, #3, and #4

## 1. Reason and pre-quality status

Amendment #4 implementation integrity testing found a structural evidence-budget
confound before any quality inspection. B1 could expose eight Fine representative
frames, while B2's fixed Top-3 Medium routing could expose fewer descendant Fine
candidates. The two integrity clips showed 15 Medium → Top-3 → 5 descendant Fine
and 12 Medium → Top-3 → 5 descendant Fine.

This amendment corrects final evidence-count control before retrieval dryrun20
quality diagnostics, B1/B2 recall inspection, 7B QA, or accuracy comparison.
It is a pre-quality experimental-control correction, not result-driven tuning.

## 2. Formal target evidence count

For each canonical clip, define:

~~~
K = min(8, total number of Fine events in the clip)
~~~

B1 scores all Fine events with the frozen Fine scorer, deterministically ranks
them, and selects Top-K Fine events and their K unique Fine representative
frames. If total Fine count is below eight, B1 uses all Fine. No duplicate or
unrelated padding is allowed.

## 3. Revised B2 Stage-1 routing

B2 preserves the frozen Top-3 Medium routing prior:

1. score and deterministically rank all Medium events;
2. initially select Top-3 Medium events, or all Mediums when fewer than three
   exist;
3. construct the deduplicated union of descendant Fine event IDs;
4. if that union has fewer than K Fine events, add the next-ranked Medium one
   at a time: Top-4, then Top-5, and so on;
5. recompute the descendant Fine union after every addition; and
6. stop when the union size is at least K or all Mediums have been selected.

This is **not** Top-1 accumulation. Whenever at least three Mediums exist, the
routing prior starts at Top-3.

## 4. Revised B2 Stage-2

After Stage 1, take the complete deduplicated union of descendant Fine events
from all selected Mediums. Score every candidate with the same frozen Fine
scorer as B1, deterministically rank them, and select Top-K Fine events and K
unique Fine representative frames.

When the union is larger than K, use global Top-K. When it equals K, use all
K. If all Mediums are exhausted while the union remains below K, record that
structural integrity condition explicitly. Under valid exhaustive parent-child
coverage, total Fine below eight plus all-Medium exhaustion causes B1 and B2 to
both reduce to all Fine events.

There is no per-Medium quota, equal frame allocation, duplicate padding, or
unrelated padding.

## 5. Comparability guarantee

B1 and B2 use the same target final evidence count:

~~~
K = min(8, N_Fine)
~~~

The comparison therefore controls shared Fine segmentation, Fine
representatives, Fine C-RADIO vectors, question encoder, Fine scorer, final
evidence-frame count K, and frozen answer model.

The primary difference remains:

- B1: global flat search over all Fine events.
- B2: Medium-guided restriction of the Fine candidate pool before the same
  Fine-level ranking.

This prevents final-frame-count differences from confounding B1 versus B2.

## 6. Dynamic Medium-count logging

Amendment #4's fixed log assertion

~~~
stage1_selected_medium_count = 3
~~~

is superseded. It now records the actual number selected after deterministic
expansion, plus:

~~~
stage1_initial_medium_count = min(3, total Medium count)
stage1_selected_medium_count
stage1_expansion_steps
stage1_total_medium_count
stage1_expansion_required
stage2_fine_candidate_count_before_topk
target_final_fine_count_K
final_selected_fine_count
stage1_candidate_sufficiency_ratio =
  stage2_fine_candidate_count_before_topk / K
~~~

Question encoding and scoring costs must not be double-counted.

## 7. Required integrity conditions

A later implementation validation must establish:

- B1 final selected frame count equals K.
- B2 final selected frame count equals K unless a clearly logged structural
  integrity failure prevents it.
- B2 starts with Top-3 Mediums when at least three exist.
- Additional Mediums follow deterministic rank order only.
- Stage-2 candidates are exactly the union of descendants of selected Mediums.
- No Fine outside selected Medium descendants enters B2 Stage 2.
- No GT annotation influences Medium expansion.

Expansion uses only candidate-count sufficiency relative to K, never GT,
answer, caption, or quality.

## 8. Edge cases

**Case A — total Fine >= 8:** K is eight. B1 selects Top-8 Fine. B2 expands
Medium routing until the descendant union has at least eight Fine or all
Mediums are exhausted, then selects global Top-8 Fine.

**Case B — total Fine < 8:** K is total Fine count. B1 uses all Fine. B2
expands until it reaches K or exhausts all Mediums; complete hierarchy coverage
then exposes the same total number of Fine events.

**Case C — total Medium < 3:** B2 starts with all available Mediums.

**Case D — duplicate descendants:** deduplicate by Fine event ID before
candidate-union counting.

## 9. Unchanged scope

This amendment does not change B0, E1, canonical clip scope, 1FPS semantics,
DINOv2 Fine segmentation, Safe-Merge, Fluid Loose/Medium, the Fine DINO-medoid
representative rule, Fine-only C-RADIO encoding, the Medium representation
rule, caption diagnostic-only status, the question/text encoder, cosine
retrieval semantics, answer model, maximum evidence budget of eight, sequential
decode optimization, retired B1′ status, or Formal Closed/Open manifests.

Amendment #4 remains historically preserved and is not edited.

## 10. Exact supersession

Amendment #5 supersedes only these Amendment #4 clauses:

- §7's fixed B2 Stage-1 statement that Top-3 Mediums alone define the Stage-2
  candidate pool;
- §7's implicit Top-8 final Fine handling when fewer than eight descendants
  are available;
- §13's fixed stage1_selected_medium_count = 3 logging value.

All other Amendment #4 definitions remain active and unchanged.

## Active protocol

The active protocol is V2.2 FINAL plus Amendments #1–#5, with this amendment
superseding only the affected Stage-1 expansion, Stage-2 K-budget handling, and
logging definitions above.
