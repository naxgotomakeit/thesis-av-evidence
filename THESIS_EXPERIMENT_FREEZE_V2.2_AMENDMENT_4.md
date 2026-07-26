# Protocol Amendment #4 — Formal E2 hierarchical coarse-to-fine routing

**Date:** 2026-07-24  
**Base protocol:** THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md  
**Prior amendments:** #1, #2, and #3

## 1. Motivation and pre-result status

Amendment #3 defined B1 as Fine-only flat retrieval and B2 as hierarchy-derived Medium events used directly as terminal flat retrieval units. That does **not** instantiate the intended hierarchical coarse-to-fine design: Fine events are discarded instead of being retained as final evidence units, and Medium is not used as a coarse routing/localization layer before Fine retrieval. It also required unnecessary full-timeline C-RADIO visual encoding.

Before any B1/B2 quality inspection, the frozen protocol was found not to instantiate the intended hierarchical coarse-to-fine design and to perform unnecessary full-timeline retrieval encoding. This amendment corrects the experimental construct and reusable index representation before quality evaluation. It is **not** a result-driven method change.

At the time of this amendment, retrieval quality diagnostics had not started, 7B QA had not started, and no B1/B2 accuracy or QA-quality comparison had been inspected. The paused dryrun20 offline artifacts and pre-optimization timings remain preserved historical engineering artifacts, not Formal E2 quality evidence.

## 2. Shared offline visual structure — unchanged

The verified frozen hierarchy and parameters remain unchanged:

~~~
canonical clip → 1 FPS → DINOv2-small features → Fine segmentation
  → Safe-Merge → Fluid Loose → Medium hierarchy
~~~

DINOv2 is responsible for temporal/event construction and Fine representative-frame selection. C-RADIO is not used for video segmentation.

## 3. Formal Fine representative rule

For every Fine event, collect valid 1-FPS frames inside its interval and use their already-computed normalized DINOv2 features. Compute pairwise DINO cosine similarity and select the frame with highest mean similarity to the other frames: the deterministic **DINO medoid**. A one-frame Fine selects itself; an exact tie selects the earlier timestamp.

This Formal Fine representative is shared identically by B1 and B2. Formal E2 must not use C-RADIO to select it, question-conditioned keyframe selection, the historical 25/50/75% candidate selector, or sharpness/exposure/entropy/discriminativeness scoring.

## 4. Formal C-RADIO indexing rule

The Amendment #3 full-1FPS-timeline C-RADIO requirement is retired for Formal E2 retrieval representation. Perform C-RADIO visual inference **only** on Formal Fine representative frames:

~~~
Formal Fine representative frame → C-RADIO visual encoder
  → normalized Fine retrieval vector
~~~

The frozen aligned retrieval family remains nvidia/C-RADIOv4-SO400M, adaptor siglip2-g, and aligned text model google/siglip2-giant-opt-patch16-384. Fine C-RADIO vectors are shared by B1 and B2. No extra C-RADIO visual inference is performed for Medium events.

## 5. Formal Medium representation rule

Medium is a hierarchy/routing unit, not a terminal evidence unit. A Medium's child Fine events already each have a Formal Fine representative frame, normalized DINO representative vector, and normalized C-RADIO retrieval vector.

Choose the **Medium representative Fine** using only pairwise cosine similarity among child-Fine DINO representative vectors: select the child Fine with highest mean similarity. An exact tie selects the earlier Fine / earlier timestamp. The Medium retrieval vector is the **existing C-RADIO vector** of that representative Fine.

Thus Medium selection is in DINO space, while later question-to-Medium and question-to-Fine similarities are in C-RADIO aligned space. There is no vector averaging/pooling, no child-Fine C-RADIO cosine-medoid selection, and zero additional C-RADIO visual inference.

## 6. Formal B1 — Fine flat retrieval

~~~
canonical clip → shared 1 FPS / DINOv2 / Fine segmentation
  → Formal Fine representatives → Fine C-RADIO vectors

question → aligned text encoder → cosine against all Fine vectors
  → deterministic global rank → global Top-8 Fine events
  → Fine representative frames → chronological order if required
  → maximum 8 unique frames → frozen answer model
~~~

No duplicate or unrelated padding is allowed. B1 uses no Medium routing, captions, Semantic Coarse, storyline, reranking, Planner, or audio.

## 7. Formal B2 — hierarchical coarse-to-fine retrieval

B2 shares B1's 1-FPS frames, DINO features, Fine boundaries, Formal Fine representatives, Fine C-RADIO vectors, aligned text encoder, cosine scorer, final ≤8 Fine-frame budget, and frozen answer model. Its only intended structural change is hierarchical coarse-to-fine routing.

**Stage 1 — Medium routing**

~~~
question → aligned text embedding → cosine against all Medium vectors
  → deterministic ranking → Top-M Medium events
~~~

M = 3 is frozen and must not be tuned from dryrun quality.

**Stage 2 — Fine evidence retrieval**

1. Take the union of all descendant Fine events of the Top-3 Mediums.
2. Score those Fine events with exactly B1's question-to-Fine cosine scorer.
3. Globally rank that union.
4. Select Top-8 Fine events and use their Formal Fine representative frames.

There is no equal allocation across Mediums, per-Medium quota, duplicate padding, or unrelated padding. B1 searches all Fine globally; B2 performs Medium routing then restricts Fine retrieval. The B1→B2 hypothesis is whether this routing improves evidence selection and/or retrieval efficiency under the same final evidence and answer-model budget.

## 8. Retirement of old B1′ control

The Amendment #3 B1′ trigger

~~~
abs(B1_event_count - B2_event_count) / B1_event_count > 0.25
~~~

and its equal-duration Fine grouping control are retired. They addressed the old Fine-flat versus terminal-Medium-flat candidate-count/granularity confound. Revised B2 returns to Fine evidence retrieval after Medium routing, so that control is no longer relevant. Historical B1′ artifacts remain preserved and labelled legacy_retired. No replacement pruning control is introduced.

## 9. Caption status

Medium captions may exist or be generated as reusable diagnostic metadata. For Formal B1/B2, however, captions are **diagnostic only**. They must not influence Medium routing, Fine ranking, selected evidence frames, or answers.

Diagnostics may separately compute Medium visual Recall@3, Medium caption Recall@3, and visual/caption complementarity. Any caption use in active retrieval requires a separately frozen system/version decision.

## 10. Diagnostic relevance definitions

Two relevance notions are frozen and must remain separate.

### A. Routing-level Medium relevance

A Medium is GT-relevant for Stage-1 Medium Recall@3 when its interval overlaps the GT evidence interval under the frozen temporal-overlap convention. Stage1_Medium_Recall@3 is true when at least one GT-relevant Medium is in the Top-3 routed Mediums. The Medium representative frame need not lie in GT.

### B. Evidence/frame-level hit

For non-point GT, a Fine representative is a hit iff:

~~~
GT_start <= representative_timestamp < GT_end
~~~

For point GT (start == end), inherit the exact E1 provisional rule from V2.2 §3.2 and src/experiments/qaego4d_e1/core.py: point_nearest_decodable_frame_provisional. Resolve the point to the nearest real decodable frame **within the canonical clip**, with exact tie going to the earlier frame. No ±T tolerance is introduced. A Fine representative is a point hit iff its source-frame identity equals that E1-resolved point reference frame. This is a frame-identity convention, not a new time-window tolerance.

## 11. Required pre-QA dry-run diagnostics

The next retrieval-only dry-run, before 7B QA, must report:

1. Fine Visual Recall@8.
2. Stage-1 Medium Visual Recall@3.
3. Medium Caption Recall@3, diagnostic only.
4. Visual/caption complementarity: hit/hit, hit/miss, miss/hit, miss/miss.

It must also report Fine/event representability, Fine representative FrameHit, nearest representative distance where applicable, hubness, event counts and fragmentation, Stage-1 candidate count, Stage-2 Fine candidate count, and offline/online timing.

Low Fine Recall@8 must first be decomposed into segmentation/representability, representative-frame realization, or visual-language retrieval/ranking. It does not automatically justify replacing Fine with Medium. Low Medium Recall@3 must not trigger post-hoc M tuning in the same run.

## 12. Sequential decode status

Repeated random-seek 1-FPS extraction was a major implementation-level cost. A sequential-decode candidate was tested on frozen clips and preserved source frame IDs, timestamps, JPEG bytes, DINO features, Fine boundaries, Medium boundaries, medoid timestamps, retrieval ordering, and final ≤8-frame identities. It may replace repeated-seek extraction as a behavior-preserving implementation optimization.

This does not change frozen 1-FPS sampling semantics. No direct-RGB/no-JPEG equivalence is claimed.

## 13. Cost and logging schema

B2 logs must separate stages without double-counting question encoding:

~~~
shared:
  question_encode_time_s

stage 1:
  stage1_medium_score_time_s
  stage1_ranking_time_s
  stage1_medium_candidate_count
  stage1_selected_medium_count = 3

stage 2:
  stage2_candidate_expand_time_s
  stage2_fine_score_time_s
  stage2_ranking_time_s
  stage2_fine_candidate_count
  stage2_selected_fine_count

total:
  total_retrieval_time_s
  total_online_latency_s
~~~

Offline accounting continues to distinguish offline_gpu_compute_s, offline_cpu_decode_s, offline_image_encode_s, storage read/write/metadata where measurable, offline_storage_bytes, and offline_wall_latency_s. Historical B1′ logging remains readable but is labelled legacy_retired.

## 14. Cost interpretation

Formal accounting distinguishes (A) method/compute cost, (B) observed end-to-end system wall latency, and (C) environment-dependent storage/I/O cost. NFS/shared-storage latency must not be attributed to hierarchy computation, but must not be omitted from observed wall time.

The pre-optimization audit remains historical engineering evidence: hierarchy construction was negligible, while wall time was dominated by repeated seek/frame materialization and full-timeline C-RADIO encoding. This amendment retires the latter for Formal E2 in favour of Fine-representative-only C-RADIO encoding.

## 15. Supersession and unchanged scope

This additive amendment supersedes only affected Amendment #3 E2 content:

- B1 retrieval representation where affected;
- B2 terminal-Medium retrieval definition;
- full-timeline C-RADIO medoid representation; and
- B1′ event-count control.

All unaffected Amendment #3 history remains preserved. Amendment #4 does not change E1, B0, hierarchy construction parameters, the frozen answer-model configuration, official Formal Closed/Open manifests, Amendments #1/#2, or E3–E6 definitions. It does not restore retired B2-A/B2-B/B2-C numbering.

## Active protocol

The active protocol is V2.2 FINAL plus Amendments #1, #2, #3, and #4, with Amendment #4 superseding only the E2 definitions enumerated above.

