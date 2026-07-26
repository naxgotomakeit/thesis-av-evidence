# Protocol Amendment #3 — Controlled E2 B1/B1′ Event-Unit Definition

**Date:** 2026-07-24  
**Base protocol:** `THESIS_EXPERIMENT_FREEZE_V2.2_FINAL.md`  
**Prior amendments:** `THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_1.md`, `THESIS_EXPERIMENT_FREEZE_V2.2_AMENDMENT_2.md`

## Scope and motivation

This amendment supersedes **only** the B1 direct-coarse-flat event definition,
the B1′ event-count-matched instruction, and the corresponding B1/B2 E2
implementation details in V2.2 §2.2, §2.4, and §2.4.1. It preserves all other
V2.2 definitions.

V2.2 originally described B1 as direct coarse flat segmentation. The E2
implementation-readiness audit identified an uncontrolled comparison: the
historical flat implementation used C-RADIO segmentation, whereas the B2
hierarchy used DINOv2 Fine → Safe-Merge → Fluid Loose → Medium. Thus those
historical paths differed in event-construction backbone, segmentation rule,
event representation, and representative-frame logic.

A controlled B1/B2 comparison requires a shared temporal starting point. This
amendment is made before any Formal E2 run, before any B1/B2 QA-quality result
has been observed, and before development-video download. It is therefore a
pre-result protocol correction, not post-hoc tuning.

## Formal B1 — Fine-only flat reusable index

For E2, Formal B1 is:

```text
offline:
canonical clip
  → 1 FPS timeline
  → DINOv2-small features
  → frozen Fine segmentation
  → stop hierarchical derivation
  → Fine events are flat retrieval candidate units
  → shared representative-frame C-RADIO aligned embedding
  → reusable flat event index

online:
question
  → shared aligned text encoder
  → cosine similarity against B1 event representations
  → shared deterministic ranking and selection
  → at most 8 representative frames
  → frozen 7B answer model
```

B1 must not use Safe-Merge, Fluid Loose, Medium, captions, Semantic Coarse,
storylines, reranking, Planner, or audio. It has no B1-specific segmentation
threshold: its event boundaries are exactly the shared frozen Fine boundaries.

## Formal B2 — hierarchy-derived Medium reusable index

B2 retains its substantive hierarchy definition, but its starting point is now
explicitly identical to B1:

```text
canonical clip
  → same 1 FPS timeline
  → same DINOv2-small features
  → same Fine segmentation
  → Safe-Merge
  → Fluid Loose
  → Medium events as retrieval candidate units
```

B2 then uses exactly the same C-RADIO aligned retrieval representation rule,
question/text encoder, cosine scorer, deterministic ranking/selection rule,
representative-frame rule, at-most-eight-frame budget, and frozen 7B answer
model/prompt/decoding as B1.

The E2 variable is therefore **Fine-only flat event units versus
hierarchy-derived Medium event units**.

## Shared event retrieval representation

DINOv2 is used only for temporal/event construction. Text-to-event retrieval
uses the validated C-RADIO aligned space for B1, B1′, and B2:

- model: `nvidia/C-RADIOv4-SO400M`, checkpoint revision
  `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8`;
- official implementation: `NVlabs/RADIO` revision
  `c0f37017930e9dda53f93424cf4bf39fc51f287e`;
- aligned adaptor: `siglip2-g`, text model
  `google/siglip2-giant-opt-patch16-384`;
- visual and text vectors are L2-normalized; scoring is cosine similarity.

Each candidate event contributes exactly one question-independent retrieval
representation. From valid 1-FPS frames within that event, compute normalized
C-RADIO visual embeddings and select the medoid: the frame with the greatest
mean cosine similarity to the other event frames. A one-frame event selects its
only frame. Exact ties select the earlier timestamp. The event retrieval vector
is the normalized C-RADIO embedding of that selected representative frame.

Formal E2 does not use multi-frame mean pooling for retrieval, cross-event
discriminativeness, captions, or question-dependent keyframe selection.

## Conditional B1′ — equal-duration count-matched Fine control

The previous instruction to adjust B1 flat-segmentation granularity is
superseded. B1′ is activated only when the absolute B1/B2 event-count
difference exceeds 25%, using the existing frozen gate.

B1′ starts from the exact same Fine events as B1. For a canonical clip with
duration `T` and `K` B2 Medium events, construct exactly `K` contiguous groups:

1. desired internal boundaries are `clip_start + j*T/K`, for `j = 1 ... K-1`;
2. a group boundary may occur only on an existing Fine boundary;
3. snap each desired boundary to the nearest valid Fine boundary while
   preserving order and leaving at least one Fine event in every group;
4. an equal-distance tie selects the earlier Fine boundary;
5. if independent nearest snapping would create an empty or non-monotonic
   group, select the nearest valid boundary that satisfies the non-empty-group
   constraint;
6. merge the Fine events in each resulting contiguous group.

B1′ may receive from B2 only `K`, the Medium-event count. It must not use
Medium boundaries, Safe-Merge decisions, Fluid Loose decisions, semantic
similarity, question information, or captions.

## Frozen interpretation

- **B1 vs B2** tests the total effect of hierarchy-derived event construction
  relative to Fine-only flat units.
- If activated, **B1′ vs B2** tests whether hierarchy-derived grouping adds
  value beyond event-count/granularity matching.

Pre-registered interpretations, not formal significance claims before Formal
E2:

- B2 > B1 and B2 > B1′: evidence consistent with hierarchy-specific benefit.
- B2 > B1 but B2 ≈ B1′: benefit likely dominated by candidate-count/granularity
  reduction.
- B2 ≈ B1: no detectable hierarchy benefit under the tested setup.
- B2 < B1: hierarchy may lose or localize evidence poorly; inspect frozen
  diagnostics without post-hoc redefinition.

## Unchanged definitions

This amendment does not change B0; E1; the E2 OpenQA/auxiliary Closed
structure; the at-most-eight-frame answer budget; the frozen 7B answer-model
configuration; E3/E4/E5/E6; Amendments #1/#2; or the official Formal
Closed/Open manifests. Retired B2-A/B2-B/B2-C numbering remains retired.

## Active protocol

The active protocol is V2.2 FINAL plus Amendments #1, #2, and #3.
