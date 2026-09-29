# HourVideo version log

This document records the HourVideo R1_AV / R3_2 research lineage.  It
separates a runnable baseline from failed or diagnostic experiments so that a
later runtime change is not mistaken for an established result.

## Current status

- **Runnable baseline:** V3, `question_symmetric_visual_review_v3`.
- **Published runtime snapshot:** `hourvideo_v3_runtime/` contains the code
  path from the no-API question projection and ten-video selection through
  hierarchy, ASR, detector/caption stages, V1 R1_AV/R3_2 retrieval and V3.
- **V4:** a failed semantic-regression smoke; retained as a visual-acquisition
  diagnostic, not a baseline.
- **V5:** its question-operator and sufficiency-routing motivation remains
  useful, but its full contract compiler is stopped as a failed experiment.

## V1 — original R1_AV / R3_2 ten-video pipeline

**Purpose.** Establish one complete ten-video HourVideo path:

```text
safe question projection -> selected ten videos -> fixed Fine/Medium hierarchy
-> ASR + detector + local captions -> R1_AV or R3_2 map
-> planner/retrieval/initial sufficiency -> local visual review -> final QA
```

R1_AV uses structural/audio navigation.  R3_2 additionally has caption- and
ASR-derived semantic coarse-map text.  The map is a navigation/retrieval aid;
R1 structural text is not visual evidence, and R3 semantic map text is not
reviewed visual confirmation.

**Limitation discovered later.** Requirements were option hypotheses: the
system exposed the full options early and expanded one requirement per option.
This can duplicate retrieval and visual work, and it makes it hard for R3's
richer map to reduce visual review.

## V2 — batched visual review

**Purpose.** Reduce repeated Gemini calls by batching visual checks for
requirements.

**Change from V1.** Several requirements could share one visual-review batch.

**Remaining limitation.** The review was still organized around option-derived
requirements.  It reduced call overhead but did not change the underlying
question representation or require a strict sufficient-evidence gate.

## V3 — question-symmetric visual review

**Purpose.** Remove asymmetric option-by-option image inspection.

**Change from V2.** For each visual-review call, Gemini receives the question,
all answer options, and deduplicated retrieved Fine images together.  It first
records neutral observations, then assesses every option symmetrically.  Gemini
does not select the final answer in that call.  R1_AV and R3_2 caches are kept
separate.

**Ten-video live replay.**

| Measure | Result |
| --- | ---: |
| R1_AV accuracy | 6/10 |
| R3_2 accuracy | 8/10 |
| R1_AV Gemini images | 52 in 8 calls |
| R3_2 Gemini images | 48 in 7 calls |
| Total Gemini images | 100 in 15 calls |
| Haiku post-review calls | 0 |

The important diagnostic is not only that V3 is runnable.  Although R3_2 has a
richer semantic map, it still used nearly as many Gemini images as R1_AV.  V3
does not itself read map facts as a reasoning state; it consumes the upstream
retrieved Fine images and initial sufficiency output.  Thus map quality can
improve navigation but does not by itself suppress visual review.

**Status.** V3 is the current usable baseline.  The independent runtime
snapshot is source-identical to its copied source/config/test files; its
selected contract suite passed 17 tests, as did the original suite.

## V4 — progressive cached visual review

**Purpose.** Address the image-cost symptom without changing the V3 answer
contract.

**Change from V3.** Instead of the fixed V3 upload set, each unresolved
requirement receives its next ranked Fine candidate only when more visual
evidence might still resolve it.  Image observations are cached by image hash
as neutral facts, not as a prior question answer.  Cache misses are uploaded;
cache hits are reused.  The R1_AV and R3_2 cache roots remain isolated.

**What it did demonstrate.** V4 could reduce image transmission and reuse a
neutral observation across questions.  Its three smoke variants reported:

| Variant | V4 / old V3 image transmissions | Cache hits | Acceptance |
| --- | ---: | ---: | --- |
| V4 | 14 / 42 | 2 | failed semantic regression |
| V4.1 | 6 / 42 | 21 | failed semantic regression |
| V4.2 | 21 / 42 | 9 | failed semantic regression |

**Why it was not promoted.** This was only a two-question, four
rung/question-run smoke.  All three variants changed answers relative to V3;
their answer selection was stable in only 2/4 runs and each had at least one
old-correct-to-V4-wrong regression.  More fundamentally, V4 still begins from
V3's option-derived requirements.  It can decide *which next image* to fetch,
but not whether R3's map already establishes the relation asked by the question.
It therefore cannot solve the reason R3 should be more image-efficient.

## V5 — question operator and sufficiency-routing direction

**Purpose.** Address the root issue exposed by V3/V4:

```text
question meaning -> question operator -> minimum evidence needed
-> sufficiency decision -> local visual action only if necessary
```

The intended operators include temporal order, duration comparison,
between-anchors sequence, shared action, spatial route and global summary.
The intended sufficiency check is not a final answer model.  It should state
which evidence is established, uncertain or contradicted; whether the question
is answerable; which options can be eliminated; and what evidence action is
needed next.

### V5.1 / V5.2

These canaries made useful diagnostics explicit: duration needs event-boundary
search rather than sparse static frames, and a malformed option contract may
require contract repair rather than more images.  V5.2 also surfaced a critical
implementation flaw: its final answer stage could bypass post-sufficiency.
Multiple remaining candidates or unresolved requirements were sometimes still
labelled `answer_ready`.

### V5.3 minimal-contract compiler

V5.3 attempted to compile options into minimal typed contracts with relation
specifications and a hard final gate.  It added useful definitions for roles,
relation discriminators and abstention states.  However, real provider output
did not reliably conform to the canonical relation schema, even after
provider-schema compatibility revisions.  The full-contract compiler is
therefore stopped; it is not a runnable baseline and must not be presented as
one.

**Retained insight.** V5's operator-aware, evidence-sufficiency framing is the
right direction for a future clean redesign.  It should be reintroduced only
after the runtime has a simple, model-compatible contract and an end-to-end
validation plan; not by patching individual questions or adding more images.

## Practical interpretation

V3 answers the engineering question “can the complete R1_AV/R3_2 pipeline run
and compare all options symmetrically?”  Yes.

V4 answers “can neutral caching and progressive image acquisition reduce image
transmission?”  Sometimes, but its smoke regressed answer behavior, so it is
not accepted.

V5 asks the research question “can map-derived facts and typed sufficiency
avoid unnecessary visual inspection, especially for R3_2?”  This remains open.

Until that question is resolved, keep V3 fixed as the reproducible comparison
baseline and treat V4/V5 as documented diagnostics rather than successive
baselines.
