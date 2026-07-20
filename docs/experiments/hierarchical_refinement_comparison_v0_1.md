# Hierarchical refinement comparison v0.1

## Research question

Can KTS act as a reusable medium-grained retrieval index, with the frozen
CoMET-style segmenter applied only inside question-relevant KTS zones, while
retaining most of the useful fine temporal evidence exposed by full-video
CoMET and reducing fine processing scope and candidate count?

This is an isolated structural experiment. It does not modify or execute the
Planner, DIG, Task5C, Task6, sufficiency/fallback, or final QA pipeline.

## Controlled inputs

The experiment exactly reuses `coarse_segmentation_3way_v0_1`:

- the same frozen 10-video EgoSchema manifest and question text;
- the same 180-frame, 1 FPS grids;
- the same cached normalized DINOv2 ViT-S/14 features;
- the same existing CLIP ViT-B/32 frame features and question-only scorer;
- the same KTS boundaries/configuration;
- the same full-video CoMET boundaries/configuration.

No segmentation parameter is retuned. Answer options, gold labels, prior QA
correctness, and dataset answer timestamps are prohibited inputs.

## Compared methods

### KTS_ONLY

All frozen KTS units are represented with the shared CLIP pooling rule, scored
against the question, and ranked deterministically. The primary result retains
Top-3 KTS units without fine segmentation.

### FULL_COMET

The already-generated full-video CoMET-style segments are scored with the same
CLIP mechanism and Top-3 are retained. This is a fine-segmentation/retrieval
reference only. It is not a reproduction of CoMET's Filter Agent or downstream
reasoning system.

### KTS_LOCAL_COMET

All KTS units are scored, Top-3 are selected, and only directly consecutive
selected KTS partition units are merged. The existing global DINO cache is
sliced by resulting candidate zones. The frozen CoMET-style segmenter runs
independently within every zone, the resulting fine segments are scored by the
same CLIP mechanism, and Top-3 local segments are retained.

Refinement always runs. There is no sufficiency-dependent early stopping.

## Diagnostic containment proxy

FULL_COMET Top-3 segments are not ground truth. Their midpoint containment in
the selected KTS zones and temporal overlap with the hybrid Top-3 are therefore
reported only as diagnostic proxies. These values must not be called evidence
recall or temporal-ground-truth recall.

## Sensitivity analysis

Top-M KTS gating is reported for M=1, 2, and 3 without choosing a value using QA
correctness. The primary frozen method remains M=3. The analysis exposes the
candidate-zone coverage versus full-CoMET containment trade-off only.

## Limitations

- There is no temporal evidence ground truth.
- CLIP relevance is a shared retrieval proxy, not semantic correctness.
- Full-video CoMET may be fragmented or miss useful context itself.
- Candidate-count reduction is only potential downstream filtering reduction;
  no Filter Agent or API cost is executed or estimated.
- Human review is required to assess boundary coherence, contextual adequacy,
  redundancy, and whether local CoMET adds useful detail.

## Safety

Claude, Gemini, Whisper, Planner, final QA, and all paid APIs are unused. The
experiment reads canonical offline CLIP frame indexes but writes only to
`outputs/experiments/hierarchical_refinement_comparison_v0_1/`. No output is
integrated into canonical Ours-v0.1.
