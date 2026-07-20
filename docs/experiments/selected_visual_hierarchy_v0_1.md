# Selected Visual Hierarchy v0.1 — Frozen

## Status

The selected experimental visual pipeline is frozen as:

```text
CoMET-style Fine events
→ original pure Fluid Loose Medium frontier
→ one informative keyframe + local Qwen2-VL factual caption per Medium
→ deterministic Semantic Coarse
```

This is the selected visual structure for the next audio-timeline and unified
audio-visual-map work. It is not yet integrated into canonical Ours-v0.1.

## Frozen structure

| Approximate duration | Fluid Loose Medium | Semantic Coarse | Interpretation |
|---:|---:|---:|---|
| 5 min | 8 | 8 | Diverse content; no additional compression is required. |
| 9.5 min | 12 | 12 | Diverse content; no additional compression is required. |
| 20.5 min | 29 | 27 | Accepted under-grouping under rapid viewpoint/shot changes. |
| 23.8 min | 55 | 18 | Useful compression without obvious cross-stage overmerge. |

The Semantic Coarse boundary rule is deterministic: Sentence-T5 caption
similarity, DINO visual-transition veto, explicit state-conflict veto, and
group-centroid anti-chaining checks. Qwen is not a boundary judge.

## Accepted limitation

Pairwise local similarity may preserve multiple Coarse nodes within one broader
event when viewpoint or shot changes are rapid. This limitation is acceptable
for the current architecture because the hierarchy is non-destructive and later
retrieval can descend from Coarse to Medium to Fine.

## Freeze rules

Until explicitly reopened, do not change:

- Fine intervals or identities;
- Safe-Merge tree topology or lineage;
- the original `fluid_loose` Medium frontier;
- Medium keyframes or captions;
- deterministic Semantic Coarse boundaries or lineage.

The authoritative machine-readable freeze record is
`outputs/experiments/semantic_coarse_v0_1/frozen_visual_pipeline_manifest.json`.

## Next workstream

The next isolated workstream is the audio timeline and unified audio-visual
map. This freeze does not prescribe its segmentation, representation, alignment,
or fusion method; those choices require a separate specification.
