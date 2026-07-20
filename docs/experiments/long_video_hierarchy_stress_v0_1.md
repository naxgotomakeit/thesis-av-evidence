# Long-video hierarchy stress v0.1

## Research question

Does the frozen CoMET-style Fine Event → Boundary-aware Safe Merge hierarchy remain structurally meaningful as video duration grows?

## Local-data limitation

The local inventory contains 760 readable videos across the EgoSchema subset and EgoSound roots. The maximum duration is 180 seconds; no video reaches 5, 10, 15, or 20 minutes. The five selected EgoSound videos are therefore the deterministic longest available held-out samples. This run cannot establish long-duration scaling and must be interpreted only as a same-duration generalization check.

## Frozen method

- 1 FPS deterministic frame grid.
- Frozen local `facebook/dinov2-small` CLS embeddings, L2 normalized.
- Existing CoMET-style smoothing and adaptive local-minimum configuration.
- Existing Boundary-aware Safe Merge implementation and 50%/25% reference cuts.
- Questions, retrieval, Planner, QA, gold, options, VLM semantics, and external APIs are absent.

## Outputs

The primary artifact is `outputs/experiments/long_video_hierarchy_stress_v0_1/hierarchy_review.html`. It exposes nested Coarse → Medium → Fine membership, timestamps, multiple representative images, and blank human-review controls.

## Interpretation

Numerical node-duration and chaining diagnostics are structural warnings, not semantic judgments. A true long-video follow-up requires new local 5–20 minute egocentric assets and was not fabricated here.

## Completed held-out results

The frozen fallback set is EgoSound `00047` (180.0s), `00109` (180.0s), `00155` (180.0s), `00249` (180.0s), and `00026` (178.067s). Mean node counts were 19.0 Fine, 9.8 Medium, and 5.0 Coarse per video. Mean/median-of-video-medians/mean-maximum durations were:

- Fine: 9.61 / 8.00 / 24.20 seconds.
- Medium: 18.53 / 9.03 / 55.60 seconds.
- Coarse: 36.52 / 19.00 / 89.20 seconds.

Mean largest-node ratios were 0.135, 0.310, and 0.497 respectively. The structural diagnostics flagged one giant reference-cut parent and four child-imbalance/chaining-risk nodes. These flags guide the manual HTML review; they do not establish semantic failure.

The final warm/cache-aware replay measured 0.31s DINO model load, 1.11s uncached DINO feature extraction for the newly corrected fifth candidate, 0.003s CoMET segmentation, and 0.056s Safe Merge construction. Persisted cache provenance records 5.15s total original DINO extraction across all five videos. Existing frame caches for the final replay were reused and are not misreported as new extraction cost; cold frame-decode timing for the corrected five-video set was not reconstructed after the resume path.
