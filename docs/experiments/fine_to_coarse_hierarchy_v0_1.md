# Fine-to-coarse hierarchy v0.1

## Research question

Can immutable full-video CoMET-style fine temporal segments be organized into a reversible Fine → Medium → Coarse hierarchy by bottom-up adjacent merging, while reducing fragmentation without destructively combining distinct egocentric hand/action states?

This is an isolated offline hierarchy proof-of-concept. It does not implement Planner routing, retrieval, sufficiency, fallback, Task6, or final QA.

## Frozen inputs

The experiment reuses exactly the 10-video manifest from `coarse_segmentation_3way_v0_1`, its full-video CoMET-style segments, DINOv2 ViT-S/14 1 FPS frame cache, original raw/smoothed adjacent-similarity signals, frame thumbnails, and KTS segmentations. KTS is a visual reference only and never influences either hierarchy.

Questions, answer options, gold labels, timestamps, prior predictions, and QA correctness are neither loaded nor used. No DINO inference or external API call occurs.

## Immutable Fine leaves

Every original CoMET segment becomes one atomic Fine leaf with the exact source ID, start/end interval, pooled normalized DINOv2 representation, original left/right boundary records, 25/50/75% thumbnails, and a DINO medoid thumbnail. Fine intervals cannot be changed or deleted.

Every internal node records two ordered adjacent children, its complete ordered Fine-descendant list, parent, interval, pooled representation, internal variability, crossed original boundaries, and representative frames. Each complete binary tree is validated as reversible to the original Fine sequence.

## Method A: adjacency-constrained Ward

Only adjacent active components are eligible. At each iteration, the implementation computes the Ward-style increase in within-cluster sum of squared error:

`n_A n_B / (n_A + n_B) * ||mean_A - mean_B||²`

The minimum-cost adjacent pair is merged; ties use stable temporal order and IDs. All evaluated accepted/rejected pairs and the full merge tree are saved.

## Method B: boundary-aware Safe Merge

This is an experimental adaptation, not a reproduction of a named paper. Only adjacent active components are eligible. Each candidate records:

- DINO semantic difference;
- original CoMET inter-boundary strength;
- left/right and merged internal variability;
- variability increase;
- duration and Fine-component size.

The merge score equally averages four per-video normalized signals: empirical semantic-difference percentile, original-boundary-strength percentile, empirical variability-increase percentile, and duration/component-size pressure. There is no fixed cosine threshold.

During Fine→Medium aggregation, original Fine boundaries in the top decile of the current video's boundary-strength distribution are vetoed. If every remaining boundary is vetoed before the 50% reference target, Medium stops early. The veto is released after the Medium cut so the complete tree can still be constructed and reversed. All accepted and rejected candidates retain explicit reasons.

## Visualization cuts

Both methods first construct complete trees. Medium and Coarse are transparent reference cuts rather than inferred ground-truth granularity:

- Medium: approximately 50% of the Fine node count;
- Coarse: approximately 25% of the Fine node count.

The Safe Medium cut may contain more nodes if the strong-boundary veto blocks further first-level aggregation.

## Structural diagnostics

The experiment reports Fine preservation, node compression, duration statistics, boundary survival by weak/medium/strong strata, potential over-merge nodes, surviving highly similar Fine boundaries (under-merge), endpoint-divergence chaining risk, cross-video granularity variability, KTS-like largest-region collapse, and stable-background/subtle-manipulation review candidates.

These are diagnostics, not action labels or semantic ground truth. A human must inspect aligned timelines and thumbnails before judging whether a merge is coherent.

## Runtime interpretation

DINO features are loaded from the prior cache and new feature inference is zero. Hierarchy construction is an offline indexing operation. Fast construction alone is not an efficiency contribution; future online benefit would require a separate question-routing/descend/fallback experiment.

## Non-changes

Canonical Ours-v0.1, Task5A/B/C, Task6, final QA, KTS/CoMET routing, and existing experiment artifacts remain unchanged.

## Completed structural result

The frozen 10 videos contain 159 Fine leaves (15.9/video). Both complete trees preserve and reconstruct 159/159 leaves. The reference cuts contain 8.2 Medium and 4.3 Coarse nodes/video for both methods, corresponding to mean node reductions of 48.6% and 73.0%.

At Medium, Ward preserves 54.7% of strong original boundaries and Safe Merge preserves 81.1%. Ward has 15 compound over-merge-risk nodes and Safe has 5. However, Safe's mean largest-region ratio is 0.376 versus Ward's 0.243, showing that a strong-boundary veto can redirect compression into long weak-boundary runs. Safe therefore appears safer by boundary-crossing diagnostics but is not uniformly safer by region-size diagnostics. Human timeline review is required.
