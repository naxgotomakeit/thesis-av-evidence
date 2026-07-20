# Adaptive fluid hierarchy v0.1

## Research question

Can the existing complete Boundary-aware Safe Merge tree support content-adaptive operating frontiers, so visually repetitive branches collapse more deeply while complex branches retain more detail, without changing any Fine leaf, merge order, node, edge, or parent-child relationship?

## Frozen data and topology

The primary input is the four-video EgoPolice hierarchy from `long_video_hierarchy_stress_v0_2`. The available ten-video short hierarchy from `fine_to_coarse_hierarchy_v0_1` is included for aggregate robustness. Input file SHA-256, per-video Fine-leaf identity, and full topology identity are recorded before execution and verified afterward.

No segmentation, feature extraction, semantic captioning, retrieval, Planner, classifier, audio, QA, API, or download is performed.

## Compared operating frontiers

1. **Fixed Reference:** stored approximately 50% / 25% Medium/Coarse cuts.
2. **Fixed Coarser:** frozen approximately 30% / 10% sensitivity baseline.
3. **Global Adaptive Elbow:** the two largest adjacent gaps in each video's sorted accepted Safe-Merge cost distribution define global Medium and Coarse cost thresholds. There is no target count.
4. **Fluid Strict:** Medium q-rank >= .70/drop <= .10; Coarse q-rank >= .55/drop <= .20.
5. **Fluid Balanced:** Medium q-rank >= .55/drop <= .20; Coarse q-rank >= .40/drop <= .30.
6. **Fluid Loose:** Medium q-rank >= .40/drop <= .30; Coarse q-rank >= .25/drop <= .40.
7. **Fluid Balanced Guarded:** Balanced plus the prescribed duration/ratio ablation guards.

The stored Safe-Merge `merge_score` is a cost where lower means safer. To match the specified quality-oriented pruning thresholds, the experiment applies only the monotonic orientation transform `q_raw = 1 - merge_score`; it does not change weights or refit a score. `q_rank` is the within-video empirical quality percentile. `local_drop` is the maximum direct internal-child quality minus parent quality; leaf children do not contribute.

Local policies start at the frozen root and either collapse a node or recurse to its frozen children. This produces different stopping depths in different branches. The relaxed Coarse traversal is verified to group every selected Medium exactly once.

## Diagnostics

The experiment records entity counts, duration distributions, selected-depth distributions, exact Fine coverage, temporal coverage, gaps, overlaps, and lineage. Existing DINOv2 medoid frames/embeddings support adjacent-Medium similarity diagnostics. Existing Fine-boundary strength and internal visual variability support non-semantic overmerge/chaining risk flags.

Fixed 60-second windows are analysis-only. Fine-boundary density, mean adjacent-frame visual change, and retained Medium midpoint density are correlated using Spearman statistics. These windows do not define any frontier.

Projected VLM calls equal Medium count and are reported only as future call-count projections. No VLM is run and no runtime saving is claimed.

## Human review

`outputs/experiments/adaptive_fluid_hierarchy_v0_1/hierarchy_comparison.html` is the primary artifact. It shows seven aligned timelines, expandable Coarse-to-Medium structures with actual frozen representative images, complete fluid decision traces, elbow curves, structural flags, complexity windows, and blank human-review controls. No winner is automatically selected.

## Interpretation limits

- Fixed count sensitivity is not a proposed semantic level.
- Largest-gap elbows are a transparent baseline, not a solved natural-stopping method.
- DINO similarity and boundary flags are structural proxies, not semantic correctness.
- A duration guard is an ablation rather than a theoretically optimal policy.
- Human review is required before choosing strict, balanced, loose, guarded, or no fluid method.

## Executed structural results

All 98 method/video operating frontiers (7 methods x 14 frozen trees) passed Fine preservation, coverage, non-overlap, and unique-lineage validation. On the four primary long videos, the fixed reference had 295 Fine, 149 Medium, and 76 Coarse nodes. Fluid Strict produced 180/141 Medium/Coarse nodes, Fluid Balanced 141/104, Fluid Loose 104/70, and Balanced Guarded 145/104. Thus Strict increased the projected Medium-caption count, Balanced reduced it by 5.4%, and Loose reduced it by 30.2%; these are projected call counts only because no VLM was run.

The deliberately coarser fixed cut reduced the primary Medium count by 39.6%, while Global Elbow reduced it by 78.5%. Both also created substantially larger parent regions: the maximum Coarse/video ratio reached 0.826 for Fixed Coarser and 0.875 for Global Elbow. These are structural overmerge-risk signals, not semantic error labels.

Fluid Balanced's 60-second complexity/density correlation was positive on three of four long videos (Spearman rho 0.926, 0.670, and 0.817), but the approximately five-minute video was a counterexample (rho 0.145, with greater retained-node density in the designated stable windows). The local rule therefore demonstrates genuinely branch-dependent stopping but does not uniformly realize the intended stable-collapse behavior.

Adaptive frontier selection itself was sub-millisecond to low-millisecond per method over all 14 trees. Cached representative loading and structural diagnostics dominated the latest deterministic cached replay (1.37 seconds total, including 0.25 seconds for HTML generation). No DINO inference, captions, API calls, or downloads occurred.
