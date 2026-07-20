# Photometric-robust Fluid Medium v0.1

## Research question

Can an operating Fine-to-Medium frontier over the exact frozen Boundary-aware Safe-Merge tree cross photometric/flicker-driven pseudo-boundaries while preserving persistent structural transitions?

This is an isolated structural experiment. It does not modify Fine segmentation, Safe-Merge topology, parent/child lineage, or canonical Ours-v0.1. It creates no Coarse frontier and runs no captioning, Planner, retrieval, audio, QA, API, or download.

## Frozen inputs

The primary inputs are the four frozen EgoPolice long-video trees from `long_video_hierarchy_stress_v0_2`. The ten locally available short-video trees from `fine_to_coarse_hierarchy_v0_1` are included as robustness inputs. Both hierarchy JSONL files, every Fine interval identity, and every full topology identity are SHA-256 verified before and after execution.

## Methods

- **A Current Fluid Loose:** exact q-rank >= 0.40 and local-drop <= 0.30 baseline from `adaptive_fluid_hierarchy_v0_1`.
- **B Local Absolute:** removes within-video percentile gating. Its merge confidence is the equal mean of native-range semantic similarity, boundary weakness, variability stability, and duration/component safety. Original definitions and equal weights are retained; nothing is refit.
- **C Photo Robust:** B plus a fixed sensitivity profile that reduces the effective strength of boundaries jointly showing high original-DINO change, low normalized-DINO change, and high luminance/color change.
- **D Photo + Temporal:** C plus fixed pre/near/far hysteresis. Persistent changes are protected; transient/reverting changes receive further boundary down-weighting.

Strict, Balanced, and Loose thresholds and the full photometric sensitivity grid were frozen in the config before any generated review output was inspected.

## Auxiliary normalized DINO representation

The exact local `facebook/dinov2-small` (ViT-S/14) model is loaded once. Each existing 1 FPS RGB frame is converted to BT.601 luminance, robustly contrast-stretched using its 2nd and 98th luminance percentiles, replicated to three channels, and encoded as an L2-normalized CLS embedding. These isolated cached embeddings supplement rather than replace the original DINO features.

## Persistence formula

At every Fine boundary `t`, normalized-DINO centroids are formed for `t-5..t-1`, `t+1..t+5`, and `t+6..t+10` seconds. A persistent boundary requires both post windows to remain different from the pre window and near/far post states to be mutually consistent. A transient boundary requires a near-state jump followed by far-state return toward the pre state. Boundary windows clip safely at video ends.

## Executed results

All 140 method/video Medium frontiers passed complete Fine coverage, no-gap, no-overlap, and unique ownership validation. On the four primary long videos, A retained 104 Mediums. The B Strict/Balanced/Loose totals were 75/32/16. C retained exactly the same 75/32/16 operating frontiers: under the frozen grid, the photometric boundary adjustment did not cross an operating threshold. D produced 76/32/16; temporal persistence changed only the Strict primary total by one Medium.

This negative B-to-C ablation result is retained transparently. Removing percentile gating, rather than the photometric auxiliary signal, caused the large count reduction. It also created clear structural overmerge risk: primary median Medium duration rose to 73 seconds for Balanced and 218.5 seconds for Loose.

The automatically identified 108–209s flashing review region is in `youtube__OfficialDCPolice__j-1F2dyUxMM_1990_3418`. Its frozen Fixed-Reference frontier overlaps nine Mediums; Current Fluid Loose overlaps four. Most absolute variants select one much larger node spanning beyond the review interval (237 or 478 seconds), so lower count alone cannot be treated as success.

## Human review and limitations

`outputs/experiments/photometric_robust_fluid_medium_v0_1/medium_comparison.html` begins with the targeted flashing interval, then strong original-and-normalized transition proxies, stable-region proxies, and all ten full-video frontiers. Subjective judgments are blank.

- DINO, luminance, histogram, and persistence measures are structural proxies, not semantic event labels.
- The targeted interval was chosen by frozen Fixed-Reference overlap count, not by new method performance.
- Projected VLM calls equal Medium counts only; no VLM runtime saving is claimed.
- No final method is selected. In particular, aggressive absolute collapse requires human overmerge review.
