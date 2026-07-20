# Event-level vs Action-level Fine leaves v0.1

## Research question

This isolated experiment asks whether a future reusable temporal hierarchy should use the frozen lightweight CoMET-style appearance Event segments or finer within-Event motion-refined Action segments as its leaves. It evaluates structural granularity, parent preservation, long-leaf reduction, possible subtle-motion discoveries, fragmentation, and offline cost. It does not evaluate QA correctness.

## Frozen inputs

- Exact 10-video manifest: `data/manifests/coarse_segmentation_3way_10.json`.
- Immutable Event boundaries: `outputs/experiments/coarse_segmentation_3way_v0_1/comet_segments.jsonl`.
- Existing 1 FPS frames and timestamp grids.
- Existing normalized DINOv2 ViT-S/14 frame cache, reused only for appearance diagnostics and with zero new DINO inference.
- Previous KTS segments are shown only as a visual reference and do not construct or score either method.

The run manifest and frozen config persist SHA-256 identities for the frozen source artifacts. Every output Event boundary signature is compared against its source signature.

## Method A — EVENT_LEVEL

`EVENT_LEVEL` is an alias for the exact prior `comet_style_dinov2` output:

1 FPS frame grid → cached DINOv2 features → adjacent cosine similarity → Gaussian smoothing → adaptive local-minimum boundaries → frozen Event segments.

No Event boundary, timestamp, duration, or representative midpoint is recomputed or changed. This is a lightweight CoMET-style Event segmentation, not a complete reproduction of CoMET's Action-level pipeline.

## Motion backend audit and Method B

Installed `torchvision 0.20.1+cpu` exposes RAFT constructors, but neither official pretrained weight file is cached. Local torchvision metadata reports approximately 3.821 MB for RAFT-Small and 20.129 MB for RAFT-Large. OpenCV and `ruptures` are absent. No package or checkpoint was downloaded because introducing uncached RAFT into the current mixed CUDA-PyTorch/CPU-torchvision environment would change dependencies and execution risk.

Method B is therefore explicitly named `LIGHTWEIGHT_MOTION_PROXY`, not RAFT and not an exact CoMET Action reproduction:

1. Read the existing ordered 1 FPS frames at 160×90 grayscale.
2. For each adjacent pair, compute absolute intensity differences, edge-map differences, and a 3×4 spatial grid of change magnitudes.
3. Smooth every feature channel with a frozen Gaussian sigma of 1 frame.
4. Within each immutable Event only, robust-standardize the features.
5. Build an RBF kernel using the per-Event median pairwise squared-distance heuristic.
6. Run exact dynamic programming for the deterministic penalized multiple-change-point objective. This has the same objective class as PELT but uses no pruning because parent Events are short.
7. Use the frozen penalty `1.0 × log(n)` and a 3-second minimum Action duration. No target Action count is imposed.
8. Preserve an unsplit Event as one Action child when no penalized change point is supported.

The penalty was selected once using a zero-label count-only structural sanity check. No questions, options, gold labels, predictions, or QA correctness were inspected. The method's limited 1 FPS frame-difference sensitivity is a central limitation.

## Contracts

- Every Action has exactly one Event parent.
- Children start at the parent start and end at the parent end.
- Ordered children have no gaps or overlaps.
- No Action crosses its parent Event boundary.
- Events with no supported motion change remain one Action.
- Motion caches bind the exact ordered frame-content identity, timestamps, method schema, and configuration; stale caches are rejected.

## Diagnostics

`stable-appearance / motion-change candidates` require both low within-Event DINO appearance change and a strong accepted motion-regime change under dataset-derived quantiles. They are review candidates, not claims of distinct semantic actions.

Possible redundant fragmentation requires high adjacent Action DINO similarity and weak motion-regime change under data-derived quantiles. Short leaves and nearby boundaries are reported separately; none is automatically labelled an error.

The known stable-background video `afbc1ef9-bc2b-49f9-9009-3d7486563764` is always reported explicitly, regardless of whether it crosses the automatic diagnostic quantiles.

## Interpretation limits

- 1 FPS frame difference is a weak motion proxy and may miss fast or subtle optical flow.
- Camera/ego-motion can raise the proxy without representing a new semantic action.
- DINO appearance similarity and motion shifts are structural proxies, not temporal ground truth.
- Manual inspection of the 25/50/75 thumbnails and motion curves is required before supporting Event-only, Action-only, or an adaptive hybrid.
- This experiment does not build a hierarchy, route questions, run retrieval, modify Ours-v0.1, or execute final QA.

## Safety

Question text, answer options, gold answers, QA predictions, and QA correctness are not loaded. External API calls and new DINO inference calls are both zero. All files are isolated under `outputs/experiments/event_vs_action_fine_leaf_v0_1/`.
