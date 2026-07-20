# Coarse segmentation 3-way v0.1 — current implementation audit

This audit records the code actually executed by the current Ours-v0.1 visual
offline index. It does not modify the canonical method.

## Executed source lineage

- EgoSound builder: `scripts/build_visual_state_regions.py::main`
- EgoSchema adapter/builder: `src/evaluation/egoschema_visual_index.py::build_visual_indexes`
- Shared algorithm: `src/visual/state_regions.py::{normalize_embeddings, adjacent_cosine_distances, segment_frame_indices, build_regions}`
- Frozen configuration: `configs/visual_mvp.yaml`
- Runtime loading: `src/canonical_pipeline/query_scoring.py` loads
  `outputs/visual_index/{video_id}/visual_state_regions.json` and
  `region_embeddings.npy`.

EgoSchema does not use a different segmentation algorithm. Its adapter reuses
the same `extract_frames`, `frame_timestamps`, and `build_regions` functions.

## Exact current algorithm

1. FFmpeg samples JPEG frames at **1 FPS**.
2. OpenAI CLIP **ViT-B/32** encodes each frame. Image embeddings are L2-normalized.
3. For every adjacent pair, the signal is cosine distance
   `1 - dot(normalized[i], normalized[i+1])`.
4. A new region begins at frame `i` only when:
   - adjacent cosine distance is at least **0.18**, and
   - elapsed time since the current region began is at least **2 seconds**.
5. A terminal region shorter than 2 seconds is merged into its predecessor.
6. A region representation is the L2-normalized mean of its normalized CLIP
   frame features.
7. Its canonical representative keyframe is the member frame with maximum
   cosine similarity to the pooled region feature (embedding medoid).
8. Regions cover the video contiguously and are saved under
   `outputs/visual_index/{video_id}/`; `frame_embeddings.npy` and the 1 FPS
   frames are reusable offline assets.

There is no smoothing, adaptive threshold, maximum duration, target segment
count, or semantic change-point model in the current segmenter.

## Why giant regions occur

The fixed threshold tests each adjacent transition independently. A long gradual
change can have meaningful accumulated semantic drift while every individual
1-second transition remains below 0.18. In that case no boundary is created.
The absence of a maximum region length or adaptive/local criterion means this can
produce one 180-second region. The 25-video statistics include two such videos.

## Experimental isolation

The new experiment reads the current outputs but writes only under
`outputs/experiments/coarse_segmentation_3way_v0_1/` and an experimental DINO
cache. No canonical index, planner, retrieval, sufficiency, Task6, or QA source is
rewritten.

KTS reference: Potapov, Douze, Harchaoui, and Schmid, *Category-specific video
summarization*, ECCV 2014. The authors' project page publishes temporal
segmentation code and specifies a kernel change-point objective with a BIC-style
segment-count penalty.

## Experimental KTS lineage

No KTS package was installed in the project environment. Method C is therefore
a local NumPy adaptation of the published KTS algorithm, not a heuristic that is
merely labelled KTS. Its source lineage is:

- Potapov et al., *Category-specific video summarization*, ECCV 2014,
  DOI `10.1007/978-3-319-10599-4_35`;
- the authors' `kts_ver1.1` distribution linked from their project page;
- the openly mirrored `cpd_nonlin` / `cpd_auto` reference interface, which uses
  dynamic programming over the full kernel, within-segment kernel scatter, and
  `vmax*m/(2*N)*(log(N/m)+1)` for automatic change-point-count selection.

The experiment uses a linear cosine kernel over the shared L2-normalized DINOv2
features. The KTS objective, dynamic program, and penalty form follow the
reference. `max_change_points=30`, `minimum_segment_duration=4s`,
`penalty_strength=1`, and normalized-kernel diagonal `vmax=1` are explicitly
frozen **experimental defaults**; they are not claimed to be paper-specified
values and were not tuned using QA labels or correctness.

## Method B parameter status

Method B implements the requested CoMET-style coarse appearance/event pipeline:
adjacent DINO cosine similarity, Gaussian smoothing, robust adaptive prominence,
and local-minimum change points. The high-level pipeline is the specified
CoMET-style design. Its smoothing, MAD multiplier, minimum prominence, and
minimum-duration values are experimental defaults because no exact official
coarse-stage values were confirmed. RAFT, PELT, and action-level motion
segmentation are intentionally excluded.
