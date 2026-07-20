# EgoPolice genuine long-video hierarchy stress v0.2

## Research question

This isolated experiment asks whether the already implemented CoMET-style Fine Event segmentation and Boundary-aware Safe Merge hierarchy remain structurally inspectable as clip duration increases from about three minutes to approximately 5, 10, 20, and 30 minutes.

It is not a QA, retrieval, planning, classification, or semantic-captioning experiment. Questions, options, gold labels, retrieval scores, Planner output, VLM summaries, and paid APIs are absent.

## Frozen implementation

- Fine segmentation: `src/experiments/coarse_segmentation/comet_style.py`
- DINOv2 feature cache: `src/experiments/coarse_segmentation/dinov2_features.py`
- Boundary-aware Safe Merge: `src/experiments/fine_to_coarse_hierarchy/hierarchy.py`
- Medium reference cut: approximately 50% of Fine node count
- Coarse reference cut: approximately 25% of Fine node count

The reference cuts are visualization/operating cuts, not claimed natural semantic levels. No thresholds or hierarchy selection rules were redesigned for v0.2.

## Input selection and download

The runner parses the user-provided `D:/ThesisData/EgoPolice_1.0.0/video.txt`, keeps YouTube rows with explicit non-negative start/end intervals, and chooses the closest interval to 300, 600, 1200, and 1800 seconds among preferred official police-channel prefixes. Source URLs are unique across selected clips and ties use `sample_id`.

`yt-dlp --download-sections` plus ffmpeg force-keyframe trimming requests only each dataset interval. Every result is checked with ffprobe for duration, stream metadata, audio presence, and midpoint decode sanity. Unavailable links are recorded and replaced by the next deterministic candidate. Downloaded videos stay in the ignored experiment output and are not repository data.

## Cold processing and accounting

Each new clip is sampled at 1 FPS, encoded once with the local frozen DINOv2 ViT-S/14 model, segmented using the existing CoMET-style parameters, and merged using the existing Safe Merge implementation. Runtime separates download/trim, decode/frame sampling, cold feature extraction, Fine segmentation, hierarchy construction, and HTML rendering. Temporal compression is not presented as compute reduction.

## Metrics and review

Fine, Medium, and Coarse entity counts and duration statistics are reported with counts below 4 seconds and above 20, 45, 90, and 180 seconds. Giant parents (at least half the clip) and child-imbalance chaining risks are structural flags only. The main artifact is `outputs/experiments/long_video_hierarchy_stress_v0_2/hierarchy_review.html`, which presents aligned timelines and expandable Coarse → Medium → Fine image-backed groupings with empty manual-review controls.

## Safety

- External/model API calls: 0
- Gold/options/questions: not used
- Canonical Ours-v0.1: unchanged
- Downloaded videos/model weights/features: not intended for Git
- Results are not integrated into the canonical pipeline

## Executed sample and results

The deterministic primary metadata choices were 305, 600, 1231, and 1803 seconds. The 600-second primary URL was unavailable and two closer replacements were age restricted; the resulting 570-second clip was the fourth deterministic attempt. The 1803-second primary URL was age restricted; the next usable candidate was 1428 seconds. The other final intervals were 305 and 1231 seconds. All four outputs passed ffprobe and midpoint decode validation and retained audio.

| Actual duration | Fine | Medium | Coarse | Largest Fine | Largest Medium | Largest Coarse | Largest Coarse ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 305.0s | 23 | 12 | 6 | 46s | 104s | 169s | 55.4% |
| 570.0s | 33 | 17 | 9 | 100s | 226s | 281s | 49.3% |
| 1231.0s | 97 | 49 | 25 | 84s | 142s | 272s | 22.1% |
| 1428.1s | 142 | 71 | 36 | 43s | 201s | 436s | 30.5% |

Across the four clips, 295 Fine leaves were preserved exactly (100%). The mechanical diagnostic marked one giant reference-cut parent and 13 child-imbalance/chaining-risk nodes. These flags identify intervals for human inspection; they are not semantic quality judgments.

Cold DINOv2 extraction took 19.85 seconds total after 15.32 seconds of 1 FPS decode/frame sampling. Fine segmentation took 0.013 seconds and hierarchy construction 0.594 seconds. Interval download and trim took 221.86 seconds. The model was loaded once on an NVIDIA GeForce RTX 3060 Laptop GPU.
