# C-RADIOv4 follow-up diagnostics

Date: 2026-07-22  
Branch/commit: `exp/dull-baseline-qwen7b` / `3fcf1de`  
Status: completed exploratory diagnostics  
Scope: **not a formal B0–B4 result**

## Executive conclusion

Action-specific text materially improved temporal exposure on this five-question,
one-video diagnostic. On the four questions with a concrete correct option, the
post-hoc correct-option upper bound improved GT Interval Hit@5/8/10/20 from
`0/1/1/1 of 4` for the generic query to `2/3/3/3 of 4`; median first-inside
rank improved from 58.5 to 5.5. The deployable, GT-independent balanced merge
over all concrete options reached `1/2/2/3 of 4`, with median rank 12.5.

C-RADIO and DINOv2 produced similar segment counts under an unchanged isolated
replay (Fine 126 versus 134; Medium 40 versus 46), but their adjacent-cosine
scales differed materially: smoothed-similarity MAD differed by 8.49×. The
same absolute floor therefore has different effective stringency. No threshold
was retuned. Boundary agreement was moderate (F1 0.636 within ±2 seconds), not
evidence that C-RADIO can safely replace the frozen DINOv2 structure encoder.

The best-supported interpretation is **A: retain DINOv2 for temporal structure
and consider C-RADIO for semantic retrieval**. This is a research direction,
not a B1 definition change.

## Experimental boundaries

Both experiments used only full source video `pasadena/YKI08`, duration
1611.584 seconds, and the deterministic 1 FPS timestamps 0 through 1611.
Existing C-RADIO embeddings were reused; they were not recomputed. No Qwen,
B0, or formal B1/B2 run occurred. The frozen manifests, visual hierarchy,
DINOv2 pipeline outputs, and formal experiment definitions were not modified.

GT intervals and correct answers were loaded only after all 25 option rankings
had been generated and saved. Every Hit@K value below is **GT Interval Hit@K**:
at least one ranked timestamp is in `[start,end)`. It is coarse temporal
exposure, not proof that decisive evidence is visible.

## Exact C-RADIO setup

| Field | Value |
|---|---|
| Model | `nvidia/C-RADIOv4-SO400M` |
| Checkpoint | `c-radio_v4-so400m_half.pth.tar` |
| Checkpoint revision | `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8` |
| Checkpoint SHA256 | `d02697ede20f2716c4db12a9d3dab1c5a9b47d15a16cdcd46a69d08062b77aaf` |
| Official implementation | `NVlabs/RADIO` revision `c0f37017930e9dda53f93424cf4bf39fc51f287e` |
| Text adaptor | Official `siglip2-g` |
| Text model | `google/siglip2-giant-opt-patch16-384` revision `a713301b217d38485fb2204c808367d10bc3cc40` |
| Existing visual representation | 1,612 × 1,536, float16, L2-normalized |
| Visual embedding compute | BF16 CUDA autocast |
| Option-query text compute | FP32 CPU |
| Query template | `Question: {question}\nCandidate answer: {option}` |

The option diagnostic ran on CPU because the GPU was occupied at that time.
The previous generic ranking used BF16 GPU text encoding. This dtype/device
difference is a limitation of the direct ranking comparison, although the
underlying stored visual embeddings and official adaptor are identical.

## Experiment A — option-semantic retrieval

All five options for each of the five questions were independently encoded.
`None of the above` was retained in raw rankings but excluded from the
concrete-option candidate merge because it has no positive visual semantics.
The all-options method used deterministic round-robin allocation by ascending
option index and within-option rank, skipping duplicate timestamps, to enforce
fixed total budgets of 1, 5, 8, 10, and 20.

### Direct query-formulation comparison

The fairest direct comparison uses the same four-question cohort whose correct
option has concrete visual semantics.

| Query formulation | n | Hit@1 | Hit@5 | Hit@8 | Hit@10 | Hit@20 | Median first-inside rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Generic question only | 4 | 0/4 | 0/4 | 1/4 | 1/4 | 1/4 | 58.5 |
| Correct-option semantic query — **post-hoc diagnostic upper bound** | 4 | 0/4 | 2/4 | 3/4 | 3/4 | 3/4 | 5.5 |
| All concrete options, balanced fixed budget — deployable diagnostic | 4 | 0/4 | 1/4 | 2/4 | 2/4 | 3/4 | 12.5 |

Across all five questions, including the question whose GT is `None of the
above`, generic Hit@1/5/8/10/20 was `0/0/1/1/1 of 5`; balanced retrieval was
`0/1/2/2/3 of 5`, with median first-inside rank improving from 89 to 17.

The correct-option result is not deployable: it uses the correct answer only
after rankings are frozen to select which option ranking to score. It is an
answer-conditioned upper bound asking whether a meaningful action phrase can
localize its interval. The all-options result is GT-independent and is the
more relevant candidate-generation diagnostic.

### Per-question results

| Question | GT interval | Correct option | Correct-option H@1/5/8/10/20 | Correct first rank | Balanced H@1/5/8/10/20 | Balanced first rank |
|---|---|---|---|---:|---|---:|
| `1s_3455` | [1104,1105) | handcuffs | 0/0/1/1/1 | 8 | 0/0/0/0/1 | 17 |
| `1s_3447` | [1093,1094) | handcuffs | 0/0/0/0/0 | 79 | 0/0/0/0/0 | 152 |
| `10s_4776` | [90,100) | None of the above | not semantically interpretable | 647 | 0/0/0/0/0 | 153 |
| `10s_3437` | [1080,1090) | handcuffs | 0/1/1/1/1 | 3 | 0/0/1/1/1 | 8 |
| `60s_1377` | [1080,1140) | handcuffs | 0/1/1/1/1 | 3 | 0/1/1/1/1 | 4 |

For all questions, the top-ranked timestamp was 1515 seconds. Its distances to
the respective intervals were 410, 421, 1415, 425, and 375 seconds. Option
semantics improved later ranks but did not fix Top-1.

### Incorrect-option rank statistics

The following compact table reports first in-interval rank for every option;
`*` marks the correct option and `N` marks `None of the above`. Parenthesized
`H20` means that option's Top-20 contains an in-interval timestamp.

| Question | Option 0 | Option 1 | Option 2 | Option 3 | Option 4 |
|---|---:|---:|---:|---:|---:|
| `1s_3455` | 111 | 770 | **8* (H20)** | 387 | 317 N |
| `1s_3447` | 596 | **79*** | 1190 | 845 | 1455 N |
| `10s_4776` | 108 | 78 | 185 | 116 | **647* N** |
| `10s_3437` | 122 | 17 (H20) | 49 | **3* (H20)** | 37 N |
| `60s_1377` | **3* (H20)** | 15 (H20) | 2 (H20) | 10 (H20) | 13 N (H20) |

The 60-second interval naturally admits many more 1 FPS timestamps, so its
high exposure rates must not be interpreted as equally difficult to the 1s or
10s cases. Full per-option Hit@K values and all rankings are preserved in the
JSON/CSV artifacts.

### Qualitative rankings

The concrete correct option for four questions is the same handcuff action, so
its C-RADIO Top-10 ranking is the same (minor floating-point score variation
only):

| Rank | Timestamp | Cosine score |
|---:|---:|---:|
| 1 | 1515 | 0.186230 |
| 2 | 1313 | 0.165216 |
| 3 | 1086 | 0.164040 |
| 4 | 1103 | 0.161953 |
| 5 | 1511 | 0.161866 |
| 6 | 1312 | 0.160180 |
| 7 | 1514 | 0.159495 |
| 8 | 1104 | 0.158278 |
| 9 | 1496 | 0.157850 |
| 10 | 1510 | 0.157625 |

Thus 1086 seconds is rank 3 and exposes both `[1080,1090)` questions, while
1104 seconds is rank 8 and exposes `[1104,1105)`. The neighboring 1-second
interval `[1093,1094)` does not appear until rank 79, demonstrating that a
semantically relevant cluster is not precise enough for every short action.

Balanced fixed-budget Top-10 timestamps were:

| Question | Balanced Top-10 timestamps (seconds) |
|---|---|
| `1s_3455` | 1515, 1512, 1396, 1313, 1340, 1511, 1086, 1123, 1103, 1514 |
| `1s_3447` | 1515, 1514, 1313, 1340, 1111, 1512, 1086, 1112, 1511, 1103 |
| `10s_4776` | 1515, 1396, 1397, 1111, 1514, 1511, 1331, 1112, 1512, 1123 |
| `10s_3437` | 1515, 1396, 1111, 1512, 1313, 1511, 1112, **1086**, 1123, 1337 |
| `60s_1377` | 1515, 1313, 1397, **1111**, 1340, **1086**, 1331, **1112**, **1103**, 1396 |

The raw `None of the above` query ranked 1515, 1510, 1068, 1313, 1512,
1397, 1514, 866, 1496, and 1340 seconds. It is retained only for completeness
and has no reliable positive semantic interpretation.

### Experiment A interpretation

The large gain from generic to action-specific queries shows that the earlier
question-only failure was mainly a query-information limitation. However,
Top-1 remained 0/4, one 1-second case remained beyond rank 20, and the sample is
only four concrete-GT questions from one video. The balanced method improves
over generic retrieval but is weaker than the answer-conditioned upper bound.
This supports further controlled study of C-RADIO, not adoption as a formal B1
scorer.

## Experiment B — C-RADIO versus DINOv2

### Controlled setup

| Field | DINOv2 | C-RADIOv4 |
|---|---|---|
| Model/checkpoint | `facebook/dinov2-small` | `nvidia/C-RADIOv4-SO400M` |
| Revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` | `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8` |
| Parameters | 22,056,576 | 431,237,232 visual backbone |
| Feature | CLS token | official `siglip2-g` visual summary |
| Dimension/dtype on disk | 384 / float32 | 1536 / float16 |
| Input preprocessing | official AutoImageProcessor, shortest edge 256, center crop 224 | official diagnostic policy, 512×512 `[0,1]` tensor |
| Timestamps/source frames | exact same 1,612 timestamps and frame indices | exact same 1,612 timestamps and frame indices |
| Normalization | L2 | L2 |

Both representations were passed through the same unchanged CoMET-style
segmentation, Safe-Merge hierarchy, and Fluid-Loose Medium logic from
`config/visual_pipeline_v1.json`, in memory and under an isolated diagnostic
path. Frozen outputs were not overwritten. C-RADIO replay retains legacy
schema field names only for compatibility; they do not misidentify its
features as DINOv2.

### Adjacent-frame similarity

| Statistic | DINOv2 | C-RADIOv4 |
|---|---:|---:|
| Mean adjacent cosine | 0.797581 | 0.979667 |
| Median | 0.824188 | 0.985026 |
| Minimum | 0.168007 | 0.837616 |
| Standard deviation | 0.111810 | 0.017589 |
| Raw MAD | 0.056181 | 0.006066 |
| IQR | 0.115737 | 0.014027 |
| Smoothed MAD used by detector | 0.047190 | 0.005558 |
| Adaptive prominence | 0.023595 | 0.005000 floor |

C-RADIO features are much more tightly clustered for adjacent frames. The
smoothed MAD ratio is 8.49×. Although both are normalized cosine features and
the same settings can be executed, the fixed absolute `minimum_prominence =
0.005` is merely a floor for DINOv2 but binds C-RADIO. Therefore direct
threshold equivalence is not established. The diagnostic reports this
incompatibility and does not silently tune either threshold.

### Segmentation and fragmentation

| Metric | DINOv2 | C-RADIOv4 |
|---|---:|---:|
| Fine segments | 134 | 126 |
| Fine boundaries/minute | 4.952 | 4.654 |
| Fine mean / median duration | 12.03 / 10.00 s | 12.79 / 11.00 s |
| Fine maximum duration | 65 s | 45 s |
| Fine segments <2 s | 0 | 1 |
| Fine segments <4 s | 1 | 1 |
| Fine segments <5 s | 5 | 4 |
| Medium segments | 46 | 40 |
| Medium mean / median duration | 35.03 / 11.00 s | 40.29 / 11.00 s |
| Medium maximum duration | 208 s | 218 s |
| Medium segments <4 s | 1 | 1 |

Counts and fragmentation are broadly similar on YKI08; C-RADIO did not cause
obvious extra fragmentation. This alone does not establish segmentation
quality because there is no independent ground-truth boundary set.

### Boundary agreement

| Tolerance | Matches | DINO boundaries | C-RADIO boundaries | Precision vs DINO | Recall vs DINO | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| exact | 31 | 133 | 125 | 0.248 | 0.233 | 0.240 |
| ±1 s | 67 | 133 | 125 | 0.536 | 0.504 | 0.519 |
| ±2 s | 82 | 133 | 125 | 0.656 | 0.617 | 0.636 |
| ±5 s | 99 | 133 | 125 | 0.792 | 0.744 | 0.767 |

The secondary rank-matched comparison, using 125 strongest boundaries from
each representation without changing frozen thresholds, yielded F1 0.496,
0.616, and 0.744 at ±1, ±2, and ±5 seconds. It reaches the same overall
conclusion: substantial overlap but meaningful representation-specific
boundaries.

Qualitative contact sheets were saved for the strongest DINO boundary at
1094 s (nearest C-RADIO boundary 1 s), the strongest C-RADIO boundary at 1345
s (exact DINO agreement), a strong DINO-only case at 369 s (nearest C-RADIO 7
s), and a strong C-RADIO-only case at 186 s (nearest DINO 6 s).

### Efficiency

| Metric | DINOv2-small | C-RADIOv4-SO400M |
|---|---:|---:|
| Embedding wall time | 250.422 s | 50.589 s |
| End-to-end throughput | 6.437 FPS | 31.865 FPS |
| Decoder time | 43.662 s | 13.292 s |
| Preprocessing time | 204.631 s | 3.074 s |
| GPU inference time | 2.104 s | 34.187 s |
| Peak allocated VRAM | 292,911,616 B (0.273 GiB) | 4,992,640,000 B (4.65 GiB) |
| Compressed artifact | 2,300,095 B (2.19 MiB) | 4,540,678 B (4.33 MiB) |
| Compressed bytes/frame | 1,426.86 | 2,816.80 |
| Raw bytes/frame | 1,536 | 3,072 |

The measured end-to-end DINO throughput is dominated by the official PIL-based
CPU processor (204.6 s); its actual GPU forward time is only 2.1 s, versus
34.2 s for C-RADIO. Therefore the 4.95× end-to-end advantage shown by the
current C-RADIO implementation is not an intrinsic encoder-speed advantage.
DINO is far cheaper in inference and VRAM, while C-RADIO remained practical on
the 3090 Ti and used only about twice the embedding storage.

## Errors and fixes

- The first DINO attempt stopped before embedding because PyTorch 2.13 rejected
  a `torch.device` argument in the CUDA memory-statistics API. The isolated
  runner now sets the intended current device and uses current-device memory
  statistics, matching the compatibility approach used elsewhere in the repo.
- Two later attempts completed DINO embedding but failed while serializing
  Transformers 5.14.1 `SizeDict` processor metadata. The compatibility helper
  now recursively converts dataclass wrappers to plain JSON dictionaries and
  writes a metadata sidecar immediately after embedding.
- The final reported DINO timing is from the successful uncontended run only:
  preflight showed 269 MiB used, 23,844 MiB free, and 1% GPU utilization.
- No C-RADIO visual embeddings were recomputed during these follow-ups.

## Interpretation matrix

- **A — best fit.** DINOv2 remains the safer structural representation because
  it is the frozen reference and C-RADIO's similarity scale is materially
  different. C-RADIO adds native option/action semantics and improved retrieval.
- B is not yet justified: similar counts on one video are insufficient to
  establish a unified replacement, especially with an 8.49× MAD difference.
- C is not supported: action-specific and balanced retrieval both improved
  materially over the generic query.
- D is not supported on this host: 4.65 GiB peak and 31.9 FPS offline throughput
  are practical, although C-RADIO is much heavier than DINO at inference.

No architecture or formal experiment definition changes automatically follow
from this exploratory result.

## Artifacts

Option-semantic outputs:

- `outputs/diagnostics/cradio_v4/option_semantic/option_rankings_pre_gt.{json,npz}`
- `outputs/diagnostics/cradio_v4/option_semantic/option_semantic_results.json`
- `outputs/diagnostics/cradio_v4/option_semantic/option_rank_statistics.csv`
- `outputs/diagnostics/cradio_v4/option_semantic/query_formulation_comparison.csv`

DINO comparison outputs:

- `outputs/diagnostics/cradio_v4/dino_comparison/dino_comparison_results.json`
- `outputs/diagnostics/cradio_v4/dino_comparison/dinov2_embedding_metadata.json`
- `outputs/diagnostics/cradio_v4/dino_comparison/{dinov2,cradio}_replay.json`
- `outputs/diagnostics/cradio_v4/dino_comparison/YKI08_1fps_dinov2_embeddings.npz`
- `outputs/diagnostics/cradio_v4/dino_comparison/{representation_efficiency,segmentation_summary,adjacent_similarity_summary,boundary_agreement,qualitative_boundary_examples}.csv`
- `outputs/diagnostics/cradio_v4/dino_comparison/qualitative_boundaries/*.jpg`

Code and tests:

- `scripts/diagnostics/cradio_v4/run_option_semantic.py`
- `scripts/diagnostics/cradio_v4/run_dino_comparison.py`
- `src/diagnostics/cradio_v4/option_semantic.py`
- `src/diagnostics/cradio_v4/dino_comparison.py`
- `tests/diagnostics/cradio_v4/test_option_semantic.py`
- `tests/diagnostics/cradio_v4/test_dino_comparison.py`

## Validation and repository state

The focused diagnostic suite passed: 9 tests. It covers option-query
construction, deterministic balanced merging, GT-independent pre-ranking,
boundary matching, distribution summaries, and Transformers metadata
serialization compatibility. JSON/CSV artifacts were parsed after completion;
the DINO and C-RADIO timestamp grids contain the same 1,612 entries.

At report generation, `git status --short` was:

```text
 M docs/EXPERIMENT_LOG.md
?? docs/reports/
?? scripts/diagnostics/cradio_v4/
?? src/diagnostics/cradio_v4/
?? tests/diagnostics/cradio_v4/
```

No commit or push was performed.
