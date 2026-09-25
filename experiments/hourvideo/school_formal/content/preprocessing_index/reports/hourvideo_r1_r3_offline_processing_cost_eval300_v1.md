# R1 and R3 offline-processing cost on the exact GenS 12-video subset

Audit date: 2026-08-27  
Dataset: HourVideo  
Subset: the exact 12 `video_id` values used by the GenS 300-question evaluation  
Purpose: thesis-ready accounting of the corpus-level R1 and latest R3 offline indexes

## 1. Definitions of R1 and R3

### 1.1 R1 in this document

**R1 is the question-independent object and tracking index.** It is the offline index produced by `hourvideo_dev50_r1_offline_v1`; it is not the query-time R1 Planner, evidence-selection or final-answer route.

For each video, R1:

1. reuses the shared 1-FPS frame cache;
2. reuses the shared 15-second Fine / 45-second Medium hierarchy and Medium SigLIP embeddings;
3. runs YOLOv8x-OIV7 with BoT-SORT over every retained 1-FPS frame;
4. stores frame-level object detections and local tracking observations;
5. deterministically projects those observations into the 45-second Medium nodes.

R1 does **not** generate visual captions, does not use VideoSEAL artifacts, and makes no external API calls. The isolated audio/ASR diagnostic found elsewhere in the 50-video experiment belongs to a video outside this 12-video population and is excluded.

### 1.2 Latest R3 in this document

**R3 is the question-independent hierarchical semantic caption index.** “Latest R3” refers specifically to `hourvideo_r3_keyframe_caption_3frame_action_preserving_v1`; it is not an older R3/R3.2 live reasoning pipeline.

For each video, R3:

1. reuses the shared 1-FPS frame cache;
2. reuses the shared 15-second Fine / 45-second Medium hierarchy and Fine/Medium SigLIP embeddings;
3. selects at most one chronological representative frame from each Fine node, normally three images per Medium node;
4. uses local Qwen2.5-VL-7B-Instruct to generate one action-preserving caption per Medium node;
5. sends the complete ordered Medium-caption sequence to one Claude Haiku 4.5 Organizer call per video;
6. produces the Coarse navigation groups and the Medium-to-Coarse map.

Fine-node captioning, audio/ASR, question content, Planner and Inspector are disabled during R3 offline construction. Qwen captioning is local; only the Organizer is a paid external API stage.

### 1.3 Shared base versus method-specific work

R1 and R3 share the following preprocessing:

```text
source video
  -> 1-FPS frame extraction
  -> per-frame SigLIP embedding
  -> fixed 15-second Fine hierarchy
  -> fixed 45-second Medium hierarchy and pooled Medium embeddings
```

After that shared base, they diverge:

```text
R1 -> YOLO/BoT-SORT -> frame observations -> Medium object projection
R3 -> representative frames -> Qwen Medium captions -> Haiku Coarse Organizer
```

The full-build totals include the shared base once. Incremental totals assume that the shared frames, hierarchy and SigLIP embeddings already exist.

## 2. Scope and accounting rules

Every number below belongs to the exact 12-video population; no 50-video total is scaled down.

The subset contains:

- 12 videos;
- 37,562.563001 seconds of source video = **10.434045 video-hours**;
- 37,561 retained 1-FPS frames;
- 300 downstream QA examples, exactly 25 per video.

The offline indexes are generated per video and reused across questions. Per-question values are therefore amortized corpus-level values, not per-question execution costs.

Measurement labels:

- **Recorded**: read directly from a final processing manifest or per-video cost file.
- **Recomputed**: summed or counted from the exact 12 final artifacts.
- **Derived**: arithmetic combination of recorded quantities.
- **Conditional estimate**: depends on a machine-hour allocation assumption.

The R1/R3 totals combine historical stages run at different times and on different machines. They are sums of recorded processing components, not a newly measured single-run end-to-end wall time.

Unless stated otherwise, storage uses decimal units: 1 MB = 1,000,000 bytes and 1 GB = 1,000,000,000 bytes.

## 3. Exact video population

1. `115774b6-534d-444f-b7aa-d1b834eb0ee7`
2. `41a86310-2cc1-48f9-b5b5-6b495a95fbac`
3. `4572b198-2c1c-4920-bcf0-95fcebe12261`
4. `6fd90f8d-7a4d-425d-a812-3268db0b0342`
5. `70f2a750-f403-41b8-aabb-480eb3ab4ed4`
6. `71fbc5bf-7e2a-415d-86bc-3a948742e904`
7. `7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b`
8. `7e512589-aa97-41e8-83d3-af2e83e4fd06`
9. `819c8af7-851f-434f-ab32-318285bc54b1`
10. `a6d45e95-8dc0-4932-83bf-ec53e265a16a`
11. `d3a0899e-2093-454c-9f65-30087883193a`
12. `db3f7933-dfa0-4678-9d4f-393b628ded45`

The 12 source MP4 files occupy **11,229,987,667 bytes = 11.229988 GB (10.458741 GiB)**. This is input-dataset storage and is excluded from generated-index storage.

## 4. Shared preprocessing

| Quantity | Exact result | Evidence type |
|---|---:|---|
| Source-video duration | 37,562.563001 s = **10.434045 h** | Recomputed |
| Retained 1-FPS frames | **37,561** | Recomputed |
| Frame-extraction elapsed | 1,000.396539 s = **16.673 min** | Recorded |
| SigLIP inputs | **37,561 frames** | Recomputed |
| SigLIP model | `google/siglip-base-patch16-224` | Recorded |
| SigLIP dimension / dtype | 768 / float16 | Recorded |
| SigLIP elapsed | 886.677223 s = **14.778 min** | Recorded |
| Weighted SigLIP throughput | **42.361526 FPS** | Derived |
| SigLIP OOM or batch retries | **0** | Recorded |
| Fine nodes | **2,509** | Recomputed |
| Medium nodes | **838** | Recomputed |
| **Shared recorded total** | **1,887.073762 s = 31.451 min** | Derived |

The frame-cache storage size is unavailable on the current machine and is excluded from both final-index storage totals.

## 5. R1 offline-processing result

| Quantity | Exact result | Evidence type |
|---|---:|---|
| Detector/tracker implementation | YOLOv8x-OIV7 + BoT-SORT | Recorded |
| Frames processed | **37,561** | Recomputed |
| Detector inference | 1,596.102763 s | Recorded |
| Detector model loading | 2.413151 s | Recorded |
| **R1-specific recorded total** | **1,598.515914 s = 26.642 min** | Derived |
| Weighted detector throughput | **23.532946 FPS** | Derived |
| Object observations | **101,247** | Recomputed |
| Medium object projections | **838** | Recomputed |
| Peak detector GPU memory | 367,689,216 bytes = **0.368 GB** | Recomputed |
| External API calls / cost | **0 / US$0** | Recorded |
| **Full R1 recorded total** | **3,485.589676 s = 58.093 min** | Derived |
| Final R1 index files | **71,001,828 bytes = 71.001828 MB** | Recomputed |

The 58.093-minute total is:

```text
frame extraction       1,000.396539 s
Fine SigLIP              886.677223 s
R1 model loading           2.413151 s
R1 detector inference   1,596.102763 s
------------------------------------------------
recorded total          3,485.589676 s
```

Hierarchy construction, Medium pooling, observation projection, serialization and validation did not retain independent wall timers. Therefore, **58.093 minutes is a measured-component lower bound**, not a fully instrumented end-to-end wall time.

### 5.1 R1 generated storage

| Artifact class | Bytes | Decimal size |
|---|---:|---:|
| Frame-level tracking observations | 53,063,538 | 53.063538 MB |
| Detector job and timing telemetry | 9,832,747 | 9.832747 MB |
| Medium object projections | 3,984,755 | 3.984755 MB |
| Medium SigLIP embeddings | 2,575,872 | 2.575872 MB |
| Shared hierarchy | 1,502,788 | 1.502788 MB |
| Remaining validation/configuration metadata | 42,128 | 0.042128 MB |
| **Total** | **71,001,828** | **71.001828 MB** |

## 6. Latest R3 offline-processing result

### 6.1 Local Medium-caption stage

| Quantity | Exact result | Evidence type |
|---|---:|---|
| Caption model | `Qwen/Qwen2.5-VL-7B-Instruct` | Recorded |
| Medium caption calls | **838** | Recomputed |
| Representative images | **2,509** | Recomputed |
| Input tokens | **1,384,214** | Recorded |
| Output tokens | **27,091** | Recorded |
| Caption batch wall time | **1,146.382967 s = 19.106 min** | Recorded |
| Peak GPU memory | 17,131,590,144 bytes = **17.132 GB** | Recomputed |
| External API calls / cost | **0 / US$0** | Recorded |

### 6.2 Coarse Organizer stage

| Quantity | Exact result | Evidence type |
|---|---:|---|
| Organizer model | `claude-haiku-4-5-20251001` | Recorded |
| API calls | **12** | Recomputed |
| Medium inputs | **838** | Recomputed |
| Coarse outputs | **252** | Recomputed |
| Input tokens | **62,585** | Recorded |
| Output tokens | **18,206** | Recorded |
| Summed case elapsed | **172.762222 s = 2.879 min** | Recomputed |
| Input-token cost | US$0.062585 | Derived using frozen run pricing |
| Output-token cost | US$0.091030 | Derived using frozen run pricing |
| **Organizer API cost** | **US$0.153615** | Derived using frozen run pricing |

The frozen run used US$1 per million input tokens and US$5 per million output tokens. This document reports the price recorded by the experiment rather than retroactively applying a later price.

### 6.3 R3 total

| Quantity | Exact result | Evidence type |
|---|---:|---|
| Shared extraction + SigLIP | 1,887.073762 s | Derived |
| R3 caption + Organizer | 1,319.145189 s | Derived |
| **Full R3 recorded total** | **3,206.218951 s = 53.437 min** | Derived |
| External API cost | **US$0.153615** | Recorded/derived |
| Final R3 index files | **72,521,409 bytes = 72.521409 MB** | Recomputed |

The 53.437-minute total is:

```text
frame extraction       1,000.396539 s
Fine SigLIP              886.677223 s
Qwen Medium captions   1,146.382967 s
Haiku Organizer          172.762222 s
------------------------------------------------
recorded total          3,206.218951 s
```

Hierarchy construction, pooling, artifact copying and validation did not retain independent wall timers. Therefore, **53.437 minutes is a measured-component lower bound**, not a fully instrumented end-to-end wall time.

### 6.4 R3 generated storage

| Artifact class | Bytes | Decimal size |
|---|---:|---:|
| Fine SigLIP embeddings | 65,065,036 | 65.065036 MB |
| Medium SigLIP embeddings | 2,575,872 | 2.575872 MB |
| Shared hierarchy | 1,502,788 | 1.502788 MB |
| Medium captions and caption telemetry | 2,022,947 | 2.022947 MB |
| Organizer artifacts and navigation maps | 1,324,041 | 1.324041 MB |
| Fine metadata and case configuration | 30,725 | 0.030725 MB |
| **Total** | **72,521,409** | **72.521409 MB** |

R1 and R3 package shared artifacts differently: R3 copies the Fine SigLIP shards into its frozen index, whereas R1 retains only the Medium embeddings needed by its runtime map. Their final-directory sizes therefore should not be interpreted as equal underlying representations.

## 7. Direct comparison

| Quantity | R1 | Latest R3 |
|---|---:|---:|
| Shared preprocessing | 31.451 min | 31.451 min |
| Method-specific processing | 26.642 min | 21.986 min |
| **Full recorded processing** | **58.093 min** | **53.437 min** |
| Primary semantic product | Object/tracking observations | Action-preserving temporal captions |
| Final temporal product | 838 Medium object projections | 838 Medium captions + 252 Coarse nodes |
| Local VLM calls | 0 | 838 |
| Paid API calls | 0 | 12 |
| Paid API cost | **US$0** | **US$0.153615** |
| Final index files | 71.002 MB | 72.521 MB |

R1 and R3 are complementary rather than interchangeable. R1 records what object categories were detected and where they persisted; it deliberately does not infer actions, roles, ownership or intent. R3 provides action-oriented temporal descriptions and coarse navigation summaries, but does not preserve dense per-frame object detections.

## 8. Normalized and amortized values

| Quantity | R1 | Latest R3 |
|---|---:|---:|
| Recorded processing per source-video hour | **5.568 min** | **5.121 min** |
| Mean recorded processing per video | **4.841 min** | **4.453 min** |
| Amortized processing per question | **11.619 s** | **10.687 s** |
| Final storage per video | 5.917 MB | 6.043 MB |
| Final storage per question | 0.237 MB | 0.242 MB |
| API cost per source-video hour | US$0 | US$0.014722 |
| API cost per video | US$0 | US$0.012801 |
| Amortized API cost per question | US$0 | US$0.000512 |

## 9. Monetary-cost formulas

No authoritative local machine-hour price was retained. Infrastructure and API costs must therefore remain separate.

```text
R1 offline cost
  = 0.277888 x frame-extraction-node rate
  + 0.246299 x SigLIP-GPU rate
  + 0.444032 x detector-GPU rate
  + US$0 external API cost
```

```text
R3 offline cost
  = 0.277888 x frame-extraction-node rate
  + 0.246299 x SigLIP-GPU rate
  + 0.318440 x Qwen-caption-GPU rate
  + US$0.153615 Organizer API cost
```

The GPU-hour conversions are conditional estimates because the recorded intervals include preprocessing, decoding, I/O and framework overhead in addition to model kernels.

## 10. Included and excluded costs

Included:

- the exact 12-video 1-FPS extraction time;
- full per-frame SigLIP computation;
- R1 detector inference and model loading;
- R1 observations and Medium projections;
- R3 local-Qwen caption time and tokens;
- R3 Organizer time, tokens and API cost;
- final R1 and R3 index files.

Excluded:

- original MP4 storage from generated-index storage;
- 1-FPS JPEG storage, because that cache is not present on the current machine;
- reusable model weights and generic caches;
- uninstrumented hierarchy, pooling, copy, projection, serialization and validation time;
- question-time planning, retrieval, inspection and final-answer inference;
- the separate R1 audio diagnostic outside the 12-video population;
- failed diagnostics and unrelated experimental variants;
- energy consumption and local compute dollar cost, because no power telemetry or machine rate was recorded.

## 11. Thesis-ready concise statements

> On the exact 12-video GenS evaluation subset (10.434 h of source video), R1 denotes the question-independent object/tracking index built from shared 1-FPS SigLIP features plus YOLOv8x-OIV7/BoT-SORT observations projected into 45-second Medium nodes. Its recorded processing components total 3,485.590 seconds (58.093 minutes), including 31.451 minutes of shared preprocessing and 26.642 minutes of R1-specific detection. R1 produced 101,247 object observations, occupied 71.002 MB excluding the shared frame cache, and incurred no external API charge.

> On the same subset, the latest R3 denotes the question-independent hierarchical semantic index built from shared 1-FPS SigLIP features, 838 local Qwen2.5-VL-7B Medium captions generated from 2,509 representative images, and 12 Haiku Organizer calls producing 252 Coarse navigation nodes. Its recorded processing components total 3,206.219 seconds (53.437 minutes). The frozen index occupies 72.521 MB excluding the shared frame cache, and the recorded external API charge is US$0.153615.

## 12. Evidence and reproducibility sources

- Frame extraction audit: `${HOURVIDEO_ROOT}/manifests/dev50_frames_1fps_audit.json`
- Fine SigLIP shards: `${HOURVIDEO_ROOT}/hourvideo_frame_embeddings_3way_v1/siglip/shards`
- R1 implementation: `scripts/experiments/run_hourvideo_dev50_r1_offline_v1.py`
- R1 final outputs: `outputs/experiments/hourvideo_dev50_r1_offline_v1`
- Latest R3 implementation: `src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1/pipeline.py`
- Latest R3 frozen configuration: `src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1/config.json`
- R3 caption summary: `src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1/caption_summary.json`
- R3 Organizer summary: `src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1/organizer_final_summary.json`
- R3 frozen index: `src/experiments/hourvideo_r3_keyframe_caption_3frame_action_preserving_v1/index`
