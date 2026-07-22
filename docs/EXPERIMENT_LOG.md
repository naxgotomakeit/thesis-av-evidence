# EgoPolice experiment log

## 2026-07-22 — Oracle-5 short-clip smoke

- Branch: `exp/dull-baseline-qwen7b`
- Result: `outputs/diagnostics/oracle/egopolice_oracle_5.json`
- Model: local Qwen2.5-VL-7B-Instruct, BF16, no quantization
- Execution: one model load, five questions, five model calls
- Accuracy: 3/5 (60%)
- Mean/median per-query latency: 1.484/0.738 seconds
- One-time model loading: 1.757 seconds
- Peak allocated GPU memory: 18,752,840,704 bytes

Observation: both 10-second questions and one 1-second question were answered
correctly. The 60-second question and the second 1-second question were wrong.
This condition uses annotated short clips and diagnoses answer-model/perception
behavior when temporal context is supplied; it does not test long-video
evidence selection.

Limitation and conclusion: five questions from one source video are sufficient
only for pipeline smoke/debug. These values are **not formal benchmark
accuracy** and must not be used to select formal ablation videos or questions.

## 2026-07-22 — B0-5 long-video smoke

- Branch: `exp/dull-baseline-qwen7b`
- Result: `outputs/diagnostics/b0/egopolice_b0_5.json`
- Model: local Qwen2.5-VL-7B-Instruct, BF16, no quantization
- Execution: one model load, five questions, five model calls
- B0 input: full 436.5-second source, eight deterministic equal-bin midpoint
  frames per question, one Qwen call
- Accuracy: 0/5 (0%)
- GT Interval Hit@8: 2/5 (40%)
- Mean/median per-query latency: 2.360/2.220 seconds
- Mean frame extraction/inference latency: 1.275/1.041 seconds
- One-time processor/model loading: 2.049 seconds
- Peak allocated GPU memory: 17,197,247,488 bytes

Observation: two Oracle-correct cases were B0-wrong with no sampled timestamp
inside the annotated interval, consistent with likely sampling failure. A third
Oracle-correct case was B0-wrong even though one timestamp was inside the
interval; temporal interval overlap alone therefore did not ensure decisive
evidence exposure.

GT Interval Hit@8 only checks timestamp membership in the annotation interval.
It does not prove that a sampled frame shows the relevant action. The preserved
result JSON uses the legacy key `gt_hit_at_8` for this definition.

Limitation and conclusion: all five questions share one video and all B0
predictions were “None of the above.” This is a technical smoke/debug result,
**not formal reported benchmark accuracy**, and it was excluded from formal set
selection.

## 2026-07-22 — formal paired-set freeze

- Parent pool: frozen 50-video manifest, unchanged
- Excluded result-driven debug video: `pasadena/bMMuC`
- Formal pool: 20 videos, 530 associated questions
- Paired subset: 98 questions across all 20 videos
- Selection inputs: source/domain, parent duration lower-bound stratum,
  question IDs and duration classes only
- Model results, answer content, GT interval values, and download status used
  for selection: no

The exact rule, seed, parent hashes, selected IDs, and fallback cases are stored
in the two versioned manifests under `config/data/`.

## 2026-07-22 — formal-video acquisition/readiness attempt v1

- Run ID: `egopolice-ablation20-acquisition-v1`
- Branch/base commit: `exp/dull-baseline-qwen7b` / `7f58500`
- Video manifest SHA256:
  `0cb8d55962634d900d440f943a7d91ca5fe5473f3f5d0c096c826685acd3e1c5`
- Question manifest SHA256:
  `fca734b9764ed132483ba3858db34243b50384ba2f1e115197c15762adcb23a5`
- Configuration: exact frozen 20 targets; canonical data root; existing files
  probed first; resumable yt-dlp; two retries maximum; stop all further Vimeo
  requests after an authentication or rate-limit blocker
- Change/reason: data readiness only, required before formal B0; no model or
  experiment-definition change
- Result: 1/20 ready, 19/20 blocked, 0 missing-unattempted, 0 invalid
- Ready media: `pasadena/YKI08`, 1611.584 seconds, `gt_1200`, 1280×720,
  30 FPS, H.264 video, AAC audio, 670,287,000 bytes
- Blocker: the first Vimeo target returned HTTP 401 while yt-dlp attempted to
  fetch the Vimeo macOS OAuth token; the other 18 Vimeo targets were not
  requested
- Formal metrics: not run; no Qwen calls
- Artifacts:
  `outputs/data_audit/egopolice_ablation20_download_attempt.json`,
  `egopolice_ablation20_readiness.{json,csv}`,
  `egopolice_ablation98_pre_evaluation.json`, and
  `egopolice_ablation98_duration_distribution.csv`
- Conclusion: formal B0 is blocked because actual ffprobe duration bins are
  unavailable for 19 videos
- Limitation/anomaly: authenticated Vimeo cookies/credentials are not available
  in this execution environment; the 93 affected questions remain in an
  explicit `unavailable` duration-bin row and are not assigned using proxies
- Next step: resume the exact same downloader with legitimate authenticated
  Vimeo cookies, then regenerate the readiness audit; do not change manifests

## 2026-07-22 — C-RADIOv4 representation smoke

**Exploratory, not a formal B0–B4 result.** No Qwen calls were made and no
frozen experiment definition was changed.

- Video/questions: full `pasadena/YKI08` (1611.584 seconds); all five frozen
  YKI08 questions from `egopolice_ablation_questions_v1.json`
- Model: official `nvidia/C-RADIOv4-SO400M` checkpoint revision
  `c0457f5d`, official `NVlabs/RADIO` implementation revision `c0f37017`,
  BF16 CUDA autocast, official `siglip2-g` adaptor
- Offline representation: 1,612 deterministic 1 FPS frames at 512×512;
  1,536-dimensional float16 embeddings; 50.589 seconds (31.865 frames/s)
- Storage: 4,540,678 bytes compressed (2,816.8 bytes/frame)
- Peak allocated GPU memory: 4,992,640,000 bytes
- One-time cold model/adaptor load: 201.137 seconds, including first download
  and construction of the SigLIP2-Giant text model; checkpoint cache lookup was
  timed separately
- GT Interval Hit@1/5/8/10: 0/5, 0/5, 1/5, 1/5
- Result/artifacts: `outputs/diagnostics/cradio_v4/`
- Complete report: `docs/reports/CRADIO_V4_SMOKE_REPORT.md`

The official text adaptor worked, but all five dataset question strings are
identical ("Which action is happening in this video clip?"). Because the task
forbade using options, every case necessarily received the same ranking. The
single Top-8/10 hit is the 60-second interval containing the rank-7 timestamp
at 1109 seconds. These GT Interval metrics are coarse post-hoc temporal
exposure diagnostics, not true evidence recall.

Conclusion: compute, VRAM, and embedding storage are practical on the RTX 3090
Ti, so C-RADIOv4 remains a technically viable candidate representation. This
smoke does not establish it as a useful B1 scorer: generic question-only text
provided no discriminative retrieval signal, and no B1 definition is changed.
The DINOv2 control was skipped because the frozen DINOv2 component has no
existing text-aligned query method; no custom mapping was invented.

An initial load failed before embedding because the default user Hugging Face
cache exceeded its quota. The successful run fixed this by placing all large
diagnostic caches under the canonical project model storage. Transformers
5.14.1 also emitted BOS/EOS range warnings for the official SigLIP2 config;
inference completed without an error.

## 2026-07-22 — C-RADIOv4 option-semantic follow-up

**Exploratory diagnostic — not a formal B0–B4 result.** No Qwen calls were made
and no frozen manifest, pipeline, hierarchy, or experiment definition changed.

- Input: the existing 1,612-frame YKI08 C-RADIO 1 FPS embedding artifact; no
  visual embedding recomputation
- Query policy: all five options for each of the five frozen YKI08 questions
  were independently ranked before GT was loaded; `None of the above` was kept
  raw but excluded from concrete-option merging
- Concrete-GT cohort (n=4), generic Hit@1/5/8/10/20: 0/0/1/1/1; median first
  inside rank 58.5
- Correct-option post-hoc upper bound: 0/2/3/3/3; median rank 5.5
- GT-independent balanced all-options retrieval: 0/1/2/2/3; median rank 12.5
- All-five balanced result: 0/1/2/2/3; median rank 17
- Result: `outputs/diagnostics/cradio_v4/option_semantic/`

Conclusion: action semantics materially improve this diagnostic, showing that
the generic-question failure was mainly query-information limited. The
correct-option result is not deployable, and the five-question one-video result
does not define or validate formal B1.

## 2026-07-22 — C-RADIOv4 versus DINOv2 representation control

**Exploratory diagnostic — not a formal B0–B4 result.** The frozen DINOv2
hierarchy was replayed unchanged in isolation and no frozen output was
overwritten.

- Input: identical YKI08 source frames and 1,612 deterministic 1 FPS timestamps
- DINOv2: `facebook/dinov2-small`, 384-dimensional float32 CLS; 250.422 seconds
  end-to-end (6.437 FPS), 292,911,616-byte peak allocated VRAM, 2,300,095-byte
  artifact
- C-RADIO: existing 1,536-dimensional float16 `siglip2-g` summary; 50.589
  seconds (31.865 FPS), 4,992,640,000-byte peak, 4,540,678-byte artifact
- Frozen-logic Fine/Medium counts: DINO 134/46; C-RADIO 126/40
- Exact-threshold boundary agreement: F1 0.519 at ±1 s, 0.636 at ±2 s, and
  0.767 at ±5 s
- Smoothed adjacent-similarity MAD ratio: 8.49× (DINO 0.04719, C-RADIO
  0.00556); the same 0.005 absolute floor has different effective stringency,
  so no threshold was retuned
- Result: `outputs/diagnostics/cradio_v4/dino_comparison/`
- Full report: `docs/reports/CRADIO_V4_FOLLOWUP_REPORT.md`

Conclusion: interpretation A fits best for now—retain DINOv2 for frozen
temporal structure and consider C-RADIO for semantic retrieval. Similar segment
counts on one video do not yet justify a unified C-RADIO replacement. DINO's
measured end-to-end time was dominated by its official CPU preprocessing; its
GPU inference and VRAM costs remained much lower.
