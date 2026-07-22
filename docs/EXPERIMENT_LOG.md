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
