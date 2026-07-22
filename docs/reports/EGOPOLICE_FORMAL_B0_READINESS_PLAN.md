# EgoPolice formal-set readiness and B0 execution plan

Date: 2026-07-22  
Branch: `exp/dull-baseline-qwen7b`  
Inspected commit: `2c9336f` (working tree contains the audit/report preparation changes)  
Scope: readiness and execution planning only; no Qwen model was loaded and no inference was run.

## Frozen inputs

- Video manifest: `config/data/egopolice_ablation20_v1.json`
  - required and observed SHA256: `0cb8d55962634d900d440f943a7d91ca5fe5473f3f5d0c096c826685acd3e1c5`
- Question manifest: `config/data/egopolice_ablation_questions_v1.json`
  - required and observed SHA256: `fca734b9764ed132483ba3858db34243b50384ba2f1e115197c15762adcb23a5`
- Data root: `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0`
- Model: `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/Qwen2.5-VL-7B-Instruct`

Neither frozen manifest was modified.

## Reproducible audit method

Run from the canonical repository:

```bash
python scripts/data/audit_egopolice_ablation20_readiness.py \
  --data-root /cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0
```

The audit verifies both manifest hashes; exact path mapping; nonzero filesystem
and ffprobe size agreement; video/audio streams, codec, duration, resolution and
FPS; and a full-file ffprobe packet scan. Any packet-scan warning is a failure.
Reported duration must also cover the parent-manifest all-question metadata lower
bound and every frozen GT interval. Every frozen question is compared field by
field with its official `mcq_1s.json`, `mcq_10s.json`, or `mcq_60s.json` row:
ID, source video, question, five options, answer, and interval.

Result: **20/20 videos ready; 98/98 questions ready; no missing, zero-byte,
corrupt, packet-scan-warning, or obviously truncated file was found.** All 20
videos have H.264 video and AAC audio. Formal-video storage is 7,321,643,201
bytes (6.82 GiB).

## Per-video results

Question counts are shown as `1s/10s/60s`. Packet counts are `video/audio`.

| Video ID | Exact duration (s) | Bin | Resolution | FPS | Video | Audio | MiB | Packets | Questions | Strict audit |
|---|---:|---|---:|---|---|---|---:|---:|---:|---|
| `copa/1090234/285139179` | 130.603000 | `le_300` | 848×480 | `30000/1001` | h264 | aac | 25.49 | 3,914 / 6,122 | 2/1/2 | pass |
| `copa/2022-0002295/724873276` | 332.502000 | `301_600` | 1280×720 | `30000/1001` | h264 | aac | 119.65 | 9,965 / 15,586 | 2/2/1 | pass |
| `copa/2022-0003054/742435578` | 182.720000 | `le_300` | 1280×720 | `30000/1001` | h264 | aac | 65.18 | 5,476 / 8,565 | 2/2/1 | pass |
| `copa/2023-0001277/828449721` | 312.000000 | `301_600` | 1280×720 | `30/1` | h264 | aac | 112.35 | 9,360 / 14,625 | 2/2/1 | pass |
| `copa/2022-0005419/795285434` | 558.000000 | `301_600` | 1280×720 | `30/1` | h264 | aac | 200.67 | 16,740 / 26,156 | 2/2/1 | pass |
| `copa/2023-0000632/817305442` | 483.286000 | `301_600` | 1280×720 | `30/1` | h264 | aac | 172.64 | 14,498 / 22,654 | 2/2/1 | pass |
| `copa/2020-4833/492626546` | 475.734000 | `301_600` | 1280×720 | `30/1` | h264 | aac | 170.96 | 14,272 / 22,300 | 2/2/1 | pass |
| `copa/1082513/194071445` | 702.143333 | `601_1200` | 848×480 | `30000/1001` | h264 | aac | 139.48 | 21,040 / 32,913 | 2/2/1 | pass |
| `copa/2023-0001622/822714261` | 601.046000 | `601_1200` | 1280×720 | `30/1` | h264 | aac | 215.35 | 18,031 / 28,174 | 2/2/1 | pass |
| `copa/2021-2665/581221131` | 646.870000 | `601_1200` | 1280×720 | `30/1` | h264 | aac | 231.78 | 19,406 / 30,322 | 2/2/1 | pass |
| `copa/1086683/241541469` | 857.343333 | `601_1200` | 848×480 | `30000/1001` | h264 | aac | 168.06 | 25,692 / 40,188 | 1/1/1 | pass |
| `copa/2023-0005234/891233509` | 622.038000 | `601_1200` | 1280×720 | `30/1` | h264 | aac | 212.94 | 18,661 / 29,160 | 2/2/1 | pass |
| `copa/2020-3121/454773869` | 1102.550000 | `601_1200` | 1280×720 | `30/1` | h264 | aac | 395.99 | 33,076 / 51,682 | 2/2/1 | pass |
| `copa/2019-1617/347297049` | 1128.043000 | `601_1200` | 848×480 | `30000/1001` | h264 | aac | 213.92 | 33,807 / 52,877 | 2/2/1 | pass |
| `copa/2022-0001369/704211509` | 7585.878000 | `gt_1200` | 1280×720 | `30/1` | h264 | aac | 2285.90 | 227,576 / 355,588 | 2/2/1 | pass |
| `copa/2021-1076/540772815` | 1238.422000 | `gt_1200` | 1280×720 | `30000/1001` | h264 | aac | 444.15 | 37,115 / 58,051 | 2/2/1 | pass |
| `copa/1082645/193957196` | 1800.255000 | `gt_1200` | 848×480 | `30000/1001` | h264 | aac | 357.68 | 53,950 / 84,387 | 2/2/1 | pass |
| `copa/2020-4177/461538142` | 1597.739000 | `gt_1200` | 1280×720 | `30/1` | h264 | aac | 508.52 | 47,932 / 74,894 | 2/2/1 | pass |
| `pasadena/pAjat` | 859.968000 | `601_1200` | 1280×720 | `30000/1001` | h264 | aac | 302.53 | 25,773 / 40,311 | 2/2/1 | pass |
| `pasadena/YKI08` | 1611.584000 | `gt_1200` | 1280×720 | `30/1` | h264 | aac | 639.24 | 48,344 / 50,362 | 2/2/1 | pass |

## Frozen question distribution

| Full-source duration bin | Videos | Questions | 1s | 10s | 60s |
|---|---:|---:|---:|---:|---:|
| ≤300 s | 2 | 10 | 4 | 3 | 3 |
| 301–600 s | 5 | 25 | 10 | 10 | 5 |
| 601–1200 s | 8 | 38 | 15 | 15 | 8 |
| >1200 s | 5 | 25 | 10 | 10 | 5 |
| **Total** | **20** | **98** | **39** | **38** | **21** |

## Existing implementation inspection

The core B0 sampler and prompt remain correct and unchanged. Timestamps depend
only on ffprobe duration `D`: for `i = 0..7`, `t_i = (i + 0.5) × D / 8`.
`extract_uniform_frames` takes no question or GT interval, seeks each timestamp
with ffmpeg, returns exactly eight RGB images, and applies only the frozen
`max_pixels=262144` cap.

The existing generic runner is **not safe for 98-question formal execution as
written**: it enforces `limit <= 5`, truncates its JSONL at process start, loads
one raw MCQ metadata file rather than the frozen paired-question manifest, and
has no durable resume/progress contract. Its CLI also currently declares
`--model-root` twice. The YKI08 runner is explicitly non-formal and writes its
five records only in the final result. No Blind-control runner exists. These are
execution-layer blockers, not changes to the frozen B0 algorithm.

## Exact formal controls

### A. Blind control

No images are passed. Each user message contains this exact text template:

```text
Question: {question}

Options:
0. {option_0}
1. {option_1}
2. {option_2}
3. {option_3}
4. {option_4}

Return exactly one option index: 0, 1, 2, 3, or 4. Do not provide an explanation.
```

### B. Uniform-8 B0

Eight image content items precede this existing frozen prompt template:

```text
You are given 8 uniformly sampled frames from one full video in chronological order. Answer the multiple-choice question using only those frames.

Question: {question}

Options:
0. {option_0}
1. {option_1}
2. {option_2}
3. {option_3}
4. {option_4}

Return exactly one option index: 0, 1, 2, 3, or 4. Do not provide an explanation.
```

Both controls use local `Qwen2.5-VL-7B-Instruct`, BF16, no quantization, seed 0,
`do_sample=false`, `temperature=0.0`, `max_new_tokens=8`, and `use_cache=true`.
The processor chat template uses one user turn and adds the generation prompt.
Prediction parsing accepts only one unambiguous index 0–4 (or A–E compatibility);
parse failures are technical failures, never silently guessed.

## Formal execution plan

1. Add an isolated formal runner without altering `core.py`, the prompt, frame
   extraction, B0 configuration, or either manifest. It must verify both hashes,
   the 20/20 readiness JSON, local checkpoint, BF16 support, no quantization, and
   **at least 20 GiB free VRAM** before model construction.
2. Run Blind and Uniform-8 as separate named conditions over the same ordered 98
   question IDs. One Qwen call per question; one model load per uninterrupted
   condition/session; never reload between questions in that session.
3. Append and `fsync` one completed question immediately to
   `per_question_results.jsonl`. Atomically replace `progress.json` after each
   append. On resume, validate the record, manifest/config hashes and unique ID,
   then skip it; never truncate or recompute valid completed records.
4. For B0, cache only the deterministic timestamps/duration per source video.
   Decode the same eight timestamps for every question from that video; do not
   persist frames. Consult GT only after timestamps are fixed and inference is
   complete. Blind records frames as 0 and GT Interval Hit@8 as not applicable,
   not zero.
5. After exactly 98 valid records, aggregate overall, by GT-duration class, by
   full-video-duration bin, and their cross-tab. Every cell includes correct,
   total, accuracy and `n`. Mark the run formal only after 98/98 validation.

Planned isolated output roots:

- Blind: `outputs/experiments/Blind/formal_ablation98_v1/`
- Uniform-8: `outputs/experiments/B0/formal_ablation98_v1/`

Each contains `run_config.json`, `environment.json`, `progress.json`,
`per_question_results.jsonl`, `final_results.json`, `aggregate_summary.json`,
aggregate CSVs, and errors/warnings. The eventual human B0 report is
`docs/reports/B0_FORMAL_ABLATION98_V1_REPORT.md`.

Per question record: IDs and both duration classes; source duration; GT interval
and answer; prediction/correctness; exact timestamps; GT Interval Hit@8, frames
inside, nearest interval distance; extraction, preprocessing, inference and total
online latency; frames; calls; text/total input, estimated visual and output
tokens; peak VRAM; dtype, quantization and actual loading mode; OOM/errors; and
all fixed-zero B0 costs. Aggregates include accuracy, GT Interval Hit@8 and all
latency/token/call/frame metrics requested above. GT Interval Hit@8 remains a
coarse interval-exposure diagnostic, not proof that decisive evidence is visible.

## Runtime, storage, and memory estimate

- The YKI08 preflight measured a 153.29 s one-time load, 6.62 s mean query time,
  and 16.02 GiB peak allocated VRAM. Linear B0 projection is about 13.4 minutes
  including one load; reserve **15–25 minutes** because source seeking, filesystem
  cache and GPU contention vary. This is a scheduling estimate, not a result.
- Blind has no valid measured timing yet. Reserve up to 15 minutes for its one
  load plus 98 text-only calls; report the measured time separately.
- Checkpoint weights are 16,584,414,560 bytes. The established guard is weights
  plus 4 GiB (20,879,381,856 bytes), so require at least 20 GiB free on a BF16-
  capable GPU; an idle RTX 3090 Ti 24 GiB passed the preflight previously.
- Frames remain in memory and are closed after every question. No index or frame
  cache is created. Machine-readable outputs should remain below roughly 10 MiB
  per condition; the formal videos already occupy 6.82 GiB.

## Readiness decision

The **data and frozen question set are ready**. Formal inference should **not be
started yet**: first implement and test the isolated resumable 98-question
Blind/Uniform-8 execution layer described above. This is an operational safety
requirement; it does not change B0 sampling, prompt, model, or evaluation.
