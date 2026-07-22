# C-RADIOv4 representation smoke report

Date: 2026-07-22  
Status: completed exploratory diagnostic  
Scope: **not a formal B0–B4 result**

This report reconstructs the completed experiment from the saved result JSON,
embedding artifact, experiment log, and repository state. The experiment was
not rerun to prepare this document.

## Purpose and experimental boundaries

The diagnostic tested whether official NVIDIA C-RADIOv4 representations are a
practical reusable semantic frame representation for later long-video
question-conditioned retrieval research. It measured full-video 1 FPS
embedding cost, storage cost, and post-hoc temporal exposure against frozen
EgoPolice GT intervals.

The run did not call Qwen, execute B0, define or execute B1, or modify the
formal 20-video manifest, formal 98-question manifest, frozen 50-video parent
manifest, B0 configuration, frozen visual hierarchy, DINOv2 segmentation, or
formal evaluation protocol.

## Exact model and checkpoint

| Field | Value |
|---|---|
| Official model ID | `nvidia/C-RADIOv4-SO400M` |
| Checkpoint file | `c-radio_v4-so400m_half.pth.tar` |
| Checkpoint revision | `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8` |
| Checkpoint SHA256 | `d02697ede20f2716c4db12a9d3dab1c5a9b47d15a16cdcd46a69d08062b77aaf` |
| Official implementation | `NVlabs/RADIO` |
| Implementation revision | `c0f37017930e9dda53f93424cf4bf39fc51f287e` |
| Text-aligned adaptor | Official `siglip2-g` |
| Adaptor text model | `google/siglip2-giant-opt-patch16-384` |
| Resolved text-model revision | `a713301b217d38485fb2204c808367d10bc3cc40` |
| Compute dtype | BF16 CUDA autocast |
| Checkpoint-configured dtype | `torch.float32` |
| Stored embedding dtype | float16 |
| Official model-card parameter count | 431,237,240 |
| Runtime backbone parameter count | 431,237,232 |
| Runtime text-model parameter count | 708,225,648 |
| Runtime total loaded parameter count | 1,184,154,656 |

The official `siglip2-g` tokenizer and text encoder loaded and encoded all five
queries successfully. No custom text projection was introduced.

## Runtime and implementation setup

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti |
| GPU memory before load | 25,282,478,080 bytes total; 24,885,526,528 bytes free |
| Python | 3.10.20 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| Transformers | 5.14.1 |
| decord | 0.6.0 |
| timm | 1.0.28 |
| einops | 0.8.2 |
| Batch size | 4 |
| Input resolution | 512 × 512 |
| Cache root | `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/diagnostic_caches/cradio_v4` |

The official checkpoint and implementation revisions are pinned by the
isolated diagnostic runner. Frames were decoded into memory in batches using
decord, converted from RGB uint8 to `[0,1]` float tensors, resized to 512×512,
and passed through C-RADIOv4 under BF16 CUDA autocast. The normalized
`siglip2-g` visual summary was saved as the per-frame representation. No frame
images were permanently written.

The full source video was sampled on the deterministic grid
`0, 1, ..., ceil(duration)-1`, retaining only timestamps strictly below the
video duration. Each timestamp maps deterministically to
`floor(timestamp × average_fps)`, clipped to the valid source-frame range.

## Video and sampling

| Field | Value |
|---|---|
| Video ID | `pasadena/YKI08` |
| Source path | `/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/data/EgoPolice_1.0.0/videos/pasadena/YKI08.mp4` |
| Duration | 1611.584 seconds |
| File size | 670,287,000 bytes |
| Source resolution | 1280 × 720 |
| Source FPS | 30.0 |
| Source frame count | 48,344 |
| Video codec | H.264 |
| Sampling FPS | 1.0 |
| Embedded timestamps | 1,612 (`0` through `1611` seconds) |
| Embedding dimension | 1,536 |

Artifact validation found an embedding array of shape `(1612, 1536)`, float16
dtype, finite values, and mean L2 norm approximately 1.0. Timestamps increase
by exactly one second and source frame indices are monotonic.

## Offline embedding efficiency

| Metric | Result |
|---|---:|
| Model/adaptor cold loading time | 201.136720 seconds |
| Checkpoint download/cache resolution | 24.550789 seconds |
| Embedding generation wall time | 50.588957 seconds |
| Decode time | 13.292165 seconds |
| Tensor preprocessing time | 3.073690 seconds |
| C-RADIO visual inference time | 34.186815 seconds |
| Embedding serialization time | 0.155640 seconds |
| Embedding artifact-ready time, excluding model load | 50.744597 seconds |
| Throughput | 31.864662 embedded frames/second |
| Text encoding time for all five queries | 0.023955 seconds |
| Total diagnostic time after checkpoint became available | 259.805891 seconds |
| Derived full successful-process time including checkpoint resolution | 284.356679 seconds |
| C-RADIO visual forward calls | 403 |
| SigLIP2 text forward calls | 1 |
| Qwen model calls | 0 |

The cold model-loading measurement includes the first download and
construction of the official SigLIP2-Giant model used by `siglip2-g`. The
C-RADIO checkpoint download/cache-resolution time was measured separately.

## GPU memory

- Peak allocated GPU memory: **4,992,640,000 bytes**, approximately 4.65 GiB.
- The same peak was observed during the embedding phase.
- After the process terminated, `nvidia-smi` returned to 111 MiB used and
  24,001 MiB free, confirming that the diagnostic process released the GPU.

## Embedding storage

| Metric | Result |
|---|---:|
| Format | compressed NumPy `.npz` |
| Stored embedding dtype | float16 |
| Artifact size | 4,540,678 bytes, approximately 4.33 MiB |
| Bytes per embedded frame | 2,816.7978 |
| Artifact SHA256 | `f10946bc238bc55b04cb6dde78131b6b221093dbaf7cd0b8b758692de5b6564e` |

The artifact stores compact embeddings, float64 timestamps, and int64 source
frame indices. It does not contain decoded JPEG or PNG files.

## Retrieval setup and metric semantics

All five frozen formal YKI08 questions in
`config/data/egopolice_ablation_questions_v1.json` were used. Retrieval used
only question text; answer options were not included. Cosine similarity was
computed between the normalized official `siglip2-g` text representation and
all 1 FPS frame representations. Ranking completed before the annotated GT
interval was consulted.

Every selected entry has the same question text:

> Which action is happening in this video clip?

Consequently, there was one unique query string and all five questions
necessarily received the same timestamp ranking. This is an important
limitation of this particular question-only diagnostic.

GT Interval Hit@K means that at least one Top-K timestamp lies inside the
annotated half-open interval `[start,end)`. It is a coarse post-hoc temporal
exposure diagnostic. It does not guarantee that the sampled frame shows the
decisive action and is not a true evidence-recall metric.

## Aggregate GT Interval results

| Metric | Hits | Total | Rate |
|---|---:|---:|---:|
| GT Interval Hit@1 | 0 | 5 | 0% |
| GT Interval Hit@5 | 0 | 5 | 0% |
| GT Interval Hit@8 | 1 | 5 | 20% |
| GT Interval Hit@10 | 1 | 5 | 20% |

## Per-question retrieval results

| Question ID | Class | GT interval | Hit@1 | Hit@5 | Hit@8 | Hit@10 | Top-1 distance to interval | First in-interval rank |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `1s_3455` | 1s | [1104,1105) | 0 | 0 | 0 | 0 | 410 s | 89 |
| `1s_3447` | 1s | [1093,1094) | 0 | 0 | 0 | 0 | 421 s | 1306 |
| `10s_4776` | 10s | [90,100) | 0 | 0 | 0 | 0 | 1415 s | 829 |
| `10s_3437` | 10s | [1080,1090) | 0 | 0 | 0 | 0 | 425 s | 28 |
| `60s_1377` | 60s | [1080,1140) | 0 | 0 | 1 | 1 | 375 s | 7 |

The only Top-8/Top-10 temporal hit was `60s_1377`, because the seventh-ranked
timestamp, 1109 seconds, lies inside `[1080,1140)`. The narrower 10-second
interval `[1080,1090)` did not hit in the Top-10; its first in-interval
timestamp was ranked 28th.

## Qualitative Top-10 retrieval ranking

Because all question strings are identical, the following ranking applies to
each of the five questions.

| Rank | Timestamp | Source frame | Cosine similarity |
|---:|---:|---:|---:|
| 1 | 1515 s | 45,450 | 0.111578666 |
| 2 | 866 s | 25,980 | 0.110420793 |
| 3 | 1070 s | 32,100 | 0.109249897 |
| 4 | 1514 s | 45,420 | 0.107391365 |
| 5 | 1068 s | 32,040 | 0.106573746 |
| 6 | 1510 s | 45,300 | 0.106538430 |
| 7 | 1109 s | 33,270 | 0.106366441 |
| 8 | 1222 s | 36,660 | 0.105824232 |
| 9 | 1345 s | 40,350 | 0.105402887 |
| 10 | 906 s | 27,180 | 0.105393991 |

## Errors, warnings, and fixes

### Initial disk-quota failure

The first official `siglip2-g` load stopped before any frame embedding because
Hugging Face defaulted to the quota-limited user cache under
`/cs/student/msc/rai/2025/xinanx01/.cache/`. The error was
`OSError: [Errno 122] Disk quota exceeded`.

The permanent diagnostic fix sets both `HF_HOME` and `TORCH_HOME` beneath:

`/cs/student/project_msc/2025/rai/xinanx01/msc_thesis/models/diagnostic_caches/cradio_v4`

The successful run used this canonical project storage. No video embeddings
had been produced by the failed attempt, and the successful run completed
without an error.

### Non-fatal warnings

- Hugging Face downloads were unauthenticated, so lower rate limits applied.
- Transformers 5.14.1 warned that SigLIP `bos_token_id=49406` and
  `eos_token_id=49407` lie outside the reported vocabulary range `0..31999`.
- The official loader reported unexpected checkpoint keys
  `patch_generator._vis_cond.norm_mean` and
  `patch_generator._vis_cond.norm_std`.

Despite these warnings, official text encoding and all visual embedding calls
completed successfully.

## DINOv2 control

No DINOv2 comparison was run. The existing frozen DINOv2 component is
vision-only and has no existing question-compatible text-aligned similarity
method. Creating a custom mapping solely for this comparison would not have
been a fair controlled diagnostic, so none was invented.

## Conclusions

C-RADIOv4-SO400M is operationally practical as a reusable frame
representation on the available RTX 3090 Ti. The measured throughput was
approximately 31.9 FPS, peak allocated VRAM stayed below 5 GiB, and a
26.9-minute video required only 4.33 MiB of compressed embedding storage.
These results support retaining C-RADIOv4 as a technically viable candidate
representation for further study.

This smoke does **not** establish C-RADIOv4 as the fixed B1 retrieval scorer.
The frozen EgoPolice question text is generic and, with answer options excluded
as required, supplies no action-specific retrieval signal. All five questions
therefore collapse to one ranking. The observed 0/5 Hit@5 and 1/5 Hit@8 cannot
cleanly distinguish representation quality from query-information failure.
No B1 definition or formal experiment was changed based on this result.

## Code and output artifacts

- Raw machine-readable result:
  `outputs/diagnostics/cradio_v4/YKI08_cradio_v4_so400m_1fps_results.json`
- Reusable embedding artifact:
  `outputs/diagnostics/cradio_v4/YKI08_1fps_siglip2g_embeddings.npz`
- Diagnostic runner: `src/diagnostics/cradio_v4/runner.py`
- Metric and manifest logic: `src/diagnostics/cradio_v4/core.py`
- CLI: `scripts/diagnostics/cradio_v4/run_yki08_representation_smoke.py`
- Isolated dependency pins: `scripts/diagnostics/cradio_v4/requirements.txt`
- Usage documentation: `scripts/diagnostics/cradio_v4/README.md`
- Unit tests: `tests/diagnostics/cradio_v4/test_core.py`
- Experiment log: `docs/EXPERIMENT_LOG.md`

The generated output directory contains only the result JSON and embedding
NPZ; it contains no permanently decoded frame images.

## Validation and tests

- Diagnostic and frozen-contract unit tests: **40 passed in 3.19 seconds**.
- Python compilation checks passed for the runner, core module, and CLI.
- `git diff --check` passed.
- Embedding shape, dtype, finiteness, normalization, timestamp grid, monotonic
  frame mapping, identical-query ranking, and zero Qwen-call accounting were
  validated from the saved artifacts.
- A broader regression command was stopped when
  `tests/test_visual_pipeline_v1.py` began invoking the frozen visual-pipeline
  fidelity script; it was not counted as a passing test and no tracked frozen
  pipeline file was changed.

## Git status at report creation

Branch and tracking state:

```text
exp/dull-baseline-qwen7b...origin/exp/dull-baseline-qwen7b [ahead 3]
```

Working tree after adding this report:

```text
 M docs/EXPERIMENT_LOG.md
?? docs/reports/CRADIO_V4_SMOKE_REPORT.md
?? scripts/diagnostics/cradio_v4/
?? src/diagnostics/cradio_v4/
?? tests/diagnostics/cradio_v4/
```

The diagnostic outputs under `outputs/diagnostics/cradio_v4/` are retained
locally but ignored by Git. No commit or push was performed for this work.

