# Thesis AV Evidence

Research code for efficient audio-visual evidence selection for egocentric
long-video question answering.

## Current research goal

Training-free, API-efficient long egocentric video QA using a reusable
hierarchical audio-visual evidence map. On `dev/thesis-av`, visual-only v1 is
implemented; audio integration is the next stage.

## Selected visual architecture

`Video -> Fine -> Safe-Merge hierarchy -> Fluid Loose Medium -> keyframe -> Medium
semantic caption -> deterministic Semantic Coarse -> storyline`

- **Fine** is immutable detailed evidence reserve.
- **Medium** is the primary semantic event/navigation unit.
- **Semantic Coarse** is adjacent-only storyline-level grouping.

Key principle: **Fine preserves information; hierarchy avoids looking at all
information every time.**

Frozen long4 validation: 295 Fine retained; 104 Medium (`8/12/29/55`); 65
Semantic Coarse (`8/12/27/18`); complete Medium lineage preserved.

Run the strict cached-input fidelity replay. It recomputes Fine, the complete
Safe-Merge tree, Fluid Loose, keyframes, Sentence-T5 boundary decisions, and
Semantic Coarse; frozen Qwen outputs are reused only after exact input identity:

```powershell
python scripts/run_visual_pipeline_v1.py --validate-fidelity
pytest -q tests/test_visual_pipeline_v1.py
```

Selected production code: `src/thesis_av/visual/`; configuration:
`config/visual_pipeline_v1.json`; design/limitations:
`docs/visual_pipeline_v1.md`.

Branch structure:

- `main` - previous stable baseline
- `exp/coarse-segmentation-3way` - research experiments and historical ablations
- `dev/thesis-av` - clean selected thesis pipeline

## EgoPolice B0: uniform full-video frames

B0 is an intentionally dull, independent baseline. For every question it
uniformly decodes eight frames from the full video, sends those frames plus the
question and five options to one local Qwen2.5-VL-3B-Instruct call, and parses
one option index. It does not use timestamps, segmentation, hierarchy,
retrieval, audio, planning, checking, or fallback. Frame decoding is repeated
per question and is included in online per-query cost.

The model path is never hard-coded. On a 6 GB GPU, the default configuration
requires bitsandbytes NF4 plus Accelerate; it fails preflight instead of trying
the approximately 7.51 GB unquantized checkpoint.

One-case smoke command:

```powershell
python scripts/baselines/smoke_egopolice_b0.py --model-path <LOCAL_QWEN_PATH> --qa-path <EGOPOLICE_MCQ_JSON> --video-root <MATCHING_EGOPOLICE_VIDEO_ROOT> --ffmpeg-path <FFMPEG> --ffprobe-path <FFPROBE> --output outputs/baselines/egopolice_b0/smoke.jsonl
```

After a successful smoke, run at most five cases by replacing the smoke script
with `scripts/baselines/run_egopolice_b0.py --limit 5`.

## Canonical Baseline v1

The current executable baseline separates reusable offline indexing from the
per-question online path:

```text
offline reusable multimodal indexing
  -> question planning
  -> fresh question-conditioned query encoding and modality-aware retrieval
  -> local refinement
  -> evidence sufficiency / conditional fallback
  -> relation-aware reranking
  -> compact evidence packet
  -> final grounded QA with uncertainty
```

Canonical research versions:

- Task 5A v2
- Task 5B v1.1
- Task 5C v1.2
- Task 6 v1.2
- Task 7A v1
- Task 7B v3 (historical on-disk outputs may use `task7b_v0_3`)

The source of truth is
[config/canonical_pipeline.json](config/canonical_pipeline.json).
Historical Task scripts remain intact as the research-development lineage and
are imported where their validated helpers are canonical dependencies. The
`src/canonical_pipeline/` package is the separate executable baseline; it does
not execute historical correction versions as sequential runtime stages.
The generalized runner accepts an explicit case manifest and scores reusable
offline visual, transcript, and acoustic indexes directly. Historical Task 4
per-question score files are retained only for regression comparison, never as
a prerequisite for a new question. Selected acoustic intervals are materialized
to local WAV evidence on demand using the final Task 5C v1.2 behavior.
Technical producer/consumer validation additionally enforces complete evidence
group membership, relation endpoint closure and uniqueness, globally unique
model-facing evidence IDs, per-video index compatibility, and an exact Task7A
to Task7B v3 schema contract. These checks stabilize serialization and
reproducibility; they do not change evidence ranking or selection policy.

## Setup

Create a Python environment with the project dependencies, copy `.env.example`
to a local `.env`, and provide credentials only in that ignored file or through
process environment variables. Copy the required `configs/*.example` templates
to their non-example names and configure local dataset/model-cache paths.

The EgoSound dataset, QA manifests/annotations, raw media, generated indexes,
embeddings, model weights and experiment outputs are intentionally not included
in Git. See [data/README.md](data/README.md) for the expected local layout.

Zero-call regression:

```bash
python scripts/canonical/run_regression.py
python scripts/canonical/run_case.py --case-id 00006_3
```

Live execution remains explicitly gated behind `--execute-live`. Provider keys
are loaded from ignored local environment configuration. Gold/reference answers
are post-hoc only and never enter the canonical online state.

## Status

Current status: **Canonical Baseline v1**. Deterministic regression and the
first two-case integrated live verification passed without a category-4
research-behavior mismatch. This is not yet a full benchmark result. The
repository makes no SOTA claim and does not yet claim a proven accuracy or
efficiency improvement.

Generated outputs, local media, embeddings, model weights, caches and secrets
are excluded from Git. See [CURRENT_BASELINE.md](CURRENT_BASELINE.md) for source
lineage and integration status.

## Linux / School GPU Setup

The portable runtime accepts `DATA_ROOT`, `MODEL_PATH` (or `MODEL_ROOT`),
`OUTPUT_ROOT`, `FFMPEG_PATH`, and `FFPROBE_PATH`; B0 has no dependency on a
Windows drive letter. Use Python 3.10, inspect the server first with
`nvidia-smi`, install a compatible PyTorch/torchvision wheel before
`requirements.txt`, and keep data/models outside the repository.

The exact Linux setup, model download, environment check, Windows smoke command,
and staged B0 commands are in
[docs/linux_school_gpu_setup.md](docs/linux_school_gpu_setup.md). The frozen,
performance-independent EgoPolice pool is
[config/data/egopolice_50videos.json](config/data/egopolice_50videos.json).
