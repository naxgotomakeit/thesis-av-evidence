# Efficient Long-Video QA — Thesis Evidence & Reproduction

Training-free hierarchical video indexing and question-conditioned evidence selection for efficient long-video question answering.

The system prepares a video representation offline, then uses each question to locate relevant evidence and produce an answer. This repository preserves the thesis evidence and execution history while also providing runnable portable demonstrations.

## Two online routes

**Staged:** Question → Planner → hierarchical retrieval / shared investigation → evidence inspection → final answer.

**Direct:** Question + prepared map → map-guided model loop → optional `inspect_frames` → final answer.

The distinction is how planning and evidence gathering are organized; it is not an assumption that either route is generally more accurate.

## Key findings

The primary goal is to reduce online resource use for long-video QA. These comparisons examine the offline index, the evidence budget and the value of additional online orchestration.

- **Indexer design changes downstream efficiency.** R1 and R3 are compared under the same downstream QA design to isolate the effect of the offline index representation. On the frozen context-feasible paired-150 subset, R3 used **20.9% fewer post-Planner requests** and **37.3% fewer post-Planner total tokens** than R1, with **R3 52 versus R1 51 correct answers**. This suggests that a better-organized offline index can shift work away from online inference, with similar observed correctness on this subset. This is a read-only extraction/recomputation from frozen Eval300; it does not establish accuracy equivalence or remove R3's full-set context-capacity limitation. The largest observed R3-versus-R1 quality difference appeared under the Direct downstream, where the prepared representation is exposed more directly to the answering model: **Direct R3 103/300 versus Direct R1 visual-only 85/300**. This does not establish universal R3 superiority.
- **Evidence budget has a configuration-specific sweet spot.** Flat/H explores how to use the prepared representation and how much evidence to expose. **Flat-30 completed 254/300 with 82 correct; Dense H-15 completed 270/300 with 81 correct; Dense H-30 completed 249/300 with 78 correct.** Moderate hierarchical retrieval improved completion with similar observed correctness, but increasing the budget further did not produce monotonic gains. More evidence is not necessarily better evidence; these results do not show that hierarchy is always more accurate or faster.
- **More online orchestration did not show a clear quality benefit.** Full Staged uses Planner → retrieval → shared investigation → evidence inspection → answer; Direct uses prepared map → map-guided model loop → optional `inspect_frames` → answer. On the supplementary paired-100 comparison, **Direct scored 32/100 versus Full Staged's 24/100**; the difference was **not statistically significant** (exact McNemar **p = 0.1686**). Recorded execution cost was approximately **$4.30 for Direct**, **$11.20 for Full Staged downstream**, and **$14.49 for reconstructed Full Staged including historical Planner cost**. These figures have different execution/accounting scopes and are not one perfectly controlled contemporaneous cost experiment. The comparison did not demonstrate a clear answer-quality benefit from additional orchestration, while the staged route introduced greater orchestration complexity and higher recorded execution cost in this comparison.

For authoritative tables, figures, per-question records and provenance, see [`THESIS_ARTIFACT_INDEX.md`](THESIS_ARTIFACT_INDEX.md).

## Design implication

The results suggest emphasizing how long-video evidence is structured, filtered and exposed to the model rather than simply adding downstream reasoning stages. A future-facing interpretation—not a directly proven empirical result—is that, as multimodal models become more capable, systems may gain more from deciding what evidence to expose, when and at what online cost than from increasingly elaborate orchestration.

## Which path should I use?

- **Replay Staged:** Inspect the decomposed Planner → retrieval → evidence → answer process to understand the staged architecture and preserved Planner/retrieval behavior. The preserved example uses an API Planner with local downstream execution.
- **Replay Direct:** Inspect the simpler preserved map-guided tool-use path, including selective frame inspection.
- **Portable Live:** Prepare a new video once, then ask multiple questions. Direct is the default online route because its simpler map-guided interface and selective frame inspection are already supported by the portable workspace. The supplementary paired-100 comparison did not demonstrate a clear answer-quality benefit from more staging, while Direct had lower recorded execution cost and complexity, subject to the accounting qualifications above. This is an engineering/efficiency recommendation, not a claim that Direct is universally more accurate.

Replay shows preserved historical execution with no new inference. Live performs fresh execution; its outputs are not thesis results.

## Quick setup

Recommended: **Python 3.11**. Replay uses only Python's standard library.

System requirements for portable Live:

- ffmpeg
- ffprobe

Install the portable demo dependencies:

```bash
python3.11 -m pip install -r requirements.txt
```

For GPU deployment, install a PyTorch build compatible with the target CUDA/runtime.
`ANTHROPIC_API_KEY` is required only for live API execution.
See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for details.

## Demo

### Replay 1 — Staged pipeline

Offline replay of a preserved Planner → retrieval → answer trace.
No API key or model inference required.
The preserved run uses an API Planner with local downstream execution.

```bash
python3 demo/run_demo.py --mode replay-staged
```

### Replay 2 — Direct tool-use

Offline replay of a preserved Direct inspect-frames → answer trace.
No API key or model inference required.

```bash
python3 demo/run_demo.py --mode replay-direct
```

Both modes read preserved records without inference, media, API keys or network.

### Live — Prepare once, ask many

Prepare a new video, then ask independent A–E questions using Direct.
Both commands are plan-only until approved.

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/example --caption-backend api
```

```bash
python3 demo/ask.py --workdir demo_runs/example --question-json question.json
```

The same prepared workspace also supports open-ended Direct answering:

```bash
python3 demo/ask.py \
  --workdir demo_runs/example \
  --answer-mode open \
  --question "What is happening in the video?"
```

Open-ended answering reuses the same map-guided inspection loop; the thesis-preserved Direct protocol remains the A–E MCQ path.

The `PORTABLE_API` caption backend substitutes for the thesis Qwen model; this is
not exact thesis preprocessing. Paid execution requires environment credentials
and separate plan approval. The replay launcher's `--mode live` remains reserved.
Prepare once, then ask independent A–E questions; ASK writes to separate ignored
`qa_runs/` directories and never changes the prepared workspace or thesis evidence.
For evidence coverage, integrity checks and reproduction requirements, see the
[demo guide](demo/README.md) and [usage guide](docs/USAGE.md).

### Public Live example

Run the reproducible Wikimedia example:

```bash
python3 demo/run_live_example.py --question audio
python3 demo/run_live_example.py --question visual
```

See [`demo/example_live/README.md`](demo/example_live/README.md) for details.

## How to read this repository

For multilingual frozen reports, use the [English reading guide](docs/ENGLISH_READING_GUIDE.md). It provides English summaries while preserving the original evidence bytes.

Start with [the thesis artifact index](THESIS_ARTIFACT_INDEX.md). It maps thesis locations to the narrowest available code, configuration, result, per-question evidence, and freeze record. The accompanying guides explain [which result is authoritative](docs/RESULT_AUTHORITY.md), [the research evolution](docs/EXPERIMENT_OVERVIEW.md), [reproduction boundaries](docs/REPRODUCTION.md), and [Git/compute history](docs/GIT_HISTORY.md).

## Repository structure

- `src/`, `config/`, `configs/`, `scripts/`, `tests/` — maintained canonical AV-QA baseline and regression material.
- `experiments/hourvideo/` — immutable archival packages for HourVideo formal results, audit chains, and supplementary/appendix evidence.
- `experiments/egopolice/` — immutable Layer-1 case-study evidence and separate Layer-2 final-figure provenance.
- `provenance/` — machine-readable experiment registry.

Repeated frozen code, prompts, configurations, and manifests across archival packages are intentional provenance, not a new shared runtime.

## Thesis result map

| Thesis result family | Authoritative entry point | Status |
|---|---|---|
| Flat / Dense H comparison | [final comparison archive — English summary](docs/english/DGX_RESULTS.md) ([frozen original](experiments/hourvideo/dgx_eval300/reports/main_comparison/FINAL_H15_VS_FLAT_DATA_ARCHIVE.md)) | formal main |
| Capacity-aware R1/R3 and paired-150 | [thesis report](experiments/hourvideo/school_formal/content/capacity_aware_eval300_paired150/reports/thesis_data_report.md) | formal main / derived analysis |
| API Planner R1/R3 | [canonical report](experiments/hourvideo/api_planner_local_eval300/artifacts/formal_run/canonical_summary_v2/thesis_data_report.md) | thesis-authoritative |
| Direct R1/R3 | [visual-only thesis data — English summary](docs/english/DIRECT_R1.md) ([frozen original](experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/thesis_data_package/DIRECT_VISUAL_ONLY_EVAL300_DATA_SUMMARY.md)) | thesis-authoritative |
| GenS / ABD / evidence support | [GenS](THESIS_ARTIFACT_INDEX.md#hourvideo-results) / [ABD and audit](THESIS_ARTIFACT_INDEX.md#abd-and-evidence-support-audit) | see index |
| EgoPolice Figures 5.2–5.3 | [figure binding](experiments/egopolice/figure_layer2_provenance/FIGURE_BINDING.json) | thesis-authoritative figure provenance |

Formal QaEgo4D E1/E2 evidence is linked in [the index](THESIS_ARTIFACT_INDEX.md#qaego4d-formal-frozen-experiments).

## Important authority notes

- Direct R1 for the thesis is **85/300**, not the earlier 88/300 result.
- ABD's `E0001`–`E0900` audit namespace is distinct from the historical R1/R3/GenS `E0001`–`E0900` namespace.
- paired-150 is a read-only extraction/recomputation from capacity-aware Eval300 records, not an independent 150-question run.
- Figure 5.3(c) uses `q_medical_assistance`; `q_handcuff_before_medical` is a distinct identity.

The thesis identifies the reported version; frozen artifacts record each run. Disagreements must be recorded explicitly; see [result authority](docs/RESULT_AUTHORITY.md).

## Reproduction boundary

The repository intentionally excludes benchmark source media, complete frame pools, model weights, caches, large embeddings/index payloads, credentials, and material whose redistribution status is not confirmed. This does not remove the retained result, per-question, manifest, and checksum evidence. See [reproduction guidance](docs/REPRODUCTION.md) and [data / third-party attribution](docs/ATTRIBUTION.md).

See [public-distribution exceptions](docs/PUBLIC_DISTRIBUTION_EXCEPTIONS.md) for withheld files and the distinction between public-subset verification and a complete original-manifest pass.

## Git history

The project developed from reusable audio-visual indexing and hierarchical retrieval into HourVideo and EgoPolice evaluations. Later formal runs occurred on school, DGX, and Myriad compute environments; their preserved outputs were subsequently imported without rewriting their experimental lineage. The import date is not an experiment date.

The July–August Git history records active development. Formal experiments later ran directly in several compute environments, then their preserved artifacts were archived in commits dated 2026-09-26 and 2026-09-27. Preserved author/committer dates and branch lineage are described in [Git history](docs/GIT_HISTORY.md).
