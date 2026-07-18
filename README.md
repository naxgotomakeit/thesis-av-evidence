# Thesis AV Evidence

Research code for efficient audio-visual evidence selection for egocentric
long-video question answering.

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
