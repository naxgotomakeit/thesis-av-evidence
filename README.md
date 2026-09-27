# Thesis AV Evidence

This repository is the examiner-facing evidence record for a master's thesis on efficient audio-visual evidence selection for egocentric long-video question answering. It contains the maintained canonical baseline and preserved, auditable artifacts for the formal experiments reported in the dissertation.

The project developed from reusable audio-visual indexing and hierarchical retrieval into HourVideo and EgoPolice evaluations. Later formal runs occurred on school, DGX, and Myriad compute environments; their preserved outputs were subsequently imported without rewriting their experimental lineage. The import date is not an experiment date.

## How to read this repository

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
| Flat / Dense H comparison | [final comparison archive](experiments/hourvideo/dgx_eval300/reports/main_comparison/FINAL_H15_VS_FLAT_DATA_ARCHIVE.md) | formal main |
| Capacity-aware R1/R3 and paired-150 | [thesis report](experiments/hourvideo/school_formal/content/capacity_aware_eval300_paired150/reports/thesis_data_report.md) | formal main / derived analysis |
| API Planner R1/R3 | [canonical report](experiments/hourvideo/api_planner_local_eval300/artifacts/formal_run/canonical_summary_v2/thesis_data_report.md) | thesis-authoritative |
| Direct R1/R3 | [visual-only thesis data](experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/thesis_data_package/DIRECT_VISUAL_ONLY_EVAL300_DATA_SUMMARY.md) | thesis-authoritative |
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

## Git history

The July–August Git history records active development. Formal experiments later ran directly in several compute environments, then their preserved artifacts were archived in commits dated 2026-09-26 and 2026-09-27. Preserved author/committer dates and branch lineage are described in [Git history](docs/GIT_HISTORY.md).
