# Experiment overview

This describes research evolution, not a claim that every retained experiment is a thesis main result. Use [the thesis index](../THESIS_ARTIFACT_INDEX.md) for exact evidence links.

```text
canonical AV-QA baseline → visual hierarchy / semantic coarse → EgoPolice exploration and formal QaEgo4D E1/E2
→ HourVideo R1/R3 and runtime → capacity-aware hierarchical retrieval
→ Flat / New Dense H → API Planner, Direct, GenS, ABD → evidence-audit correction and thesis freeze
```

| Material | Classification | Entry point |
|---|---|---|
| Canonical baseline | dependency / method code | [`src/`](../src/) and [`config/canonical_pipeline.json`](../config/canonical_pipeline.json) |
| Visual hierarchy / early EgoPolice | historical development | [Git history](GIT_HISTORY.md) |
| QaEgo4D E1/E2 Closed-500 | FORMAL-FROZEN; stored on historical branch; thesis location unverified | [formal evidence](../THESIS_ARTIFACT_INDEX.md#qaego4d-formal-frozen-experiments) |
| Capacity-aware R1/R3 Eval300 | FORMAL-MAIN | [report](../experiments/hourvideo/school_formal/content/capacity_aware_eval300_paired150/reports/thesis_data_report.md) |
| paired-150 | formal derived analysis | same frozen Eval300 report |
| Flat-30 / Flat-15 / Dense H-8/H-15/H-30 | FORMAL-MAIN | [DGX verification](../experiments/hourvideo/dgx_eval300/metadata/VERIFICATION_REPORT.md) |
| Old-H | APPENDIX-ONLY / HISTORICAL-ONLY | [archive](../experiments/hourvideo/dgx_eval300/appendix_candidates/old_h/) |
| Direct old 88 | SUPERSEDED-FOR-THESIS | [old formal](../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/SUPERSEDED_FOR_THESIS/old_formal_88/) |
| Direct visual-only 85 and R3 | THESIS-AUTHORITATIVE | [thesis package](../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/thesis_data_package/) |
| GenS V2 selector | dependency snapshot / selector evidence | [Myriad package](../experiments/hourvideo/myriad_gens_v2/README.md) |
| GenS Haiku v3 | FORMAL-MAIN | `experiments/hourvideo/school_formal/content/gens_haiku_v3/` |
| Full Staged R3 vs Direct R3 paired-100 | SUPPLEMENTARY | [results — English summary](english/PAIRED100.md) ([frozen original](../experiments/hourvideo/school_formal/content/supplementary/full_staged_paired100/reports/thesis_package/PAIRED100_THESIS_RESULTS.md)) |
| ABD A/B/D and final audit | FORMAL-MAIN / evidence support | [final metrics](../experiments/hourvideo/abd_evidence_audit/content/abd_lineage/05_final_authoritative_900/ABD_FINAL_METRICS.md) |
| Historical R1/R3/GenS audit | HISTORICAL-ONLY comparison | [audit report — English summary](english/HISTORICAL_AUDIT.md) ([frozen original](../experiments/hourvideo/abd_evidence_audit/content/historical_r1_r3_gens_audit_900/audit/EVIDENCE_AUDIT_REPORT.md)) |
| EgoPolice figures | thesis figure provenance | [Layer 2 binding](../experiments/egopolice/figure_layer2_provenance/FIGURE_BINDING.json) |

`DO-NOT-CITE` marks material that must not support a formal result claim, including invalid/superseded analyses explicitly marked “do not use.” Smoke and preliminary records may still document development; cite their historical role explicitly. Repeated frozen runtime/config copies remain in their original packages as provenance.
