# Result authority and identity rules

A **THESIS-AUTHORITATIVE** result is the result to cite for the submitted thesis. **HISTORICAL**, **SUPERSEDED-FOR-THESIS**, and **APPENDIX-ONLY** material remains available for audit, but must not silently replace a thesis result.

The submitted thesis determines which result/version was reported; frozen artifacts determine what each preserved run contains. Record any disagreement explicitly. Never alter, reinterpret, or suppress frozen facts to match the thesis. These authority rules identify the adopted version and do not change another version's experimental identity. Both Direct 85 and earlier formal 88 are real preserved results.

## Direct R1

The [recovered corrected-R1 evidence-support audit](../experiments/hourvideo/direct_r1_visual_only_thesis/corrected_evidence_audit/README.md) preserves the 300-question post-review/deblinded label and outcome closure (175 new reviews, 125 validated reuses). It documents evidence support for the corrected visual-only lineage; this pointer does not change result authority. Do not join its neutral IDs to other audits without origin/source/question identity.

Earlier formal Direct R1 is **88/300**, with 300 predictions and ASR-exposed maps on 175 routes. It is [HISTORICAL / SUPERSEDED-FOR-THESIS](../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/SUPERSEDED_FOR_THESIS/old_formal_88/).

The thesis result is **85/300**, 299 completed: 175 visual-only routes were rerun and 125 routes were SHA-verified for reuse. Cite [the thesis data summary — English summary](english/DIRECT_R1.md) ([frozen original](../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/thesis_data_package/DIRECT_VISUAL_ONLY_EVAL300_DATA_SUMMARY.md)), [per-question output](../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/final_outputs/scored_results_with_gold.json), and [route reuse manifest](../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/manifests/route_reuse_manifest_v8.json).

`88 → 85` is real lineage, not an offline derivation. It cannot be causally attributed wholly to ASR removal: the correction ran later, and cross-time provider behaviour plus recovery/transport handling are documented non-determinism.

## HV2 API Planner accounting

R1/R3 full-chain totals are **42,709,209 / 36,680,380 tokens** (Planner plus post-Planner workflow). Planner-only input tokens are **2,870,342 / 8,510,653** and Planner API costs are **$4.965457 / $10.096273**. Do not interchange these scopes. See [canonical report](../experiments/hourvideo/api_planner_local_eval300/artifacts/formal_run/canonical_summary_v2/thesis_data_report.md) and [planner cost](../experiments/hourvideo/api_planner_local_eval300/artifacts/formal_run/canonical_summary_v2/planner_cost.json).

## Audit namespaces

`E0001`–`E0900` is not globally unique. ABD A/B/D uses `audit_id` with ABD `variant`, `question_id`, and `task_id` in [abd_lineage](../experiments/hourvideo/abd_evidence_audit/content/abd_lineage/). Historical R1/R3/GenS uses `neutral_id` with `source_group`, `source_identity`, and `question_id` in [its separate audit](../experiments/hourvideo/abd_evidence_audit/content/historical_r1_r3_gens_audit_900/). Never join populations by `E####` alone.

## Derived and historical comparisons

- paired-150 is a pre-defined, read-only extraction/recomputation of frozen capacity-aware Eval300 records, not a rerun.
- [Old-H](../experiments/hourvideo/dgx_eval300/appendix_candidates/old_h/) is **APPENDIX / HISTORICAL ONLY** and cannot substitute for New Dense H.
- `OLD_VALIDATION_REPORT_INVALID_AGGREGATION_DO_NOT_USE.json` is expressly non-authoritative.

## EgoPolice figures

Figure 5.2 binds V v2.2 (6 regions) and AV-Speech v2.2 (10 regions). Figure 5.3 uses `q_weapon_visible`, `q_visible_injury`, and **`q_medical_assistance`**. `q_handcuff_before_medical` is distinct. The AV-Speech medical-assistance panel is a format-only recovery with zero new frames and zero new inspection calls; it does not change primary completion. See [the authoritative binding](../experiments/egopolice/figure_layer2_provenance/FIGURE_BINDING.json).
