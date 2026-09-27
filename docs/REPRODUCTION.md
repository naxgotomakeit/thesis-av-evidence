# Reproduction and audit boundary

This repository does not provide one-command reproduction of all experiments. Reproduction capability and experimental classification are separate dimensions.

| Status | Meaning |
|---|---|
| AUDIT-REPRODUCIBLE | Recompute statistics from retained frozen predictions/results/labels, inspect identities and manifests, and verify available bytes against SHA records. Missing external payloads cannot themselves be hash-verified here. |
| RE-RUNNABLE-WITH-EXTERNAL-ASSETS | Implementation/configuration or launch/plotting code exists; execution requires benchmark media/frames, models/API access, weights or omitted indexes/embeddings, plus environment/path setup. Numerical identity across new API runs is not promised. |
| RESULT-ONLY | Result evidence is retained but executable implementation/configuration is genuinely absent. Do not assign this merely because media or model weights are external. |
| HISTORICAL-ONLY | Retained for historical comparison or supersession; this describes citation role, not necessarily absence of audit capability. |
| APPENDIX-ONLY | Appendix comparison, not a main-result substitute. |
| DO-NOT-CITE | Invalid/non-authoritative material must not support formal result claims; its history may still be documented. |

**AUDIT-REPRODUCIBLE ≠ semantic re-review without source media.** Recomputing ABD support statistics from frozen labels does not independently validate the semantic judgment behind those labels.

| Experiment | Capability and boundary |
|---|---|
| QaEgo4D frozen E2 / reused E1 B0 | Audit retained 1,500 records at the pinned Git ref; rerun E1/E2 implementations with external assets. Standalone E1 raw outputs are not included at that ref. |
| Capacity-aware / paired-150 | Audit frozen routes and subset aggregation; rerun the parent workflow with external assets. paired-150 is extraction, not new inference. |
| Flat-30 / Flat-15 / Dense H-8/H-15/H-30 | Audit condition-specific final records; rerun frozen runtime/launch code with external assets. |
| API Planner / Direct / GenS Haiku / ABD formal | Audit retained results and recorded costs; rerun requires external media and model/API services. |
| GenS V2 selector | Audit selection tables/provenance; wrapper/config/launcher exist, so RE-RUNNABLE-WITH-EXTERNAL-ASSETS applies. Raw GenS responses/token IDs and full third-party source are omitted. |
| ABD final / historical evidence audits | Audit frozen labels, identity mapping and statistics; semantic re-review needs the original model-visible media. Historical R1 labels do not become labels for corrected Direct R1 merely through shared question IDs. |
| EgoPolice figure pipeline | Audit binding, stored PDFs, routes and checksums; plotting script exists, so rendering is RE-RUNNABLE-WITH-EXTERNAL-ASSETS, including the omitted frame inputs. |
| Old-H | APPENDIX-ONLY / historical comparison; condition manifests retain audit provenance. |

Paths in the registry are repository-root-relative at the specified `git_ref`. `CURRENT` means this checkout; other values are pinned Git commits. `N/A` denotes no applicable artifact. Configs are not represented as result or freeze records. Multiple reproduction capabilities may be separated with `;`.

Benchmark source media, full frame pools, model weights, caches, large embeddings/indexes and credentials are intentionally omitted. Consult [data and third-party attribution](ATTRIBUTION.md) and package exclusion records. This is not a claim that every benchmark-derived annotation or textual result is absent. No public navigation should depend on private local-path mappings.
