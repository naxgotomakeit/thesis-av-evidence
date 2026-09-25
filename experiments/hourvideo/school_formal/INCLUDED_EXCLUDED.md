# Included and excluded material

## Included

| Area | Included material | Per-question/audit closure retained |
|---|---|---|
| Preprocessing and indexes | R1 and latest R3 entry code, embedded/full prompts, frozen configs, exact-12-video telemetry and validation, offline-cost report, R3 Variant-C/V7.4 source and package manifests | Index build summaries and hashes; no index payloads |
| Direct R1/R3 Eval300 | Frozen provider config, full Direct-v1.2 prompt implementation, controller/provider/runtime code, candidate builder, launch scripts, authoritative v3 manifest/lock, 600 route statuses, request/attempt telemetry, canonical per-route table and reports | 600 fixed-denominator rows |
| GenS Haiku v3 | Full prompt/config/code, Myriad GenS V2 selector manifest package, v3 preflight freeze, 300 route results/statuses, request/attempt telemetry, scoring and validation outputs | 300 selector rows and 300 scored routes |
| Capacity-aware Eval300 and paired-150 | Frozen prompts/config, runtime and aggregation code, service/input freeze, 150-UID eligibility set, 300 per-question `model_attempts.jsonl` files containing both executed routes where applicable, route statuses, 600 canonical rows and reports | Sufficient to re-extract Post-Planner paired-150 calls/tokens/images from formal telemetry |
| Full Staged paired-100 | Frozen config, complete stage prompts in `contracts.py`, staged runtime plus exact fingerprinted legacy dependency closure, preflight manifest, route statuses, request/attempt telemetry, score tables and thesis package | 100 fixed-denominator paired rows |

## Deliberately excluded

| Excluded class | Reason |
|---|---|
| Source videos and extracted frames | Large benchmark media; not needed for the first code/result import and not assumed redistributable |
| Model weights, model cache, virtual environments and bytecode cache | Large/reproducible infrastructure artifacts; may have separate licensing |
| Full R1/R3 indexes, embeddings, detector observations and map payloads | Large derived intermediates; manifests, code and summary telemetry are retained instead |
| Raw benchmark annotations, question-input files and copied third-party datasets | Redistribution status is not established; per-question IDs, predictions, correctness and resource telemetry are retained without copying the source dataset |
| Direct route artifacts and controller/lifecycle journals | They duplicate canonical results and contain expanded question/map/model text; route status plus request/attempt telemetry and canonical tables are retained |
| Full Staged `provider_responses.jsonl`, controller results and `legacy_live` tree | Duplicate raw response bodies and expanded legacy intermediates; audit-grade statuses, attempts and thesis tables are retained |
| Capacity-aware full frozen Planner outputs | Large duplicated prompts/responses; `planner_attempts.jsonl`, canonical Planner cost, prompt freeze and manifests are retained |
| Tar archives and duplicate transfer packages | Their content manifests and final-version package reports are retained; bulky duplicate archives are not |
| Credentials and `.env` files | Never copied |
| Historical pilots, smoke/fake runs, failed service-start logs and superseded result versions | Listed as historical below rather than mixed with formal results |

## Historical or failed versions not imported as results

- GenS v1/v2, v3 smoke and parser-development outputs; only the formal
  `v3_direct_parser_aligned` run is included.
- Direct-v1.1 and early Direct-v1.2 pilots/fake-provider runs; only authoritative
  formal manifest/launch-lock v3 and its final canonical summary are included.
- Full Staged Eval300/aligned-v2 pilots and the obsolete protocol statement
  corrected by the paired-100 thesis package; only `paired100_final_v2` is
  included as a supplementary result.
- The capacity-aware legacy top-level validation aggregation. The retained file
  `OLD_VALIDATION_REPORT_INVALID_AGGREGATION_DO_NOT_USE.json` is present only as
  an explicit warning; versioned `canonical_summary_v1` is authoritative.
- Older R3/R3.2 indexing variants; the package keeps the latest
  action-preserving caption index and the Variant-C/V7.4 frozen materialization
  metadata used downstream.

