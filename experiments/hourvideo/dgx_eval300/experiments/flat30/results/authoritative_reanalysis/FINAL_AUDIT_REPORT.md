# HourVideo Eval300 authoritative reanalysis

Generated: 2026-08-26T11:55:23.604393+00:00  
Frozen UID manifest: `/home/naxucl/data/HourVideo/benchmark/v1.0_release/hourvideo_eval300_v1_uids.txt`  
SHA-256: `6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1`  
Coverage: 300 unique UIDs; Flat, H-6, H-15 and H-30 all cover the identical frozen set.

## Verified authoritative counts

| config | completed/300 | correct/300 | correct/completed | timeout | illegal pred | success strict-invalid | E2E mean / median / P90 / P95 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| flat | 254 | 82 (27.33%) | 32.28% | 36 | 41 | 10 | 318.990 / 252.526 / 631.404 / 809.998 |
| h6 | 280 | 64 (21.33%) | 22.86% | 18 | 20 | 2 | 279.504 / 221.083 / 522.963 / 758.698 |
| h15 | 287 | 64 (21.33%) | 22.30% | 11 | 13 | 2 | 271.692 / 216.761 / 487.916 / 699.659 |
| h30 | 257 | 66 (22.00%) | 25.68% | 42 | 43 | 1 | 294.693 / 235.625 / 549.194 / 795.833 |

Timeout, illegal prediction, and strict invalid can overlap and must not be added to derive incompletion.

## Corrected paired results

All accuracy comparisons use all 300 UIDs, treating incomplete as not correct. Completion comparisons pair strict-completed state. Latency uses only jointly strict-completed UIDs. Exact values, Holm adjustment, paired bootstrap, Wilcoxon, and video-cluster bootstrap are in `paired_statistics.json`.

- h6: accuracy discordance Flat-only=51, Hier-only=33, exact p=0.0629723, Holm p=0.169993, difference=-6.00 pp, CI=[-0.12, 0.0]; completion exact p=0.000535436, Holm p=0.00107087; latency n=240, mean Hier-Flat=-48.695s, paired bootstrap CI=[-82.84376171854167, -14.82786471208332].
- h15: accuracy discordance Flat-only=49, Hier-only=31, exact p=0.0566644, Holm p=0.169993, difference=-6.00 pp, CI=[-0.11666666666666667, 0.0]; completion exact p=5.55052e-06, Holm p=1.66516e-05; latency n=244, mean Hier-Flat=-50.288s, paired bootstrap CI=[-83.30322788627049, -17.575659400307305].
- h30: accuracy discordance Flat-only=49, Hier-only=33, exact p=0.0970309, Holm p=0.169993, difference=-5.33 pp, CI=[-0.11333333333333333, 0.0033333333333333335]; completion exact p=0.809949, Holm p=0.809949; latency n=221, mean Hier-Flat=-32.866s, paired bootstrap CI=[-67.26702697658371, 1.3978660372171963].


## Stale and corrected evidence

- The old hierarchical `paired_with_flat` reference used first-pass Flat (219 strict-valid), not authoritative merged Flat (254). Those results are stale; no source file was overwritten.
- The old `parser_failures` counter is not an event counter. It can apply a trajectory-level invalid-response note while iterating multiple steps, and timeout attempts may disappear when only completed trajectories are selected. This report reconstructs events step by step: no parsed action and no nonempty `<final>`.

## H-15 attribution limits

Directly observable: terminal status, predictions, action sequence, returned candidate metadata, summaries, Inspector inputs/outputs, reconstructed protocol events, frames recorded as sent, fallback flags present in artifacts, and elapsed time. Candidate sets often differ and timeout/protocol events are observable.

Not directly provable without ground-truth temporal visual evidence: that a candidate difference is a recall failure, that a summary omission alone caused a wrong answer, or that a parser event independently caused timeout/error. Automated labels are deliberately conservative; `retrieval_candidate_difference` means only an observed candidate difference.

The dataset contains 300 questions from 12 videos, not 300 independent video samples. Both UID bootstrap and 12-video cluster bootstrap are reported.

## Decision gate

**{decision['status']}**. This audit does not provide enough artifact-only evidence to approve full extension, nor enough evidence to declare a systematic retrieval redesign requirement. Review the deterministic packet (seed {SEED}) and resolve the stated observability/stale-comparison issues before authorizing remaining Flat/H-15 execution.

## Execution boundary

This was an offline filesystem analysis. It did not start a model, GPU workload, API call, retry, rescoring run, or remaining-question run. Original results, runtime, indexer, planner, inspector, configurations, trajectories, predictions, and formal manifests were not modified.
