# QaEgo4D E2 Closed-500 — B1/B2 Final Freeze

Status: **frozen final Closed-500 E2 result**. B3 is not started.

## Run identity and provenance

- Run ID: `qaego4d-e2-closed500-b1-b2-fp16-amendment6-v1`
- Run code commit at inference start: `bce1041afadde46d8e5fa37beb5bce2b147261f0`
- Branch: `exp/b1-flat-structure`
- Host(s): `cream.cs.ucl.ac.uk` (retrieval/preflight), `vanilla.cs.ucl.ac.uk` (resumed answer run)
- GPU for completed answer run: physical GPU 0, Quadro RTX 6000
- Random seed: `20260723` (frozen answer decoding seed)
- Paired-analysis bootstrap seed: `20260726`
- Backend/dtype: HuggingFace / FP16, unquantized
- Model: Qwen2.5-VL-7B-Instruct
- `max_pixels`: `262144`
- Decoding: greedy, `do_sample=false`, `use_cache=true`, Closed `max_new_tokens=8`
- Scope: `canonical_clip`

### Frozen input hashes

| Artifact | SHA256 |
|---|---|
| Formal E2 answer config | `dcbf2cf32e983df7a4f389e9569d5376985da2032b8e4a11feaed63cc5b4e27a` |
| Amendment #6 index config | `e2818d8797163eb773869c39f5b6a5e0b22dcd9eeef17ca9b6420b6ea5a8aa5b` |
| Closed-500 manifest | `44c461e674edd4645d57eccd673601f869a75f2c52e274fd38d666b4e696579d` |
| Closed canonical mapping | `56ec17a1062e26d5dec4019285c8d80a7d0ca3964e7a4c40cb9e3f1ac1254a9b` |
| Amendment #6 per-clip build records | `350b3188eb8968c6c17a424cbd1f1f9c2b519431969124c7beca2f7ad8129dcf` |
| Amendment #6 build summary | `2a9f4f770ca480888463ca9719859ef8499f4338aba4b385346a5bc6afb77fae` |

Active protocol is V2.2 FINAL plus Amendments #1–#6. No B0/B1/B2 definitions,
retrieval parameters, hierarchy parameters, or answer settings were changed
for this freeze. B0 is reused from Formal E1 Closed Uniform-8; this run
generated B1 then B2 only in one Qwen session.

## Frozen results

| Method | Correct / total | Accuracy | Wilson 95% CI |
|---|---:|---:|---:|
| B0 (reused E1 Uniform-8) | 262 / 500 | 52.4% | 48.02–56.74% |
| B1 Fine-flat | 288 / 500 | 57.6% | 53.23–61.86% |
| B2 Medium hierarchy | 291 / 500 | 58.2% | 53.83–62.44% |

Paired comparisons:

- B0→B1: +5.2 pp; McNemar exact `p=0.02097`; paired bootstrap 95% CI
  `[+1.0, +9.4]` pp.
- B0→B2: +5.8 pp; McNemar exact `p=0.00997`; paired bootstrap 95% CI
  `[+1.4, +10.2]` pp.
- B1→B2: +0.6 pp; McNemar exact `p=0.8126`; paired bootstrap 95% CI
  `[-2.8, +4.0]` pp. This does not establish a measurable B2 quality gain
  over B1 at this sample size.

Efficiency summary:

- Mean online latency: B0 `4.671 s`, B1 `2.571 s`, B2 `2.536 s`.
- Mean answer-model time: B0 `2.033 s`, B1 `2.028 s`, B2 `2.021 s`.
- B2 mean retrieval time: `2.71 ms` (Stage 1 `1.47 ms`, Stage 2 `1.08 ms`).
- B2 mean selected Medium count: `3.83`.
- B2 mean candidate reduction: `72.3%`.
- Peak VRAM: `17.56 GiB`; failures/OOMs: `0`.
- Active measured run interval (retrieval start through final report):
  approximately `47 minutes`; interrupted-session pauses are not folded into
  per-query latency.

## Complete artifact preservation

The complete, unpruned output directory remains at:

`outputs/experiments/qaego4d_e2_closed500_b1_b2_amendment6_v1/`

It contains all 1000 B1/B2 retrieval checkpoints, all 1000 answer checkpoints,
the reused B0 records, `REPORT.md`, `summary.json`,
`per_question_results.jsonl`, provenance, progress files, and figures.

No B3 process, checkpoint, or result was created.
