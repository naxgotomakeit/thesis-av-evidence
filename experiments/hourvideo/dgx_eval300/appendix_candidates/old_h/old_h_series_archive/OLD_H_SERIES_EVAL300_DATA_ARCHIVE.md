# HourVideo Eval300 Old-H Series Data Archive

## 1. Archive identity and status

This document archives the historical H-6/H-15/H-30 experiments that preceded the New Dense Semantic Beam-B series.

**Classification: `HISTORICAL DIAGNOSTIC / NOT NEW DENSE-H`**

These results are valid records of the systems that were actually run, but they must not be presented as results of the later Dense Semantic Beam-B Retriever and must not be mixed with New H-8/H-15/H-30 under a single method name.

The corrected old-H view in `full_video_fallback_fix_20260826T145534Z/corrected_eval300` is the final citable version of this historical series. The original pre-correction values remain preserved as execution history.

All configurations use the same frozen Eval300 set of 300 unique UIDs from 12 videos. Accuracy uses a fixed denominator of 300 unless explicitly marked completed-only.

## 2. What the old series actually tested

| Profile | Historical retrieval behavior | What it does not test |
|---|---|---|
| Old H-6 | Coarse lexical/keyword routing followed by the historical Medium/Fine SigLIP-plus-lexical scoring path; small returned evidence budget | Does not use `text-embedding-3-large` Dense Coarse and Dense Medium routing |
| Old H-15 | Same historical lexical-gated hierarchy at a larger returned evidence budget | Does not represent the New Dense Semantic Beam-B H-15 |
| Old H-30 | Historical all-Fine/global SigLIP diagnostic path rather than the later mandatory Dense Coarse → Dense Medium → local Fine path | Does not represent the New Dense Semantic Beam-B H-30 |

The old H-6/H-15 results therefore test a lexical-gated hierarchical implementation. Old H-30 tests a global-Fine SigLIP diagnostic. They are useful for implementation history, failure analysis and showing why the Dense-semantic redesign was necessary. They are not the final “Ours” results.

In particular, the old budget labels are not directly interchangeable with New Beam-B:

- old H-6/H-15 did not apply one common `B` as global Top-B at every hierarchy layer;
- old H-30 could bypass Coarse/Medium routing;
- New H-8/H-15/H-30 uses `text-embedding-3-large` for Dense Coarse/Medium routing and local Fine SigLIP under selected Medium nodes.

## 3. Metric definitions

- **Strict completed**: selected attempt has `status=success`, prediction is exactly one letter A–E, and the completed trajectory contains a nonempty final answer.
- **Correct / 300**: strict completed and prediction equals GT; all 300 frozen UIDs form the denominator, so incomplete/invalid cases count as not correct.
- **Correct / strict completed**: conditional accuracy among completed UIDs only. It must not replace `correct/300`.
- **Timeout**: the final selected attempt has timeout status/error.
- **Success but strict invalid**: runner reports success, but the result fails the strict-completion contract.
- **E2E mean/median/P90/P95**: `metric.elapsed_sec` over strict-completed final-selected attempts only. Timeout and strict-invalid attempts are excluded from this distribution.
- **Final-selected calls/frames**: one selected terminal attempt per UID. When a retry or fallback-correction attempt replaces an older attempt, the older attempt is not added to method-performance totals.

Timeout, illegal prediction and strict invalid can overlap in the original audit definitions and must not be added to derive the number of unfinished UIDs.

## 4. Original old-H results before fallback correction

These values reproduce the historical formal results before the known full-video fallback endpoint defect was corrected:

| Metric | Old H-6 | Old H-15 | Old H-30 |
|---|---:|---:|---:|
| Strict completed / 300 | 280（93.33%） | 287（95.67%） | 257（85.67%） |
| Correct / 300 | 64（21.33%） | 64（21.33%） | 66（22.00%） |
| Correct / completed | 22.86% | 22.30% | 25.68% |
| Timeout | 18 | 11 | 42 |
| Success but strict invalid | 2 | 2 | 1 |
| E2E mean | 279.50秒 | 271.69秒 | 294.69秒 |
| E2E median | 221.08秒 | 216.76秒 | 235.62秒 |
| E2E P90 | 522.96秒 | 487.92秒 | 549.19秒 |
| E2E P95 | 758.70秒 | 699.66秒 | 795.83秒 |

These values remain historically accurate for the originally executed artifacts, but they are superseded for method reporting by the one-for-one corrected view below.

## 5. Corrected old-H canonical view

Thirteen affected selected fallback attempts were rerun after the full-video fallback fix and substituted one-for-one. Original artifacts were not overwritten, and the additional correction compute is not added to the corrected method's final-selected efficiency.

| Metric | Corrected Old H-6 | Corrected Old H-15 | Corrected Old H-30 |
|---|---:|---:|---:|
| Strict completed / 300 | 279（93.00%） | 286（95.33%） | 257（85.67%） |
| Correct / 300 | 61（20.33%） | 63（21.00%） | 66（22.00%） |
| Correct / strict completed | 21.86% | 22.03% | 25.68% |
| Timeout / 300 | 18（6.00%） | 12（4.00%） | 42（14.00%） |
| Success but strict invalid | 3（1.00%） | 2（0.67%） | 1（0.33%） |
| Final not strict completed | 21（7.00%） | 14（4.67%） | 43（14.33%） |

### Correction effect

| Profile | Completed before→after | Correct before→after | Timeout before→after | Strict invalid before→after |
|---|---:|---:|---:|---:|
| Old H-6 | 280→279 | 64→61 | 18→18 | 2→3 |
| Old H-15 | 287→286 | 64→63 | 11→12 | 2→2 |
| Old H-30 | 257→257 | 66→66 | 42→42 | 1→1 |

The correction changed observed answers as well as fallback frame delivery, so the “before” accuracy values must not be silently combined with the “after” efficiency values.

## 6. Corrected latency and operational load

| Metric | Corrected Old H-6 | Corrected Old H-15 | Corrected Old H-30 |
|---|---:|---:|---:|
| E2E count | 279 | 286 | 257 |
| E2E mean | 273.57秒 | 268.43秒 | 288.41秒 |
| E2E median | 218.60秒 | 216.59秒 | 230.07秒 |
| E2E P90 | 493.37秒 | 484.57秒 | 534.11秒 |
| E2E P95 | 694.06秒 | 671.90秒 | 774.23秒 |
| Planner calls | 1,265 | 1,183 | 971 |
| Retrieval calls | 740 | 648 | 446 |
| Summarizer calls | 740 | 648 | 446 |
| Inspector calls | 200 | 197 | 210 |
| Inspector images sent | 11,260 | 11,654 | 12,284 |
| Final-selected cumulative E2E, all 300 terminal attempts | 26.30小时 | 24.59小时 | 31.62小时 |

Interpretation:

- E2E distribution statistics include strict-completed attempts only.
- Final-selected cumulative E2E sums one terminal attempt for every one of the 300 UIDs and therefore includes terminal timeout/invalid elapsed time.
- Inspector image counts do not include local SigLIP-scored frames or text evidence sent to the Summarizer.
- The 13 correction attempts consumed an additional 7,217.17 seconds (2.00 hours). That extra correction cost is archived separately and excluded from the corrected method-efficiency view.

## 7. Corrected old-H versus authoritative Flat

The Flat reference is the authoritative merged Flat result: 254/300 strict completed and 82/300 correct. The following pairings use identical frozen UIDs, count unfinished as not correct, and use the corrected old-H selected attempts.

### Accuracy pairing

| Comparison | Both correct | Flat only correct | Old-H only correct | Neither correct | Old-H−Flat | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| Flat vs corrected Old H-6 | 30 | 52 | 31 | 187 | −7.00 pp | 0.0275 |
| Flat vs corrected Old H-15 | 33 | 49 | 30 | 188 | −6.33 pp | 0.0422 |
| Flat vs corrected Old H-30 | 33 | 49 | 33 | 185 | −5.33 pp | 0.0970 |

These are unadjusted exact p-values. H-6 and H-15 fall below 0.05 individually, but multiple old-H comparisons exist; the archive therefore reports effect sizes and raw p-values rather than declaring a family-wise significant ranking without a prespecified correction.

### Completion pairing

| Comparison | Both complete | Flat only | Old-H only | Neither complete | Old-H−Flat | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| Flat vs corrected Old H-6 | 239 | 15 | 40 | 6 | +8.33 pp | 0.00102 |
| Flat vs corrected Old H-15 | 244 | 10 | 42 | 4 | +10.67 pp | 9.06×10⁻⁶ |
| Flat vs corrected Old H-30 | 221 | 33 | 36 | 10 | +1.00 pp | 0.8099 |

### Paired latency among mutually completed UIDs

| Comparison | Paired UIDs | Flat mean | Old-H mean | Old-H−Flat mean | Difference median | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| Corrected Old H-6 | 239 | 313.15秒 | 260.02秒 | −53.13秒 | −32.54秒 | 0.00127 |
| Corrected Old H-15 | 244 | 316.69秒 | 265.07秒 | −51.62秒 | −41.61秒 | 0.000508 |
| Corrected Old H-30 | 221 | 308.51秒 | 268.34秒 | −40.17秒 | −23.15秒 | 0.0202 |

These paired E2E results describe the complete old VideoSEAL pipelines on mutually completed questions; they do not isolate Retriever-only latency or prove that the old routing algorithm caused the accuracy differences.

## 8. What these data can and cannot support

Supported:

- The historical lexical-gated H-6/H-15 and global-Fine old H-30 were real Eval300 executions with full frozen-UID coverage.
- Under their actual implementations, the old H methods returned fewer correct answers than authoritative Flat, while corrected old H-6/H-15 completed more questions and had lower paired E2E on mutually completed UIDs.
- The full-video fallback defect affected 13 selected attempts and required a corrected one-for-one view.
- The results motivated replacing lexical/global-Fine routing with a fully specified Dense Semantic Beam-B Retriever.

Not supported:

- These old results do not measure the final semantic Hierarchical Indexer.
- They cannot be labeled as New Dense H-6/H-15/H-30 or “Ours final.”
- Old H-30 cannot be used as evidence that Dense hierarchical H-30 was tested, because its routing path was different.
- Lower E2E does not by itself demonstrate better retrieval quality.
- Candidate-count labels across old and new series are not guaranteed to represent the same search budget or evidence unit.
- Parser/control-flow counts from the pre-correction audit should not be combined with the corrected 13-attempt view without reconstructing those events again.

## 9. Recommended naming in reports

Use these exact labels:

- `Old-H6 (lexical-gated; corrected fallback view)`
- `Old-H15 (lexical-gated; corrected fallback view)`
- `Old-H30 (global-Fine SigLIP diagnostic; corrected fallback view)`

Reserve the labels `Dense H-8`, `Dense H-15` and `Dense H-30` for the later New Dense Semantic Beam-B experiments.

Recommended one-sentence summary:

> The earlier H-series is retained as a historical diagnostic: H-6/H-15 used lexical-gated routing and H-30 used a global-Fine SigLIP path, so their corrected Eval300 results document the limitations of the preliminary integration but do not constitute results of the final Dense Semantic Beam-B Indexer.

## 10. Authoritative sources

- Original unified reanalysis: `outputs/eval300_authoritative_reanalysis_20260826T114630Z/`
- Original formal old-H artifacts: `outputs/formal_eval300_20260821T095418Z/`
- Fallback correction and corrected old-H view: `outputs/full_video_fallback_fix_20260826T145534Z/corrected_eval300/`
- Architecture lineage statement: `outputs/dense_semantic_hierarchical_protocol_candidate_20260827T192035Z/SOURCE_LINEAGE.md`

This archive was created from existing artifacts only. No model, GPU, API, retry, scoring run or experiment was started.

