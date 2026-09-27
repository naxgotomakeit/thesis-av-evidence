# Full Staged R3 versus Direct R3 paired-100

English presentation summary, not a replacement experimental record. The linked frozen originals remain authoritative for byte-level provenance. Numerical values and artifact identities reproduced here are copied without alteration; this summary does not introduce new results or revise frozen labels. See the [source register](../ENGLISH_READING_GUIDE.md) for original-file hashes and checksum membership.

[Frozen thesis report](../../experiments/hourvideo/school_formal/content/supplementary/full_staged_paired100/reports/thesis_package/PAIRED100_THESIS_RESULTS.md).

This supplementary comparison uses a fixed subset of 100 questions from 12 videos. Full Staged reuses frozen Planner outputs and executes Shared/Fine/Final stages; Direct reuses frozen Direct-v1.2 R3 routes.

| Metric | Full Staged downstream | Reconstructed full Staged | Direct R3 |
|---|---:|---:|---:|
| Correct / 100 | 24 | 24 | 32 |
| Predictions | 97 | 97 | 99 |
| Logical calls | 478 | 578 | 487 |
| Physical requests | 662 | 762 | 487 |
| API cost | $11.195644600 | $14.494889600 | $4.301863350 |

The historical Planner contribution is $3.299245000, a recorded estimate. Its 2,853,515 input tokens lack an ordinary/cache split; do not fabricate precise combined token categories. Cross-run reconstructed latency is not measured end-to-end wall time.

The paired outcomes are 15 both correct, 59 neither correct, 17 Direct-only and 9 Staged-only. Exact two-sided McNemar p=0.1686375439 is not significant at 0.05 and does not prove equivalence. Questions within videos may be correlated, limiting question-level independence.

The frozen rule permits two validation retries after the initial attempt, hence three attempts. Earlier wording claiming two total Final attempts is a documentation error; the original text is retained. The actual paired-100 run had zero transport retries. Two Final failures exhausted three semantic validation attempts; another failed on the missing `uncertainty` field after validation. Direct's historical failure exhausted its 32-turn limit.

Staged transmitted 1,424 images including validation retransmissions and reviewed 1,312 per-question-deduplicated images. Direct transmitted 928 new unique images. Staged's 16-image limit applies to a Fine batch; Direct's 16-image limit is per question. These are different budget semantics, and the systems also differ in reasoning topology and retrieval responsibility.

[Per-question table](../../experiments/hourvideo/school_formal/content/supplementary/full_staged_paired100/reports/thesis_package/paired100_per_question.csv) / [archive manifest](../../experiments/hourvideo/school_formal/content/supplementary/full_staged_paired100/results/analysis/archive_manifest.json). This is distinct from paired-150, which is a read-only extraction/recomputation from capacity-aware Eval300.
