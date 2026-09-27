# Direct R1 visual-only result and ASR lineage

English presentation summary, not a replacement experimental record. The linked frozen originals remain authoritative for byte-level provenance. Numerical values and artifact identities reproduced here are copied without alteration; this summary does not introduce new results or revise frozen labels. See the [source register](../ENGLISH_READING_GUIDE.md) for original-file hashes and checksum membership.

## Thesis-authoritative result

[Frozen thesis data summary](../../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/thesis_data_package/DIRECT_VISUAL_ONLY_EVAL300_DATA_SUMMARY.md).

| Condition | Correct / 300 | Completed |
|---|---:|---:|
| R1 visual-only | 85 | 299 |
| Frozen R3 visual-only | 103 | 298 |

The corrected R1 consists of 175 rerun routes and 125 SHA-verified reused routes. R3 retains its frozen 300 routes. The common UID population has no gaps or duplicate questions. Failures remain incorrect. The paired table is 51 both correct, 34 R1-only, 52 R3-only and 163 neither correct.

Earlier formal R1 had 88/300 correct and 300 predictions. The correction is a genuine later execution, not an offline inference of 85 from 88. The intended input change removes nonempty `coarse_regions[*].audio_channel`; other visual/structural map fields are unchanged. Five unaffected video maps remain byte-identical. The retained schema label does not itself indicate nonempty audio observations.

Observed answer transitions are 71 correct→correct, 17 correct→incorrect, 14 incorrect→correct and 198 incorrect→incorrect. Sixty final answers changed, all among the 175 rerun routes. Cross-time provider nondeterminism and transport/recovery changes prevent attributing all changes solely to ASR removal.

Known complete-method route costs are $15.02991285 for R1 and $13.00064950 for R3. R1 has 1,956 recorded sends and 1,951 confirmed responses; five network outcomes have unknown usage/cost/latency. The $1.26280000 formal reserve is not actual expenditure. The 175 corrected routes alone cost $8.53059750, which must not replace the full R1 cost. Route durations aggregate multiple execution segments, not one continuous Eval300 wall-clock run.

Historical evidence-support labels or frame-selection analyses depending on the rerun routes cannot automatically transfer to corrected R1. The frozen source also records missing historical full-frame hashes and absence of proof of a byte-identical provider model across dates.

[Per-question results](../../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/final_outputs/scored_results_with_gold.json) / [reuse manifest](../../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/manifests/route_reuse_manifest_v8.json).

## Earlier ASR input audit

[Frozen audit dated 2026-09-13](../../experiments/hourvideo/direct_r1_visual_only_thesis/artifacts/lineage/frozen_sources/DIRECT_R1_ASR_INPUT_AUDIT.md). Seven of twelve R1 maps had nonempty ASR: 1,007 distinct attached audio IDs, 1,026 Coarse references and 21,897 deduplicated text characters. This exposed 175/300 R1 routes; the other 125 R1 routes and all 300 R3 routes had no nonempty ASR.

The earlier experiment-specific contract intentionally used structural/ASR-oriented R1 maps. The audit identified a mismatch with the subsequently specified all-visual-only thesis condition, not accidental runtime injection. Exposure does not establish that the model used ASR or that ASR affected accuracy.

Map and code hashes demonstrate that the provider loaded the full map into system input without filtering. Complete request bodies were not retained, so this is an implementation-and-input provenance inference rather than direct request-body replay.

Audio lineage follows matching video UIDs, WAV hashes and local Whisper-small/CPU records. The audit did not perform human listening and did not rehash all current MP4 bytes; it therefore cannot establish speech-level correctness for every transcript segment. Repetition, punctuation-only text, mixed-script text and out-of-range timestamps are quality flags, not proof of hallucination. The original transcript examples are raw evidence and remain untranslated.

The audit's proposed future checks describe its historical stage. The later correction and thesis result above provide the subsequent lineage; neither frozen document is rewritten to erase that sequence.
