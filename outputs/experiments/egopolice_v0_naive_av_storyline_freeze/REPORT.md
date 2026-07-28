# V0 freeze report

The EgoPolice V0 baseline is frozen as **V0 — Naive Audio-Visual Storyline Baseline**. No model was started and no new long video was processed for this freeze.

Four completed representative runs are preserved by provenance: 822714139, 970733198, 540772226, and 382630103. Each has visual event/caption/storyline results, audio speech/ASR/alignment results, a prompt snapshot, structured answer, validation report, run manifest, and API call record. The answer outputs passed the current local validator at freeze time.

The freeze records the exact configuration/code hashes, model names and snapshots, API contract/retry policy, dependency environment, known limitations, and external runtime locations. Large runtime artifacts are not copied into the repository.

## Verification scope

- Configuration and prompt source paths: checked.
- Structured answer validation: checked for all four representative runs.
- API records: checked for secret markers; API keys are not included.
- `.env`: ignored by Git; `.env.example` remains separate.
- Freeze output: checked for large media/model/cache files.
- Tests and Python compilation: run before staging.
