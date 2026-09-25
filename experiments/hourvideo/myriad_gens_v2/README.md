# GenS V2 Eval300 selection staging

This is a Myriad-local, GitHub-ready staging of the formal GenS V2 Eval300 frame-selection run. It is evidence for the selector only: it contains no downstream A-E predictions and no downstream API outputs.

## Primary artifacts

- `data/selected_frames_downstream.jsonl`: the lightweight 300-question table intended for downstream frame loading.
- `audit/selection_provenance.jsonl`: audit-only selection telemetry; do not send this file to an answer API.
- `audit/question_audit_300.jsonl`: one bounded audit row per formal question, including hashes of the original per-question artifacts without copying raw responses or token IDs.
- `reports/FINAL_SELECTOR_REPORT.json`: the original formal V2 aggregate report.
- `reports/STAGING_VALIDATION.json`: independent local cross-check against all 300 formal per-question outputs.
- `provenance/COPY_PROVENANCE.jsonl`: source absolute path, source SHA-256, copy SHA-256, role, and experiment for every byte-copied file.

## Reproduction identity

- Official GenS source: `https://github.com/yaolinli/GenS`, commit `c5a60f59a2011e6008e90e0046228f82f47d10fd`.
- GenS model: `yaolily/GenS-qwen2d5-vl-3b`, revision `6454c5d7b034b57d61c84f4303f43610233eca80`.
- Formal V2 profile SHA-256: `3479770e71ff5f68f0669dc6b51ce3db1922fe8ae39b2f7a0ff493634aa8d0f6`.

The exact local wrapper/config/launcher files are retained byte-for-byte. Local absolute paths are provenance, not credentials; see `reports/SENSITIVE_SCAN.json`. No prompt or selection result was redacted or rewritten.

## Exclusions

No videos, JPEG frames, CLIP embeddings, model weights, caches, raw GenS responses, generated token IDs, credentials, or full third-party source tree are included. Model/source manifests record identities without including those payloads.

No model or API was run while creating or validating this staging. Nothing was transferred to another machine, committed, pushed, or uploaded.
