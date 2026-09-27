# API Planner + Local Eval300 — GitHub import staging

This package archives the thesis-authoritative HourVideo formal experiment
`hourvideo_v6_6_2_planner_api_ablation_formal_eval300_v1`. It was assembled
from existing school-server artifacts without running a model, API, scoring
job, or experiment. Source files were not modified, moved, or deleted.

## Thesis-authoritative result

| Metric | R1 | R3 |
|---|---:|---:|
| Correct / fixed Eval300 denominator | 78 / 300 | 77 / 300 |
| Planner API cost | $4.965457 | $10.096273 |
| Planner + post-Planner total tokens | 42,709,209 | 36,680,380 |
| Planner-only input tokens | 2,870,342 | 8,510,653 |

The 42.71M and 36.68M figures are **full-chain token totals**: Planner API
tokens plus the local post-Planner Shared/Fine/Final workflow. They are not
Planner-only input-token counts. The Planner-only input-token counts are
2,870,342 and 8,510,653 respectively. These two accounting scopes must not be
interchanged.

## Archive layout

- `artifacts/formal_run/canonical_summary_v2/`: canonical results, reports,
  cost breakdowns, validation, and the authoritative raw-closure manifest.
- `artifacts/formal_run/manifests/`: formal Eval300 and downstream manifests.
- `artifacts/formal_run/telemetry/`: Planner request, attempt, and budget
  ledgers.
- `artifacts/formal_run/per_question/`: 300-question result, Planner,
  downstream-stage, route-status, and model-attempt evidence needed to audit
  the reported accuracy and usage.
- `artifacts/configs/`, `artifacts/frozen_prompts/`, `artifacts/eval300/`:
  formal configuration, complete frozen prompts, and the ordered Eval300 UID
  list.
- `artifacts/source/`, `artifacts/scripts/`: formal runtime source, direct
  imported dependencies, launch script, and canonical aggregation script.

## Raw closure and portability

The original formal raw closure contains 5,966 files and 184,533,682 bytes,
with tree SHA-256
`020ad1a5d12805a3bf03c5d719911aa9fc98deed7c3be1d89e5bf0218c37bcd8`.
This staging verifies that closure against the school source. To avoid
duplicating large intermediate payloads, the upload tree keeps the necessary
per-question evidence while the complete original file inventory and hashes
remain embedded in `canonical_summary_v2/canonical_manifest.json`.

Absolute school-user path prefixes in uploadable text files were replaced by
portable tokens only. `SOURCE_MANIFEST.csv` records both original and staged
SHA-256 values and identifies every transformed file. No prompt, answer,
numeric result, or recorded hash value was intentionally changed. Exact local
source paths are isolated under `DO_NOT_COMMIT/`.

See `VERIFICATION.md`, `SENSITIVE_SCAN.md`, and `INCLUDED_EXCLUDED.md` before
importing this package.
