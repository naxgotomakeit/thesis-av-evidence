# ABD formal candidate files and recovery contract

This is an offline candidate. It does not authorize provider execution or a US$30 budget.

## Candidate implementation

- `config/abd_formal_candidate_v1.json`
- `config/abd_draft_v1.json`
- `config/abd_common_system_prompt_v1.txt`
- `src/abd_draft_v1/core.py`
- `src/abd_draft_v1/provider.py`
- `src/abd_draft_v1/store.py`
- `src/abd_draft_v1/formal_runtime.py`
- `scripts/preflight_abd_formal_candidate_v1.py`
- `scripts/run_abd_formal_v1.py`
- `scripts/score_abd_formal_v1.py`
- `tests/test_abd_draft_v1.py`
- `tests/test_abd_formal_v1.py`

Exact SHA-256 values and historical source relationships are stored in `formal_candidate_manifest.json`. `freeze_record.json` binds that manifest to the implementation commit. The three input manifests are `../inputs/A.jsonl`, `../inputs/B.jsonl`, and `../inputs/D.jsonl`.

## Launch gate

The generation entry defaults to offline validation. Real execution additionally requires all of:

1. `--execute`;
2. the exact approval token printed in `PREFLIGHT_REPORT.md`;
3. a separately reviewed authorization JSON matching experiment ID, authorized budget, config SHA, and candidate-manifest SHA;
4. the frozen live Python interpreter;
5. the exclusive experiment runner lock.

No authorization JSON exists in this candidate.

## Resume behavior

- `terminal_success` and `terminal_failed`: skip permanently.
- saved provider response but no terminal status: reconstruct usage/cost/parser result and terminal artifact without another provider call.
- request-start without a saved provider response: set `pending_provider_outcome_review`, retain the budget hold, and do not resend.
- two simultaneous runners: the second fails immediately on the exclusive lock.

Generation never loads gold and never imports or invokes the independent scorer.
