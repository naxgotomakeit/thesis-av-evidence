# Flat-15 retry recovery execution amendment

- Parent audit: `retry_infrastructure_audit_20260913T115931Z`
- Scientific configuration is unchanged: Flat Top-K 15, concurrency 1,
  max steps 16, per-question hard timeout 1000 seconds, frozen prompts/models,
  frozen 57-UID retry manifest.
- `retry_raw_v1_20260913T041000Z` is diagnostic only and is never used for
  skip/resume/merge decisions.
- The credential loader restores the frozen entrypoint's existing external
  credential-file loading step. No credential value is logged.
- `embedding_usage_proxy.py` is observation-only: it delegates the exact
  frozen `embeddings.create` request and records response-reported token usage,
  timing and cost without recording query text or credentials.
- The smoke alone sets `MLLM_RETRY_TIMES=1` to enforce its one-Summarizer-call
  cap. Formal retry retains frozen `MLLM_RETRY_TIMES=3`.
- Formal outputs stop at a gold-free retry raw closure pending freeze. No
  backfill, aggregation, gold loading, or scoring runs in the controller.

