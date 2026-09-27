# Included and excluded material

## Included

- Complete `canonical_summary_v2`, including thesis statistics and the raw
  closure inventory.
- Formal/downstream manifests and all Planner telemetry ledgers.
- For all 300 questions: blind answers, Planner artifacts, final answers when
  present, route statuses, direct-final inputs/raw outputs, shared-stage
  artifacts, route events, and model attempt/request-start records.
- Formal and inherited configs, frozen prompt texts and prompt manifest,
  ordered Eval300 UID list, relevant source modules and direct dependencies,
  launch and aggregation scripts.
- Portable/source manifests, checksums, verification, and sensitive scan.

## Excluded

- `live_model_calls.json` and `shared_attempt_audit.jsonl`: duplicated large
  intermediate logs; their original SHA/size provenance remains in the
  canonical raw-closure manifest.
- Videos, extracted frames, maps/index payloads, model weights, caches,
  virtual environments, bytecode, and unrelated experiments.
- `.env`, API keys, access tokens, passwords, and credential files.
- Pilot/gate/smoke outputs outside the formal Eval300 namespace.

Exclusion does not reclassify or alter any original artifact. The school-side
raw closure remains intact and was verified read-only.
