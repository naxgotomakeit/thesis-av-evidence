# Included and excluded material

## Included

- Complete `direct_visual_only_eval300_thesis_data_v1` package, excluding only
  Python bytecode/cache.
- Recovered Eval300 final outputs, final report, and `FINAL_AUDIT_SHA256`.
- Route-reuse v8, source identity, input/map/frame inventories, and lineage
  manifests.
- v8-v13 formal/recovery manifests, locks, validations, ledgers, execution
  records, and the 175 rerun route artifacts/statuses.
- All 600 old-formal route artifacts/statuses plus old candidate, launch lock,
  formal manifest, and canonical summary. These are segregated and labelled
  `SUPERSEDED_FOR_THESIS`; they also supply the SHA-verifiable sources for 125
  reused R1 routes and frozen R3.
- Final prompt/runtime/config, v8-v13 run/validation scripts, merge/scoring
  scripts, ASR input audit, and frozen original launch identity.

## Excluded

- Request payload image bodies, provider-response payload copies, route
  checkpoints, and redundant journals not needed after route artifacts and
  ledgers were retained.
- Historical smoke/pilot attempts and their result directories.
- Videos, extracted frames, model weights, caches, virtual environments,
  Python bytecode, and unrelated experiments.
- `.env`, API keys, access tokens, passwords, and credential contents.

The five formally unknown provider outcomes remain disclosed in the source
reports; no missing token, cost, response, or latency value was zero-filled.
