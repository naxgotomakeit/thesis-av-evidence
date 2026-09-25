# New Dense H-8 Eval300 Finalization Amendment

## Authority

- Amendment status: `COMPLETE / VALID`
- Parent formal run: `dense_semantic_beam_b_h8_eval300_formal_v3_20260831T201510Z`
- This amendment is additive. It does not modify, replace, or overwrite any parent artifact.
- The parent `h8/status/state.txt` remains byte-for-byte unchanged and contains `FORMAL_RUNNING`.

## State resolution

The parent state label is a non-data-bearing finalization omission. The following independent terminal evidence proves completion:

- `finished_utc.txt`: `2026-09-02T11:18:44Z`;
- first pass coverage: 300/300 frozen UIDs;
- frozen retry list: 46 unique UIDs;
- retry coverage: 46/46;
- merged manifest: 300 unique UIDs;
- final report and immutable merged manifest exist;
- Planner and Inspector processes exited and ports 18082/18083 were released.

Therefore the authoritative archival status is:

```text
COMPLETE / VALID
```

The stale parent `state.txt=FORMAL_RUNNING` must not be interpreted as an active or incomplete experiment and must not be silently edited.

## Scope

This amendment changes no UID, attempt selection, prediction, score, latency, trajectory, runtime, Retriever, configuration, or manifest in the parent run. It only records the verified terminal status and points to the canonical report in this directory.

