# Thesis demonstrations

Python 3.10 or newer is required. Both replay modes use only the standard library:
no installation, credentials, network, GPU or benchmark media is needed.

| Mode | Demonstrates | New inference |
|---|---|---|
| `replay-staged` | API Planner plus local staged navigation, evidence and answering | No |
| `replay-direct` | Direct prepared-context identity, tool use and answering | No |
| `live` | Future full fresh execution; not implemented | No execution available |

## Staged replay

```bash
python3 demo/run_demo.py --mode replay-staged
```

**API-Planner + Local Downstream Frozen Pipeline Replay** uses the preserved HV2
R3 technology-objects question. It displays Planner rationale and region selection,
coarse/medium/fine evidence excerpts, inspection findings, investigation status,
final answer and separately scoped resource telemetry.

It is not a fully-local Planner run. The School capacity-aware archive retains
status and telemetry but excludes the complete Planner outputs and downstream
content needed for an equally complete replay. No missing decisions are invented.

The [staged manifest](examples/local-staged-replay.json) pins existing source
files. Its historical filename is retained; the presentation and CLI identify
the actual API-Planner/local-downstream split. The full source map is not loaded.
The frozen answer B, “Phone, electric kettle”, is shown without inferring
correctness from the termination label `confirmed`.

## Direct replay

```bash
python3 demo/run_demo.py --mode replay-direct
```

The question asks which item organized grapes on a kitchen counter:
`R3:db3f7933-dfa0-4678-9d4f-393b628ded45_17_13`.
The recorded execution inspected frames at 142, 157 and 172 seconds, then answered
**E — Bowl**. The final scored artifact marks this answer correct. There were two
successful API attempts, three images and no structural corrections.

The [Direct manifest](examples/replay-direct.json) binds route trace/status,
thesis reuse/scoring records and the archive's original-to-staged SHA mapping.
The trace is stored under `SUPERSEDED_FOR_THESIS/old_formal_88`, but this R3 route
is explicitly retained as `frozen_reuse` in the final thesis lineage. The earlier
R1 authority must not be inferred from this R3 storage location.

Coverage boundaries are visible in the terminal:

- Full source map exists in the archived Variant-C package but is not loaded by
  this compact replay. The replay displays its recorded identity/hash.
- The original image-bearing `tool_result` payload is not locally preserved.
  The replay shows recorded resolved-frame identities, hashes, image counts and
  accepted controller disposition instead; it does not recreate a tool message.
- Question/options text is taken from the matching question ID in the frozen HV2
  record. It is not presented as the omitted original Direct input JSON.
- Final rationale is an actual recorded model statement, not an independent
  reinspection of the images.

Thus this is a replay of preserved execution records, not a complete request-body
or media replay. No images or hidden reasoning are reconstructed.

## Future live mode

```bash
python3 demo/run_demo.py --mode live
```

This exits with status 2 and:

```text
FULL LIVE MODE IS NOT IMPLEMENTED YET.
See docs/USAGE.md for the planned scope and reproduction requirements.
```

The earlier experimental live adapter has been removed. There is no API transport,
key-loading, asset-loading or output-saving path in the current demo. Obsolete
`--mode replay`, `--mode live-direct` and live-only flags are rejected.

A small prepared-request continuation may be technically feasible, but it is not
the definition of future `live`: that mode means full fresh execution. See the
[usage guide](../docs/USAGE.md) for prerequisites and distinctions.

## Integrity and offline tests

Both replays verify source SHA256 before presentation and fail closed on missing
sources or identity conflicts. Direct also checks R3 method/route agreement,
unique exact scoring/reuse matches, original-to-staged provenance, recorded frame
hash agreement, tool actions, predictions and scoped resource totals.

```bash
python3 -B -m unittest discover -s demo -p 'test_*.py'
python3 demo/run_demo.py --help
```

Tests block network connections and exercise corrupted in-memory records and
temporary manifests, never modifying frozen artifacts. Neither replay writes
results or imports the experimental runtime.
