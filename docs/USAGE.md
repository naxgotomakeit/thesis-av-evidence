# Usage Guide

## Quick start

Clone this repository, enter its root and use Python 3.10 or newer. The two
replays require no additional Python packages, API key, GPU, media or network.

### Inspect thesis evidence

Start with the [README](../README.md), then the
[thesis artifact index](../THESIS_ARTIFACT_INDEX.md). Follow a result's specific
aggregate, per-question and freeze/provenance links. Consult
[result authority](RESULT_AUTHORITY.md) before comparing versions.

### Replay the staged pipeline

```bash
python3 demo/run_demo.py --mode replay-staged
```

This is **API-Planner + Local Downstream Frozen Pipeline Replay**, not a
fully-local Planner execution. It displays the preserved HV2 R3 example about
technology objects: question/options, recorded Planner rationale, selected
regions, hierarchical evidence, inspection finding, investigation status,
answer and historical resource use.

The School local capacity-aware package retains formal statuses and telemetry,
but not a comparably complete content trace. The demo therefore retains HV2
rather than fabricating the missing local decisions.

### Replay the Direct pipeline

```bash
python3 demo/run_demo.py --mode replay-direct
```

The Direct R3 example asks which item organized grapes on the kitchen counter.
One recorded `inspect_frames` call selected frames at 142, 157 and 172 seconds;
the next recorded action answered **E — Bowl**. It completed without correction
and is marked correct in the final scored artifact.

Unlike the staged pipeline, Direct uses prepared native context and a
single-agent tool-use/answer protocol. Its inspection rationale, exact tool-call
identity, resolved-frame timestamps/hashes, final rationale and telemetry are
read from preserved artifacts, not from a hand-authored transcript.

### Reserved replay-launcher live mode

```bash
python3 demo/run_demo.py --mode live
```

This exits with status 2 and explicitly states that full live mode is not
implemented through this launcher. It does not read credentials or assets and cannot send API requests.
The earlier `live-direct` adapter and its flags are no longer supported.

## What the replays show

| Property | Staged replay | Direct replay |
|---|---|---|
| Process | Planner, selection, shared/fine investigation, answer | Prepared-context identity, inspection tool call, answer |
| Example | Phone and electric kettle | Grapes organized in a bowl |
| Available rationale | Planner, investigation and final-answer text | Inspection request and final-answer text |
| Map coverage | Preserved excerpts and recorded map hash | Recorded map identity/hash; archived full map is not loaded |
| Media | No source images loaded | No source images loaded |
| Correctness | Not asserted by replay | Explicitly recorded in final scored results |

Outputs are compact; excerpt truncation is labelled and repository-relative
source paths remain visible. They work from another working directory when the
script is addressed by its full path. Neither command creates output files.

Direct's `RECORDED TOOL RESULT` section reports saved controller disposition,
resolved-frame identities and image counts. The image-bearing tool-result
payload itself is **not** locally retained and is not regenerated. Its final
rationale must not be mistaken for separately preserved visual-observation text.

Direct question/options text comes from the same question ID in the frozen HV2
record; the original Direct input JSON remains omitted. No HV2 answer or
investigation evidence enters the Direct replay. The final answer and correctness
come exclusively from Direct trace/scoring records.

Although the Direct trace resides in a historical directory named
`SUPERSEDED_FOR_THESIS/old_formal_88`, this specific R3 route is explicitly
retained as `frozen_reuse` in the thesis-authoritative merged results. The demo
checks that binding and the original-to-staged SHA mapping, rather than deciding
authority from the directory name.

## Recorded rationale versus hidden reasoning

Only decision/rationale text explicitly preserved by the original run is shown.
The demos do not expose or reconstruct hidden chain-of-thought. Missing
Unpreserved SEARCH_MORE messages and tool-result bodies remain marked missing.
Full source map exists in the archived Variant-C package but is not loaded by
this compact replay. See the [archived map](../experiments/hourvideo/dgx_eval300/shared/dense_index/indexer_build_provenance/organizer_stage/formal_variant_c/cases/db3f7933-dfa0-4678-9d4f-393b628ded45/parsed_map.json).
A final investigation snapshot is not a complete sequence of all model messages.

Recorded resource use belongs to the historical run, not the replay. Staged
Planner and post-Planner totals remain separate; their latencies are not added
into a claimed continuous end-to-end measurement. Direct ordinary input,
cache-write input and cache-read input remain separately labelled.

## Planned full live scope and reproduction requirements

Future `live` means a **full fresh execution**: a new question/context session,
new model decisions, authorized evidence access and a new answer. It must never
be labelled a thesis result or write into preserved evidence. It is not
implemented in this iteration.

The existing Direct protocol uses an Anthropic provider and dynamically selected
timestamps. A future implementation needs authorized benchmark media/frames,
the exact prepared question/options and native map, verified asset bindings,
provider access, explicit cost controls and safe output isolation. Full staged
execution additionally needs suitable indexes/embeddings, model weights and model
services. These prerequisites are not supplied by a repository clone alone.

Earlier feasibility work identified a three-JPEG prepared Direct continuation.
That would reuse a historical inspection decision and issue only the next
request; it is **not full fresh live execution**. A fresh model can ask for other
timestamps, so a tiny historical subset cannot silently replace the complete
available evidence. No continuation adapter remains executable in this demo.

Benchmark media is not redistributed. Do not copy images, weights, credentials
or unlicensed assets into this repository. See [attribution](ATTRIBUTION.md).

## Verify thesis evidence

- **Direct R1:** follow [Table 5.5 in the index](../THESIS_ARTIFACT_INDEX.md#hourvideo-results)
  to scored rows and reuse/freeze records. Thesis authority is 85/300 with
  299 completed; earlier 88/300 remains a separate preserved formal result.
  The R3 demo does not replace either R1 lineage.
- **HV2 API Planner:** follow the same index to canonical summary v2. Accuracy is
  78/300 and 77/300. Full-chain tokens are 42,709,209 / 36,680,380; Planner-only
  input is 2,870,342 / 8,510,653; Planner API cost is USD 4.965457 / 10.096273.
  These are different scopes, not the single-example replay figures.
- **EgoPolice:** follow [direct figure provenance](../THESIS_ARTIFACT_INDEX.md#egopolice-direct-figure-provenance)
  to Figure 5.3(c), `q_medical_assistance`, format-only recovery, binding and
  checksums. `q_handcuff_before_medical` is a distinct identity.

The thesis determines the reported version; frozen artifacts determine each
run's contents. Discrepancies must be recorded rather than suppressed.

## Reproduction boundaries

Audit reproduction is not semantic re-review without source media. Source SHA
validation proves the presented files match the pinned bytes, not that a model's
visual claim is true. Direct's displayed JPEG hashes are recorded expected/observed
values, not a new verification of unavailable images.

School, DGX and Myriad artifacts preserve formal executions; they are not a
portable application guaranteed to rerun on any laptop. See
[reproduction](REPRODUCTION.md) and [Git history](GIT_HISTORY.md).

## Troubleshooting

- Use `--help` for the three current mode names; old names are rejected.
- Missing source or SHA mismatch: stop and restore the correct repository
  version. Do not edit a frozen package or bypass the manifest.
- Identity conflict: stop; R1 and R3 records are not interchangeable.
- `--mode live` returning status 2 is expected, not an API failure.
- There are no asset/key/output flags in this iteration. CLI errors deliberately
  do not echo supplied values, which could accidentally contain a secret.
- Run offline tests with
  `python3 -B -m unittest discover -s demo -p 'test_*.py'`.
