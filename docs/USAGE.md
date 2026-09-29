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
The new portable prototype uses separate `prepare.py` and `ask.py` commands below.

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

## Portable API-assisted execution

Use Python 3.10+ and install `ffmpeg`/`ffprobe` through your operating system.
No Python packages or local GPU are required. Only use media you are authorized
to process and send to the API provider. This prototype accepts videos up to
10 minutes and workspaces strictly below the repository's ignored `demo_runs/`.
No benchmark media is redistributed.

### Prepare and approve

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/example --caption-backend api
```

This runs local 1-FPS extraction (JPEG quality scale 6, no resize), checksums every
frame, builds 15-second Fine and 45-second Medium windows, and constructs caption
requests without sending them. Representatives are Fine-center frames in temporal
order, including partial final windows. Effective hierarchy duration is the frame
count in seconds; container duration is recorded separately. Coarse count is
determined by Organizer output, not fixed.

The printed plan reports duration, frame/window counts, paid call counts, models,
cost assumptions and a plan SHA. Review it before authorizing execution. Set
`ANTHROPIC_API_KEY` through your environment or secret manager; never pass a key
as a command-line argument. No `.env` file is read or created.

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/example --caption-backend api --execute --approve-plan <reviewed-plan-sha256>
```

Execution makes one caption request per Medium and one Organizer request. The
action-preserving caption prompt is pinned; its API backend is a **portable
substitute** for Qwen2.5-VL-7B-Instruct, not an exact thesis reproduction.
Organizer uses the exact frozen Variant-C system prompt, schema, temperature,
model and output-token cap. Caption text is not fabricated during dry-run:
Organizer's actual input can only be constructed after captions complete.
Invalid/incomplete Organizer output stops preparation; no fallback segmentation
or automatic repair is applied. Native R3 conversion reuses the pinned pure
conversion function without importing its server-dependent surrounding module.

Every workspace and plan records `preparation_profile=PORTABLE_API`,
`caption_backend=API`, actual `caption_model=claude-haiku-4-5-20251001`,
`thesis_caption_model=Qwen2.5-VL-7B-Instruct`,
`model_identity_matches_thesis=false`, `full_historical_embedding_index=false`,
and `direct_targeted_preparation=true`. `local-qwen` and `remote-qwen` are reserved
backend interfaces and currently exit without execution.

### Ask a fresh question

After successful preparation, create your own question JSON with no gold labels:

```json
{"question_id":"my-question-1","question_text":"Which object was moved?","answer_options":{"A":"Cup","B":"Bowl","C":"Plate","D":"Spoon","E":"Bottle"}}
```

```bash
python3 demo/ask.py --workdir demo_runs/example --question-json question.json
python3 demo/ask.py --workdir demo_runs/example --question-json question.json --execute --approve-plan <reviewed-ask-plan-sha256>
```

The first command verifies the READY workspace and constructs the original Direct
system/map/question/tools without inference. The second, separately approved
command starts a fresh Direct conversation, with no historical answer or inspection
decision. It imports the preserved loader, resolver, controller and Anthropic agent.
All extracted frames remain available for new timestamp requests (three new images
per turn, sixteen unique images total, at most 32 controller turns). It uses the
preserved smoke-profile USD 1 accounting budget, not an Eval300 launch. A transport
safety wrapper refuses redirects and resends after an uncertain network failure;
this deliberately tightens historical retry behavior without changing prompts or
action semantics. The budget is accounting-based, not a provider billing guarantee.

Prepare once, ask many: each new MCQ starts an empty Direct session. The public
A–E object is normalized into the archived loader's ordered option rows through
a temporary file; no prompt, options or answer semantics are changed. Only the
three illustrated top-level fields are accepted. Gold labels, historical state
and open-ended questions are rejected. The input question is not added to the
question-independent preparation workspace.

ASK plans and new answers are saved only in the ignored sibling
`demo_runs/example-outputs/qa_runs/<plan-sha>/`.
They are **not thesis results** and have no asserted correctness. The prepared
workspace is unchanged. Exact repeated execution plans are locked against
accidental paid reruns; use a new question identity for a deliberately new execution.

### Workspace and integrity

`plan.json`, `request_preview.json`, `shared_hierarchy.json`, `frame_sha256.json`
and `frames_1fps/` are generated before approval. Paid preparation additionally
creates `caption_responses/`, `medium_captions.json`, `organizer_request.json`,
`organizer_response.json`, `organizer_output.json`, `r3_2_navigation_map.json` and
`preparation_manifest.json`. `EXECUTION_STARTED.json` prevents automatic resume
after a potentially billed failure. Do not delete the marker to force a retry.
Review partial execution manually and explicitly authorize a new workspace.

Relative paths permit relocation; the Direct resolver receives absolute paths
only in memory. The READY manifest hashes all prepared files; ASK checks those
hashes and the pinned implementation before execution. Source-video SHA, actual
model identities and source-code SHA bindings remain recorded. Do not edit READY
workspaces. They contain user media and model responses: keep them private.
The plan records the source-video SHA, ffmpeg version and relocatable extraction
command/config. Schema-v2 workspaces do not automatically migrate older prototype
layouts. A matching unexecuted plan may be reviewed or explicitly approved again;
an unrelated, incomplete or already executed workdir is refused.

### Planning costs and limitations

At 120 seconds: 120 frames, 8 Fine, 3 Medium, 3 caption calls + 1 Organizer call.
At 300 seconds: 300 frames, 20 Fine, 7 Medium, 7 caption calls + 1 Organizer call.
Approximate preparation ranges are USD 0.0067–0.0246 and USD 0.0146–0.0555 respectively
(heuristics, not measured bills; use the printed plan for computed values).
ASK is adaptive: budget for roughly 2–8 calls and USD 0.01–0.20 for short examples,
but it may use one call or more than eight; the transport ceiling is 66 attempts.
The original Direct accounting budget is USD 1. Preparation also prints a much
more conservative token-cap accounting allowance; Organizer retains its original
64k output cap. These estimates do not guarantee future model availability or billing.
Pricing assumptions are [Anthropic Haiku 4.5 input/output rates](https://platform.claude.com/docs/en/about-claude/pricing):
USD 1 / 5 per million tokens, with Direct cache-aware pricing retained separately.

Full historical preparation still requires omitted embeddings/index components
and historical caption models. This Direct-targeted prototype does not build a
SigLIP index, implement a local staged Planner, or recreate thesis results.
Optional Whisper audio preparation is described below; ASK does not inspect raw audio.

## Further reproduction requirements

The portable profile performs a **fresh execution**: a new question/context session,
new model decisions, authorized evidence access and a new answer. It must never
be labelled a thesis result or write into preserved evidence. It is not
an exact thesis-preprocessing reproduction: the portable profile above substitutes
API captioning and is explicitly labelled accordingly.

The existing Direct protocol uses an Anthropic provider and dynamically selected
timestamps. The prototype requires authorized user media, newly prepared native
context, new question/options, verified asset bindings, provider access, explicit
cost approval and safe output isolation. Full staged
execution additionally needs suitable indexes/embeddings, model weights and model
services. These prerequisites are not supplied by a repository clone alone.

Earlier feasibility work identified a three-JPEG prepared Direct continuation.
That would reuse a historical inspection decision and issue only the next
request; it is **not full fresh live execution**. A fresh model can ask for other
timestamps, so a tiny historical subset cannot silently replace the complete
available evidence. No continuation adapter remains executable in this demo.

Benchmark media is not redistributed. Do not copy images, weights, credentials
or unlicensed assets into this repository. See [attribution](ATTRIBUTION.md).

## Optional Whisper audio

Visual-only (the default):

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/visual --audio-mode none
```

Optional AV extension:

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/av --audio-mode whisper
```

Both commands first produce a plan. Review its model/profile identities, costs
and SHA; rerun the same command with `--execute --approve-plan <PLAN_SHA256>`
only when authorizing execution. Whisper transcription happens after approval,
before caption API calls. Planning probes for an audio stream but performs no ASR.

- `none`: `audio_profile=THESIS_FINAL_VISUAL_ONLY`; empty ASR and the unchanged
  visual-only Variant-C Organizer. No placeholder audio artifacts are created.
- `whisper`: `audio_profile=PORTABLE_AV_EXTENSION`; mono 16 kHz WAV, Whisper-small
  with English transcription, segment timestamps, overlap alignment and the
  separately pinned AV-aware v2.2 Organizer prompt. This is not the thesis-final
  R3 condition. API captions remain a substitute for Qwen2.5-VL-7B-Instruct.

Optional dependencies are ffmpeg/ffprobe, `openai-whisper`, PyTorch, NumPy and
`soundfile` (with libsndfile). A GPU is not required. CUDA uses FP16; CPU does not.
Word timestamps and conditioning on previous text are disabled, matching the
preserved historical transcription settings. Model weights must be available
locally or may be downloaded during approved execution. No model is loaded by
the plan-only command. Whisper mode rejects video without an audio stream;
choose `none` explicitly if appropriate. Speech recognition errors remain possible.

The AV workspace additionally contains `audio_16khz_mono.wav`, `audio_asr.json`
and `audio_cost.json`. The READY manifest covers all three hashes and records
the audio profile, model/language, alignment rule, segment count, waveform hash
and Organizer prompt identity. Native `exact_source_captions` and
`exact_source_asr` retain their separate origins. ASR overlap is strict:
`start < medium.end` and `end > medium.start`; a boundary-touching segment does
not overlap, while a spanning segment may belong to multiple Mediums.

ASK uses the same command and preserved Direct controller for either workspace.
It consumes transcript-bearing map text, **not raw audio**, and has no audio
inspection tool. No historical question or answer enters the new session.
Transcripts add Organizer/Direct tokens: the plan includes a heuristic transcript
allowance, not a measured token count or billing guarantee. Existing approval and
request limits still apply. Private speech is sent as text to the API; use only
authorized media and review privacy before approving execution.

Implementation changes invalidate implementation-bound old plans/workspaces;
choose a fresh workdir rather than editing a READY manifest or its hashes.

## Open-ended Direct

MCQ Direct remains the thesis-preserved A–E protocol and the default for
`--question-json`. Open-ended Direct is a **PORTABLE_OPEN_ENDED_EXTENSION** using
the same prepared workspace and Direct inspection loop, not a thesis result.
It requires explicit `--answer-mode open`; a text question never switches modes
implicitly. Do not supply `--question-json` or answer options in open mode.

```bash
python3 demo/ask.py --workdir demo_runs/example --answer-mode open \
  --question "What is happening in the water bath?"
```

This is plan-only. The plan records a newly generated question ID, the question,
`answer_mode=open`, `thesis_preserved_protocol=false`, original and derived
prompt hashes, and the extension implementation hash. To execute later, reuse
that question ID and the exact text:

```bash
python3 demo/ask.py --workdir demo_runs/example --answer-mode open \
  --question "What is happening in the water bath?" \
  --question-id <PLAN_QUESTION_ID> --execute --approve-plan <PLAN_SHA>
```

Environment-only credentials and separate cost approval still apply. A changed
question, ID, prompt, implementation or workspace changes the approved plan.
No previous question, answer or inspection state is loaded. The public Wikimedia
example runner still exposes only its two MCQs.

The derived prompt removes MCQ selection wording while preserving map navigation,
evidence grounding, uncertainty, timestamp limits, and the three-images-per-turn /
16-unique-images limits. The separate `final_answer` tool requires a non-empty
`answer` string and a brief `reason`; malformed or empty answers are rejected.
Only explicit recorded rationale is logged, never reconstructed hidden reasoning.
Inspection is optional: the model may answer from the map if sufficient.

Outputs include the question, free-text answer, recorded rationale, inspected
timestamps/frame hashes, usage, cost, latency and extension identity. They remain
outside the READY workspace. ASK consumes ASR text, not waveform audio.

Diagnostic limitation: compact ASK results retain an invalid-action event and
correction count, but not the rejected provider payload or detailed parser-error
category. An eventual final answer alone cannot establish the exact recovery
cause or whether correction changed its meaning. A completed run also does not
certify factual correctness; map/caption errors may persist in the answer.

Robustness to structured-output errors. In the open-ended smoke test, the selected model produced one malformed action that did not satisfy the expected tool/schema format. The Direct runtime detected the invalid action and used its built-in correction/retry mechanism to continue execution successfully. This demonstrates recovery from occasional structured-output failures without assuming that every model or configuration will exhibit the same behavior. The smoke test was not specifically configured to maximize first-pass structured-output reliability.

Reader compatibility: this release accepts the exact audited predecessor
`ask.py` SHA256 `fed80c6194f35d54c99c03bf9c189504b9979f1a105be3a49f0e8fc71ea4bf48`
in preparation manifests, as well as the current reader. All other implementation
hashes and every workspace artifact still require exact matches. No manifest is
rewritten; unknown reader versions fail closed. New ASK plans bind the current
implementation and need new approval, including for MCQ.

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
- Replay commands have no asset/key/output flags. Preparation and ASK accept no
  command-line API key. CLI errors deliberately
  do not echo supplied values, which could accidentally contain a secret.
- Run offline tests with
  `python3 -B -m unittest discover -s demo -p 'test_*.py'`.
