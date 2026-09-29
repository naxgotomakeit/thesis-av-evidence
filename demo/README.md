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

This reserved replay-launcher mode remains unavailable. Fresh portable execution
now has separate `prepare.py` and `ask.py` entrypoints, described below. The earlier
prepared-continuation adapter remains removed. Obsolete `--mode replay` and
`--mode live-direct` are rejected.

A small prepared-request continuation may be technically feasible, but it is not
the definition of future `live`: that mode means full fresh execution. See the
[usage guide](../docs/USAGE.md) for prerequisites and distinctions.

## Portable API-assisted fresh execution

Public [Wikimedia Live example](example_live/README.md), with separate approval for PREPARE and ASK:

```bash
python3 demo/run_live_example.py --question audio
python3 demo/run_live_example.py --question visual
```

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/example --caption-backend api
python3 demo/ask.py --workdir demo_runs/example --question-json question.json
```

Each command defaults to **plan-only**, stopping before paid requests. Preparation
does extract local frames. Read the [portable execution guide](../docs/USAGE.md#portable-api-assisted-execution)
for approval flags, question schema, output locations and costs. ASK requires a
successfully prepared workspace; it cannot use a plan-only workspace.

This is `PORTABLE_API`, not exact thesis preprocessing: Anthropic Haiku captions
substitute for historical Qwen2.5-VL-7B-Instruct. Fine=15 s, Medium=45 s,
chronological representative frames, the pinned Variant-C Organizer contract,
native R3 conversion and the preserved Direct online protocol are retained.
No SigLIP index is built. The default visual-only profile uses the Python standard library; preparation additionally
requires `ffmpeg` and `ffprobe`, authorized media and disk space, but no local GPU.
Paid execution requires environment-only `ANTHROPIC_API_KEY` and explicit approval.

Prepare once, then ask many independent A–E questions. The public question format
uses `"answer_options": {"A":"...","B":"...","C":"...","D":"...","E":"..."}`;
a thin adapter translates this object to the preserved loader's ordered option rows.
READY artifacts remain immutable. ASK plans/results go outside the workspace,
under ignored `demo_runs/<name>-outputs/qa_runs/<plan-sha>/`.

### Optional speech during preparation

```bash
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/visual --audio-mode none
python3 demo/prepare.py --video /path/to/owned-video.mp4 --workdir demo_runs/av --audio-mode whisper
```

`none` remains the default: `THESIS_FINAL_VISUAL_ONLY` describes the thesis-final
R3 **audio policy**, not exact preprocessing reproduction. The API-caption
substitution warning still applies. No audio files are generated in this mode;
the original visual-only Variant-C request continues to receive empty ASR.

`whisper` is a `PORTABLE_AV_EXTENSION`, not the thesis-final condition. It uses
mono 16 kHz extraction, local Whisper-small (English), and strict temporal-overlap
alignment. A segment spanning Mediums is attached to each overlapping Medium.
The separate preserved EgoPolice v2.2 AV-aware Organizer prompt distinguishes
speech from visual confirmation and preserves disagreement. Dataset-specific
EgoPolice inputs and retry policy are not reused.

Install optional `openai-whisper`, PyTorch, NumPy and `soundfile`/libsndfile;
CPU is supported and CUDA is used when available. Plan-only mode does not import
Whisper, download weights or transcribe. Approved execution may download missing
weights. A video without an audio stream fails explicitly in Whisper mode.
Use a new workspace when switching profiles.

Whisper adds `audio_16khz_mono.wav`, `audio_asr.json` and `audio_cost.json` to READY
integrity coverage. ASK consumes **ASR text from the map**; it does not inspect
raw audio or listen to waveform clips. Both modes retain explicit plan approval
before execution. See the [usage guide](../docs/USAGE.md#optional-whisper-audio).

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
