# EgoPolice AV Organizer V1 — Freeze Definition

## Status and scope

**EgoPolice AV Organizer V1** is the first complete successful
**single-video validated audio-visual Organizer baseline** for video
`540772226`. It is not evidence of multi-video generalisation and is not the
final thesis system.

Freeze date: 2026-07-29.

## 1. Frozen architecture

```text
Audio Indexer + Video Indexer
        ↓
Global AV Phase Grouping
        ↓
Typed Minimal Evidence Organizer
        ↓
Staged Phase Presenter
        ↓
Global Executive Summary
```

The reusable **Organizer** is:

> Global AV Phase Grouping + Typed Minimal Evidence Map

The Presenter is not part of the reusable Organizer index. It is an
on-demand global-question / incident-summary layer.

## 2. Frozen components

### A. Visual Index input

- The current 30 frozen Visual Medium intervals.
- Existing weighted single-keyframe selection.
- `Qwen/Qwen2-VL-2B-Instruct`, snapshot
  `895c3a49bc3fa70a340399125c650a463535e71c`.
- v0.4 event-first ego-aware caption prompt.
- Existing preprocessing and decoding configuration.
- `max_new_tokens=44`.
- v0.5 interaction-first captions are excluded as a retained failure case.
- No segmentation, keyframe-selection or caption regeneration is part of
  this freeze operation.

Visual limitations present in the source are preserved, including occasional
v0.4 token truncation.

### B. Audio Index input

- Silero VAD.
- Enhanced timestamped Whisper ASR using Whisper `small`.
- `language=en`, `task=transcribe`; translation disabled.
- Original transcript text, timestamps and fallback metadata are retained.
- Speaker, channel and source type remain `unclear`; no content-based role
  reassignment is frozen.
- AST/non-speech events are excluded.
- Keyword filtering and audio retrieval are excluded.

Transcript text represents words present in ASR and is not a confirmed fact.

### C. Organizer Stage 1

The successful global AV phase grouping from v0.2 is reused:

- nine chronological phases;
- independent visual and audio timelines;
- deterministic timestamp-based node-to-phase assignment;
- timestamp-overlap alignment;
- fixed semantic boundaries in
  `frozen_phase_boundaries.json`.

Stage 1 is not rerun or reinterpreted.

### D. Organizer Stage 2

Frozen condition: **v0.7 Typed Minimal Evidence Map**.

Canonical artifact:

`outputs/experiments/egopolice_visual_full_asr_fusion_v0_7_typed_minimal_evidence_map/minimal_evidence_map.json`

Claude model: `claude-haiku-4-5-20251001`, temperature `0.0`.
Selector output budget: 550 tokens; format-only repair budget: 550.

Haiku only selects visual/audio evidence IDs and typed AV links. It does not
rewrite captions or transcripts, create prose AV notes, phase summaries, or
storylines. Canonical node registry determines modality. Deterministic code
controls timestamp assignment, PRIMARY/boundary-context membership,
modality/context rerouting, stable deduplication, invalid-link removal,
exact-copy atom assembly and provenance.

Frozen schema and sanitization constraints are recorded in
`frozen_schema.json`.

Successful metrics:

- hard-valid phases: 9/9;
- phase and duration coverage: 100%;
- accepted duration: 1235.307/1235.307 seconds;
- critical evidence audit: 21/21;
- visual atoms: 18;
- transcript atoms: 33;
- selected low-information visual atoms: 0;
- selected obvious ASR corruption: 0;
- calls: 9;
- input/output/total tokens: 20,239 / 2,099 / 22,338;
- latency: 28.063 seconds;
- Presenter calls in Organizer run: 0;
- canonicalizer used: no;
- AST used: no.

### E. Staged Presenter

P01–P08 source:

`outputs/experiments/egopolice_v0_7_minimal_map_staged_presenter_v0_1/`

P09 and Global Summary source:

`outputs/experiments/egopolice_v0_7_minimal_map_staged_presenter_v0_2_p09_continuity_recovery/`

Claude model: `claude-haiku-4-5-20251001`, temperature `0.0`.
Phase budget: 650 tokens. Successful global-summary budget: 800 tokens.

Each phase independently produces:

- `visual_account`;
- `audio_account`;
- `combined_account`;
- `uncertainty`.

Audio-only information must remain attributed. Unsupported roles use
`person`/`individual`. Knife, live-round and injury reports may not become
visual confirmation. Chest and right-groin reports may not be assigned to one
person without support.

P09 may cautiously use “previously restrained person” or state consistency
with preceding restraint/medical phases. This is a
`contextual_continuity_reference`, not a direct visual role or identity
tracking. The successful audit records:

- `role_source=contextual_continuity_reference`;
- `direct_visual_role=false`;
- `identity_tracking_claimed=false`.

After all nine phase-local accounts validate, only their compact combined
accounts and uncertainties are sent to the Global Executive Summary.

## 3. Design rationale

- v0.2 staged fusion retained content well, but was expensive and its verbose
  prose schema encouraged over-interpretation.
- v0.3 showed that a compact representation could materially reduce
  repetition and cost.
- v0.4 and v0.5 showed that complex prose schemas and an LLM canonicalizer
  introduced unstable semantic/structural validation and extra cost.
- v0.6 established the Minimal Evidence Map direction: models select evidence
  while code preserves the original atoms.
- v0.7 added explicit typed input and deterministic sanitization, reaching
  9/9 hard-valid phases and 100% coverage without free-text AV notes.
- The staged Presenter recovers v0.2's phase-local reasoning advantage while
  operating only on selected v0.7 evidence.
- P09 continuity recovery separates direct current-phase visual evidence from
  cautious cross-phase continuity wording.
- LLMs are responsible for bounded semantic selection and interpretation.
- Deterministic code is responsible for time, type, membership, provenance,
  coverage gates and final assembly.

## 4. Known limitations

- Successful validation currently covers only video `540772226`.
- Cross-video stability is not established.
- Scaling of phase count with video duration is not established.
- Visual captions remain vulnerable to wrong keyframes, occlusion, motion
  blur, single-frame insufficiency and Qwen2-VL-2B capability limits.
- v0.4 captions can still be token-truncated.
- ASR speaker/channel/source classification is not reliable.
- ASR mistranscription remains present.
- Transcript statements are not confirmed facts.
- Non-speech AST evidence is not in the formal Organizer.
- No person identity tracking exists.
- P09 “previously restrained person” is contextual continuity, not identity
  confirmation.
- There is no audio retrieval.
- There is no Planner.
- There is no query-conditioned retrieval.
- The validated output is a global incident storyline; it does not mean the
  QA system is complete.

## 5. Next experiments (not implemented by this freeze)

- 3–5 new-video zero-tuning stability test;
- pilot16 evaluation;
- Visual-only versus Visual+ASR;
- verbose v0.2 versus typed minimal V1;
- ablation without typed sanitization;
- AST non-speech addition;
- full ASR versus retrieved ASR;
- keyframe quality gate / conditional second frame;
- subsequent retrieval and Planner experiments.

## Reproduction and verification

Read-only verification:

```powershell
python scripts/experiments/run_egopolice_av_organizer_v1.py `
  --video-id 540772226 `
  --reuse-existing-artifacts `
  --verify-only
```

This path performs zero API calls. Fresh execution requires an explicit
`--allow-api-calls` flag and a new non-canonical experiment destination; the
freeze facade refuses to overwrite canonical roots.

The artifact inventory and source hashes are in:

`outputs/experiments/egopolice_av_organizer_v1_freeze/`

