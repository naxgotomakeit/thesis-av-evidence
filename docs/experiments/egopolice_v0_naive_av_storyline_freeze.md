# EgoPolice V0 — Naive Audio-Visual Storyline Baseline

**Status:** frozen baseline, V0  
**Freeze date:** 2026-07-28  
**Repository commit at freeze:** `7c5c73999369c9251efce4fe8ad90daea7a6c319`  

## Frozen pipeline

The V0 path is:

`visual hierarchy → weighted keyframe selection → Medium / Coarse captions → VAD + timestamped ASR → audio–visual temporal alignment → Claude Haiku storyline / structured incident answer`

V0 processes all visual events and reasons mainly over caption and transcript proxies. Audio is speech/VAD/ASR only. It has no task-aware filtering, non-speech audio event detection, Coarse→Medium→Fine retrieval, local original-video multi-frame verification, or salience-aware resource allocation.

The answer stage uses Anthropic Structured Outputs with the frozen compact V0 contract: incident summary, timestamped fused timeline, officer/civilian actions, important speech, six compact critical-event strings, uncertainty, and conflicts. The current runner permits no automatic retry for the representative runs; the configured retry ceiling is 9000 tokens only for the explicit max-token policy.

## Frozen implementation and evidence

Visual configuration and runner:

- `config/experiments/pilot16_semantic_map_smoke_v0_1.json`
- `scripts/experiments/run_pilot16_semantic_map_smoke_v0_1.py`
- `src/experiments/medium_semantic_abstraction/qwen_local.py`

Audio configuration and runner:

- `config/experiments/pilot16_audio_diagnostic_v0_1.json`
- `scripts/experiments/run_pilot16_audio_diagnostic.py`

Answer configuration and runner:

- `config/experiments/egopolice_av_answer_v0_1.json` and per-video variants
- `scripts/experiments/run_egopolice_av_answer_v0_1.py`

The four representative runs are recorded in `outputs/experiments/egopolice_v0_naive_av_storyline_freeze/RUN_PROVENANCE.json`. Large frames, keyframes, features, audio and model caches remain under `/Volumes/512GSSD/EgoPolice/runtime/` and are intentionally excluded from Git.

## Freeze rule

These files, prompts, contracts, and parameters are evidence for comparison with V1. They must not be silently changed. Any protocol or algorithm change requires a new version and a new freeze record; V1 work must not be added to this freeze.

## Known limitations

See `outputs/experiments/egopolice_v0_naive_av_storyline_freeze/LIMITATIONS.md`.
