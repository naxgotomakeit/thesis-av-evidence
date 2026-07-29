# EgoPolice AV Organizer V1 Freeze Report

- Freeze status: `single_video_validated`
- Video: `540772226`
- Date: `2026-07-29`
- Organizer phases: 9/9 hard-valid
- Duration coverage: 100% (1235.307 seconds)
- Critical evidence: 21/21
- Evidence atoms: 18 visual, 33 transcript
- Typed AV links contain no prose note
- Low-information visual selections: 0
- Obvious ASR-corruption selections: 0
- Organizer cost: 9 calls, 22,338 tokens, 28.063 seconds
- Presenter timeline: P01–P09 complete
- P09 continuity audit: passed
- P09 role source: `contextual_continuity_reference`
- P09 direct visual role: false
- Identity tracking claimed: false
- EMS, ambulance and transport coordination: retained
- Chest/right-groin assignment uncertainty: retained
- AST: excluded
- Audio retrieval: excluded
- Planner: excluded
- Multi-video validation: pending

## Canonical read-only entry

`scripts/experiments/run_egopolice_av_organizer_v1.py`

The default frozen operation verifies existing paths, schemas, artifact
hashes, phase coverage and epistemic constraints without network or API calls.

## Integrity

`artifact_hashes.json` records 24 source artifacts using SHA-256. The freeze
directory contains no image, audio, video, API log or copied review asset.

This freeze does not modify the earlier frozen V0 tag
`egopolice-v0-naive-av-storyline-freeze`.
