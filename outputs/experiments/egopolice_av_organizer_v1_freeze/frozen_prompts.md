# Frozen prompts — EgoPolice AV Organizer V1

The authoritative executable prompt constants remain in the canonical runners
listed below. Their UTF-8 SHA-256 values are frozen here.

## v0.4 visual caption

Source: `scripts/experiments/run_egopolice_caption_density_diagnostic_v0_4.py`

Version: event-first ego-aware, Qwen2-VL-2B, `max_new_tokens=44`.

```text
You are an efficient analyst of first-person body-camera footage.

Describe the most important visible event in concise, high-information language. Prioritise:
1. the main event and foreground interaction,
2. who is doing what,
3. relevant camera-wearer hands, arms, equipment or actions,
4. critical objects and brief scene context,
5. uncertainty.

Return one compact line using only relevant fields:

EVENT: ... | EGO: ... | CONTEXT: ... | UNCERTAIN: ...

Rules:
- EVENT must describe the main visible situation and actions, not general scenery.
- Mention EGO only when camera-wearer involvement is visible or reasonably attributable; otherwise omit the field.
- CONTEXT should contain only scene details or objects needed to understand the event.
- Put unclear ownership, object identity or action under UNCERTAIN.
- Do not copy instructions or field definitions.
- Do not infer intent, causality or events outside the frame.
- No introduction, JSON, Markdown or repeated information.
- Omit irrelevant or empty fields rather than writing “none”.
- Maximum 28 words.
```

## v0.7 Typed Minimal Evidence selector

Source: `scripts/experiments/run_egopolice_visual_full_asr_fusion_v0_7_typed_minimal_evidence_map.py`

- User prompt SHA-256: `da547e83016bfa1eb5fdb892de9b7c3cda65dfe8ec18cbab908524154119e7ed`
- System prompt SHA-256: `20a4793fee37ce08ba158924ecb462f7a2d26f406a4dc2d9b91db98bd0daa8ef`

The selector is restricted to typed visual/audio IDs, typed AV relations,
boundary-context IDs and uncertainty tags. It may not rewrite captions,
transcripts, summaries or free-text AV notes. Full wording remains frozen in
the canonical runner.

## P01–P08 staged Phase Presenter

Source: `scripts/experiments/run_egopolice_v0_7_minimal_map_staged_presenter_v0_1.py`

- Phase prompt SHA-256: `d542eec92287923d494d2373c391faf5455065aab0872c06176c8edd6f10cb34`
- System prompt SHA-256: `3c13c1d3082f95f65187591295e2f5c520a9ffa470dbcb31521395da758e6cee`

The frozen phase schema contains `visual_account`, `audio_account`,
`combined_account` and `uncertainty`. Audio claims remain attributed; roles
default to person/individual; knife, live-round and injury reports may not be
promoted to visual facts.

## P09 continuity recovery

Source: `scripts/experiments/run_egopolice_v0_7_minimal_map_staged_presenter_v0_2_p09_continuity_recovery.py`

- P09 prompt SHA-256: `f788ae9d3420212bb50bc2a075a2b6e2f5cc0a1322ab397bb5a588cdcc35d9fb`
- System prompt SHA-256: `3c13c1d3082f95f65187591295e2f5c520a9ffa470dbcb31521395da758e6cee`

P06–P08 validated combined accounts are continuity-only context. “Previously
restrained person” is allowed only as a cautious
`contextual_continuity_reference`; it is not a direct visual role or identity
tracking result.

## Global Executive Summary

Canonical successful source:
`scripts/experiments/run_egopolice_v0_7_minimal_map_staged_presenter_v0_2_p09_continuity_recovery.py`

- Prompt SHA-256: `1d5b50e59c25bcf565cc80660b0a745309903eaea460529797a5671abc449c8a`
- Output budget: 800
- Maximum summary length: 180 words
- Maximum critical uncertainty items: 5

Only the nine validated compact phase accounts are supplied. Original visual
atoms and ASR nodes are not reintroduced at this stage.
