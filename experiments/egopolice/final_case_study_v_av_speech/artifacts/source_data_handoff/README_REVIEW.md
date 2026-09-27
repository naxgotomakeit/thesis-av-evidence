# 226 case-study human review bundle

This compact bundle is for local, read-only inspection of the adopted 226 V/AV-Speech case-study results. It contains no model weights, embeddings, audio waveform, full 1-FPS pool, historical packets, v2.3 diagnostic directory, provider image payloads, or caches.

## Recommended review order

### A. Start with the six contact sheets

Open `final_qualitative_audit/contact_sheets/`. Each sheet shows every physical frame inspected for that question, labels it V / AV_SPEECH / COMMON, and gives the post-hoc usefulness score.

### B. Inspect all 25 actual Direct-inspected frames

Open `inspected_frames_all/`, then use `inspected_frames_all.csv` to trace each timestamp and SHA to every question/condition that inspected it. These 25 JPEGs are exact-byte copies from the frozen formal 1-FPS pool.

### C. Read the final qualitative audit

Read `final_qualitative_audit/FINAL_CASE_STUDY_AUDIT.md`, followed by `per_frame_visual_audit.csv`, `answer_support_audit.json`, and `route_consistency_audit.json`.

### D. Review the six V/AV answers and routes

Read `direct_sixq/DIRECT_OPEN_ENDED_SIXQ_SUMMARY.md`, `direct_sixq/route_results.json`, and `direct_sixq/route_inspected_frame_manifest.json`. The primary outcome remains **8/12 protocol-complete**. The four reason-length failures remain primary failures; `format_only_recovery/` is a separate **4/4 diagnostic**, with zero new inspection calls and zero new frames.

### E. Finish with the adopted maps and limitations

Compare `maps_v2_2/V/parsed_map.json` with `maps_v2_2/AV_SPEECH/parsed_map.json`, then read `v2_2_interpretation_audit/V2_2_FINAL_INTERPRETATION.md` and its claim reclassification.

## Interpretation locks

- **v2.2 is the adopted matched V/AV map pair.** v2.3 is diagnostic and is not included.
- The 19 images under `final_qualitative_audit/candidate_frames/` are all frames actually inspected by frozen Direct routes. No manually selected frame was substituted.
- The human timeline was used only after route completion for qualitative comparison; it did not guide model navigation or frame selection.
- Non-speech/sound-event evidence was not used in the current AV condition.
- AV-Speech differs upstream through the speech-enhanced semantic map. Raw ASR was **not** directly supplied to the Direct answerer.
- No gold/correctness data was used in the audit.
- Files are copied for review only; source experiment artifacts were not modified.

Run `sha256sum -c MANIFEST.sha256` from this directory to validate the review bundle.
