# EgoPolice Figures 5.2–5.3: Layer-2 provenance import

This is a GitHub-ready, Mac-side provenance package for the exact PDF figure files included by the submitted dissertation.  It is intentionally separate from `experiments/egopolice/final_case_study_v_av_speech/` (Layer 1), which this package neither changes nor duplicates wholesale.

## Status

| Thesis artifact | Status | Identity |
|---|---|---|
| Figure 5.2 | COMPLETE | Formal V (6 regions) and AV-Speech (10 regions) maps. |
| Figure 5.3(a) | COMPLETE | `q_weapon_visible`, V and AV-Speech. |
| Figure 5.3(b) | COMPLETE | `q_visible_injury`, V and AV-Speech. |
| Figure 5.3(c) | COMPLETE | Thesis-used `q_medical_assistance`, V and AV-Speech.  The AV-Speech displayed answer is a separately identified format-only recovery. |

`q_handcuff_before_medical` is a distinct school-side case identity.  It is **not** the source of Figure 5.3(c), is not renamed here, and is not inferred from shared frames.

## Scope and reading order

1. `FIGURE_BINDING.json` is the authoritative identity and cross-layer binding record.
2. `DISPLAYED_FRAME_MANIFEST.json` records every frame timestamp, portable source path, and SHA-256 used in Figure 5.3.  JPEGs are deliberately absent pending redistribution clearance.
3. `scripts/` and `source/previews/` contain the real Mac plotting scripts and the preview validation metadata.
4. `artifacts/figures/` contains the exact two PDF files included by LaTeX.
5. `source_snapshots/q_medical_assistance/` preserves the small, thesis-used legacy route/recovery records not materialized as primary routes in the Layer-1 archival package.
6. `artifacts/compiler_evidence/main_log_figure_inputs.txt` records final LaTeX compiler use of both PDFs.

The legacy input archive `case_study_code.zip` and all JPEGs are not copied.  Their provenance roles and SHA bindings are recorded in the manifests; the ZIP is an opaque duplicate whose only required plotting constants are independently recorded and bound to the Layer-1 maps.

Run `shasum -a 256 -c CHECKSUMS.sha256` from this directory to verify staged files.
