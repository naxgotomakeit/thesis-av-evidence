# Corrected Blinded Evidence-Support Audit Report

## Scope

This is a frozen, blinded evidence-support review of **176/176** opaque audit records. Each decision uses only the evidence actually present in the supplied package for that record. No gold association, correctness linkage, group/arm identity inference, target/control inference, or accuracy computation was performed. No external API was used.

## Input integrity validation

- Source archive SHA-256: `4b90933f994cb5463a7b3973573bc602f52a6793e397f3074347176e09a6f602`
- `MANIFEST.json` SHA-256: `f382298fca25319d2faa8734285e4a7eb4e27d564d39894de1a3ef24a32020ad`
- Declared `MANIFEST.sha256`: exact match
- Manifest-listed files verified: **1,132/1,132**
- Missing manifest-listed files: **0**
- SHA-256 mismatches: **0**
- JPEG evidence files: **839**, all successfully opened
- Records: **176**, all unique and present
- Map-bearing records: **116**, maps read as supplied
- Image-bearing records: **90**, actual supplied frames visually reviewed
- Package declares `blinded=true` and `prohibited_material_included=false`

## Review rule

Labels were assigned only from record-local evidence: `supported`, `partially_supported`, `unsupported`, `contradicted`, or `unreviewable`. Missing images, missing maps, sparse evidence, or an absent option image are evidence limitations and were **not** treated as technical `unreviewable` failures. No missing video facts were filled with common sense. The task was to assess support for the selected option, not to re-answer the underlying question.

## Special evidence condition

Several navigation/spatial questions encode the selected option only as an opaque path such as `navigation_images/.../B.png` or `spatial_layout_images/.../B.png`, while the corresponding option image is not included as record evidence. Those records can still be technically reviewed, but the specific selected visual option cannot be verified; they are therefore handled as evidence-insufficient (`unsupported`), not `unreviewable`.

## Completion and freeze

- Expected records: **176**
- Completed records: **176**
- Duplicate audit IDs: **0**
- Missing audit IDs: **0**
- Results are ordered according to the source manifest and frozen in `corrected_blinded_reviews.jsonl`.
- No gold-linked or accuracy statistics are included in this package.

Freeze timestamp (UTC): `2026-09-11T19:59:27Z`
