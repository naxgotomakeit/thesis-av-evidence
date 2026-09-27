# Verification

## Required artifact identities

| Artifact | Expected SHA-256 | Result |
|---|---|---|
| Submitted dissertation | `29b3bf720579458784778d1fdd008e97a348bf81a9e5ad7c87909900cc0c64b7` | PASS; verified at source before staging. |
| Figure 5.2 maps.pdf | `f077bc22d22064d9ae5df81b4bb8ba50669aa360084ab9cf890fef8b98d31026` | PASS. |
| Figure 5.3 evidence.pdf | `92af1b14c78ac71814860148ebeda31311017d23d88141e6a3f5bd93f10cba1b` | PASS. |
| build_case_compact.py | `2c432da96b2fd6f3f9a1f8293796c3b841bbb8797e49c7cbf7ff2eed5ef09525` | PASS. |
| build_preview.py | `7b13e84c3b6cc86f58d335df33593a2325064f98f33aa4f55e54266d38ac1c7b` | PASS. |

The requested preview-script SHA without `df` after `...58d335` is 62 hexadecimal characters and is not a valid SHA-256.  The 64-character SHA above is the actual byte hash of the source and staged file; it is the one used in this package.

## Binding checks

- Figure 5.2 V and AV-Speech `parsed_map.json` copies in the local review bundle were byte-identical to their Layer-1 counterparts.
- Figure 5.3(a) and (b) four route results were byte-identical to Layer-1 primary routes.
- The staged `q_medical_assistance` route/recovery files match hashes held in Layer-1's upstream-review inventory.
- The 12 displayed-frame SHA values are recorded in `DISPLAYED_FRAME_MANIFEST.json`; no JPEG is staged.
- `artifacts/compiler_evidence/main_log_figure_inputs.txt` records final compiler use of both PDFs, and the final build PDF equals the submitted-PDF SHA above.

## Staging checks

- missing required files: 0
- checksum mismatches: 0
- no plotting command, model/API call, or Git operation was run.
