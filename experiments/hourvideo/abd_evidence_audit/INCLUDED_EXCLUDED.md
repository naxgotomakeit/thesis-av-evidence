# Included and Excluded Material

## Included

| Area | Included material | Archival role |
|---|---|---|
| ABD formal | Analysis results/report/manifests; raw journals, task artifacts/status, budget and controls; frozen prompt/config; runtime and run/score/freeze scripts | Formal experimental record |
| ABD initial audit 900 | Freeze, identity/batch/package manifests, accepted 192 review batches, scored records, reports, cross-statistics and reproducibility records | Parent audit for ABD correction/final lineage |
| ABD correction 176 | Build report, package manifest and SHA sidecars; 176 corrected reviews, correction freeze, reconstructed mapping, validation reports | Correction stage (116 targets + 60 controls) |
| ABD independent review 31 | Preparation and review freezes, 31 blinded outputs, identity mapping, comparison report/CSV/manifest, reviewer instructions and package manifests | Independent resolution of 31 control disagreements |
| ABD diagnostics/final | Disagreement diagnostic, strict-support statistics, and all files in the final authoritative output directory | Derived analysis and final 900 freeze |
| Historical R1/R3/GenS 900 | Freeze, identity and batch/package manifests, frozen/scored records, accepted reviews, reports/statistics, and audit scripts | Separate historical comparison audit |

## Excluded from the upload tree

| Material | Reason | Retained proof |
|---|---|---|
| Original benchmark videos | Size/licensing; not required for the result archive | Result/freeze manifests and identifiers |
| Extracted benchmark frames, evidence images, contact sheets | Size/licensing | Package/preparation manifests retain member names, sizes, and SHA-256 values |
| Historical `review_maps/`, `review_packages/`, `contact_sheets/` | Large derived evidence payload | Historical package freeze/manifests and review records |
| ABD initial `review_packages/`, `package_fragments/` | Large duplicated map/image/input payload | Initial package freeze/manifest and accepted reviews |
| Invalid/quarantined first/second-pass review outputs | Explicitly non-authoritative | Initial invalid/valid ledger and reproducibility manifests only |
| Correction input tarball and unpacked item tree | Duplicate large payload; unpacked tree includes 839 images and 116 maps | Build report, tarball SHA sidecar, package manifest, manifest SHA, README |
| Independent-review item images/maps/input payload | Evidence payload and licensing/size concern | Preparation freeze and package manifest contain the full member hash inventory |
| Analysis result tarballs | Duplicate of included unpacked authoritative artifacts | Archive manifests and recorded archive SHA-256 |
| Model weights, caches, `__pycache__`, virtualenvs | Not part of the experiment record | None needed |
| `.env`, credentials, API secrets | Security | Excluded; see sensitive scan report |
| Pilot, smoke, fake, and invalid/quarantine outputs | Non-authoritative | Not included |
| Unrelated benchmark/config data | Out of scope and potentially redistributability-sensitive | Not included |

## Path-only redaction

Uploadable text copies containing the source user's absolute workspace prefix use portable placeholders. This is the only permitted content transformation and is itemized per file in `SOURCE_MANIFEST.csv`. Prompts, labels, predictions, rationales, token values, and statistics were not edited. The local absolute source-root mapping exists only in `DO_NOT_COMMIT/SOURCE_LOCATIONS_LOCAL_ONLY.md`.
