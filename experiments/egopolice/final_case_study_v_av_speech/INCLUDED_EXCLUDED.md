# Included and excluded material

## Included

- The adopted v2.2 V and AV-Speech maps and their matched-pair validation.
- Organizer v2.2 prompt, schema/contracts, config, speech-alignment contract, runner, pre-API freeze, and checksums.
- The frozen question set and the three thesis question identities.
- Exactly six selected primary route directories, each with route input/result, turn history, validation, route checksum, and minimal provider-attempt metadata.
- Direct open-ended prompt/config/schemas, runner, launch preflight, execution freeze, final manifest, checksums, and local comparison code/report.
- The separately labelled format-only diagnostic recovery for `q_handcuff_before_medical__V`.
- Timestamp/frame-SHA manifests and the no-gold qualitative source-data handoff used to choose the three main-paper cases.
- Portable source manifest, staged checksums, closure verification, sensitive scan, and compact tree.

## Included only as upstream provenance indexes

- The original Direct final manifest/checksum covers the complete 12-route six-question run. Only the six thesis-case primary routes are materialized here; the other six routes and their large request/response payloads are intentionally not copied.
- The upstream review-bundle manifest/inventory is retained as a historical package index. `CHECKSUMS.sha256` is authoritative for the contents of this staging.

## Excluded

- Original EgoPolice video.
- Full 1-FPS frame pool, candidate JPEGs, inspected-frame JPEG copies, contact sheets, and final thesis figures.
- SigLIP arrays, embeddings, model weights, caches, virtual environments, and bytecode.
- Provider request bodies containing encoded images and other large repeated payloads.
- `.env`, credentials, API keys, bearer tokens, passwords, and private configuration.
- Unselected route payloads and unrelated case-study questions except compact aggregate comparison records.
- v2.1 outputs, which are superseded.
- v2.3 outputs, which are diagnostic and not adopted.
- Earlier semantic-coarse smoke, naive storyline, audio diagnostic, Organizer V1, pilot, and failure artifacts already represented by historical Git lineage.
- Mac-side plotting/processing artifacts and final Figures 5.2–5.3; these belong to Layer 2 and were not present in the school-server closure.

## Local-only

`DO_NOT_COMMIT/` contains only the school absolute-prefix map and local handling note. It contains no model/API output, image, video, or credential.
