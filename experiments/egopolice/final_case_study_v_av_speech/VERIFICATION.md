# Closure verification

## Final status

**PASS** — 255/255 structural and identity checks passed; 0 failures.

Verifier: `verify_archive.py`.

## Thesis identity

| Check | Result |
|---|---|
| V artifact identity | `v_av_speech_v2_2 / V` |
| V coarse-region count | PASS — 6 |
| AV artifact identity | `v_av_speech_v2_2 / AV_SPEECH` |
| AV-Speech coarse-region count | PASS — 10 |
| Frozen thesis questions present | PASS — 3/3 |
| Materialized primary route records | PASS — exactly 6 |
| Route identity closure | PASS — 2 conditions × 3 questions |

Frozen thesis questions:

- `q_weapon_visible`
- `q_visible_injury`
- `q_handcuff_before_medical`

## Byte identity

| Artifact | Source/staged SHA-256 | Result |
|---|---|---|
| V parsed map | `fc480efee797c379afbb31c5d1bdad0d14f44fe777fbce15b562ab55a22d9d65` | PASS |
| AV-Speech parsed map | `c8af9b585d661448c496c44128db05452c0334b412e6031b8d1022f2d23110f1` | PASS |
| `q_weapon_visible__V` route result | `b46ae16fea2a923747faaa547605bc876df385a885b4beb0683af7a73ac5c8c0` | PASS |
| `q_weapon_visible__AV_SPEECH` route result | `c346d0c859ad110604b796adae389145c942de15f7f59e5d6208ed77b29eb6a9` | PASS |
| `q_visible_injury__V` route result | `54deda4f25a344354a164736874b7b261a1899ba1967a3b397bc1d894b293526` | PASS |
| `q_visible_injury__AV_SPEECH` route result | `c106b88ed44ec981d4114da5ca69048d07561bb6faf788ace2a0c552d7b982c2` | PASS |
| `q_handcuff_before_medical__V` route result | `52011dec95b99b559f3d42853e4ef21b485aefff97355b215e86d728698e0778` | PASS |
| `q_handcuff_before_medical__AV_SPEECH` route result | `4b9ad434f4b419388c6013867d86e9ab32ad5b233d69988bc209e8c37198d3f8` | PASS |

`SOURCE_MANIFEST.tsv` contains 106 source artifacts: 79 byte-identical copies and 27 documented portable path rewrites. Every staged artifact is manifested, and every manifested staged SHA matches.

The 106 original source locators were re-resolved against the school workspace: source missing = 0 and original-source SHA mismatch = 0. The upstream review bundle's independent `MANIFEST.sha256` also verifies 172/172 entries with 0 failures.

## Route/evidence closure

For all six primary routes:

- route question text matches the frozen question text;
- route condition/question pairs match the expected Cartesian product;
- `resolved_timestamps_sec` matches the corresponding entry in `route_inspected_frame_manifest.json` in value and order;
- the frame SHA sequence matches the handoff manifest;
- the handoff's `source_route_result_sha256` matches the staged route result.

| Primary route | Resolved timestamps (seconds) | Result |
|---|---|---|
| `q_weapon_visible__V` | 547, 562, 577 | PASS |
| `q_weapon_visible__AV_SPEECH` | 547, 562, 577 | PASS |
| `q_visible_injury__V` | 547, 652, 727 | PASS |
| `q_visible_injury__AV_SPEECH` | 547, 637, 727 | PASS |
| `q_handcuff_before_medical__V` | 630, 720, 1035, 585, 675, 900, 600, 650, 1100, 615, 1150, 1200 | PASS |
| `q_handcuff_before_medical__AV_SPEECH` | 585, 720, 1035, 630, 675, 600 | PASS |

## Primary failure versus recovery

- Primary `q_handcuff_before_medical__V`: PASS identity check — `completed=false`, `terminal_status=invalid_action:invalid_reason`, `final_answer=null`.
- Recovery classification: PASS — `FORMAT_ONLY_RECOVERY_DIAGNOSTIC`.
- Primary completion unchanged: PASS.
- New inspection calls: PASS — 0.
- New frames: PASS — 0.
- Recovery validation: PASS.

The diagnostic answer does not replace the primary route outcome.

## Source-data handoff

Required handoff files are present: route/frame manifest, inspected-frame CSV, per-frame audit, per-question quality, case ranking, figure candidate plan, final case-study audit, review README, and upstream review-bundle manifest.

No image, video, embedding array, or model-weight extension exists under `artifacts/`.

## Closure totals

- Missing staged/source artifacts from `SOURCE_MANIFEST.tsv`: 0.
- Original-source SHA mismatches across 106 source artifacts: 0.
- Upstream review-bundle checksum failures across 172 entries: 0.
- Staged SHA mismatches: 0.
- Formal map/route source-vs-staged SHA mismatches: 0.
- Route/frame timestamp mismatches: 0.
- Route/frame SHA mismatches: 0.
- Identity conflations between primary and recovery: 0.
