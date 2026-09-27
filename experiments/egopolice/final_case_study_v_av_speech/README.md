# EgoPolice v2.2 V vs AV-Speech — thesis Layer-1 archive

## Scope and authoritative identity

This staging archives the school-server experimental source evidence used for thesis §5.3 and as the source-data basis for Figures 5.2–5.3.

- Thesis-used final map pair: `v_av_speech_v2_2`.
- Visual-only condition: `V`, exactly 6 coarse regions.
- Speech-enhanced condition: `AV_SPEECH`, exactly 10 coarse regions.
- Frozen case-study questions: `q_weapon_visible`, `q_visible_injury`, and `q_handcuff_before_medical`.
- Primary route closure: exactly 2 conditions × 3 questions = 6 primary route records.

`v2.1` is superseded. `v2.3` is `DIAGNOSTIC_NOT_ADOPTED` and is not included as a thesis result.

## Two provenance layers

### Layer 1 — covered by this archive

```text
school formal experiment
  -> frozen v2.2 V/AV-Speech maps
  -> six primary Direct routes
  -> inspected timestamps and evidence identities
  -> frozen source-data handoff
```

This layer includes the prompts, schemas, configs, launch/freeze records, manifests, checksums, route outputs, evidence identities, and post-hoc no-gold qualitative audit needed to verify that chain.

### Layer 2 — not claimed by this archive

```text
frozen source data
  -> local Mac plotting/processing
  -> final thesis Figures 5.2–5.3
```

The school server did not generate the final thesis figures. This archive does not claim to contain the Mac plotting code, local processing closure, or final figure exports.

## Primary failure and diagnostic recovery

The primary route `q_handcuff_before_medical__V` remains a formal protocol failure: `terminal_status=invalid_action:invalid_reason`, `completed=false`, and `final_answer=null`. Its 12-frame navigation and evidence record is preserved unchanged.

`artifacts/diagnostic_format_only_recovery/` is a separate `DIAGNOSTIC FORMAT-ONLY RECOVERY`. It used the frozen history, added zero inspection calls and zero frames, and does not replace or relabel the primary route outcome.

## Portability and frozen bytes

The two `parsed_map.json` files and all six primary-route records are byte-identical copies of their formal source artifacts. Their historical absolute paths are part of the frozen record and are not silently edited.

New archive metadata uses portable source IDs. Manifest/config/handoff copies that only carried machine-local path strings were rewritten from the school prefixes to `source://school-project` or `source://school-home`; `SOURCE_MANIFEST.tsv` records both original and staged SHA-256 values and marks these files `PORTABLE_PATH_REWRITE`.

The private absolute-prefix mapping is stored in `DO_NOT_COMMIT/SOURCE_ROOT_MAP.tsv` and must not be uploaded.

## Frame policy

Frame JPEG redistribution/licensing has not been verified. This archive therefore includes timestamps, original path identities, frame SHA-256 values, captions/audit descriptions, and provenance only. It excludes candidate JPEGs, contact sheets, the complete 1-FPS frame pool, and the source video.

## Verification

See `VERIFICATION.md`, `SOURCE_MANIFEST.tsv`, `CHECKSUMS.sha256`, and `SENSITIVE_SCAN.md`. No model, API, map/route generation, or figure rendering was run while constructing this archive.
