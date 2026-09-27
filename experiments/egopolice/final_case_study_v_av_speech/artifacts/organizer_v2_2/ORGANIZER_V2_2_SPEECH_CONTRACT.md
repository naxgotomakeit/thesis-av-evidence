# Organizer v2.2 symmetric V vs V+Speech contract

This contract derives only from frozen Organizer v2.1. It preserves every v2.1 structural and semantic-grouping instruction and inserts one speech-aware paragraph. The provider output schema and canonical local strict validator are unchanged.

## Conditions

- V: exact frozen visual representation, exact v2.2 Organizer, and empty `overlapping_asr` for all 28 Mediums.
- AV-Speech: exact same visual representation and Organizer, with the 107 frozen downstream ASR nodes attached by deterministic positive temporal overlap.
- The only permitted input difference is `timeline[*].overlapping_asr` content.
- Non-speech/sound-event evidence is excluded.

## Frozen identities

- v2.1 prompt SHA-256: `c5ab6fa09ebb84bdf85ce31782676b2103289abe11bb8cb7a12935a9c5158994`
- v2.2 prompt SHA-256: `532c58e7df9c2e86e12669f5c9dffc1fc4c607cfb516ce12668f9369be271a1d`
- Provider schema canonical JSON SHA-256: `20b247005d3714d1807a806f1d4632585b3d9d3bd6f0f7e54399bc33abd82723`; byte-identical to v2.1.
- Canonical validator runner SHA-256: `dd2da49836430efb67a3572cb31cef506e66ad3559579ee5d6b55178807584f0`; validator function identity inherited unchanged.
- Frozen speech SHA-256: `a88ee06b9ef5c9b855e01a05540e286c43bc5c9474b28ed313a0eb3bc67ccf2e`; 107 nodes.

## Speech alignment

`asr.start_sec < medium.end_sec AND asr.end_sec > medium.start_sec`

This is positive overlap under half-open interval semantics. A boundary-crossing node may be repeated in both Mediums. Source order and every source field are preserved; there is no transcript rewrite, summarization, question filtering, confidence filtering, empty-transcript filtering, or deduplication.

Speech counts by Medium index: `[0, 0, 1, 1, 4, 4, 3, 3, 0, 0, 1, 0, 1, 10, 6, 3, 7, 6, 4, 3, 11, 10, 5, 8, 9, 7, 8, 7]`.

## Unchanged contract

Model `claude-haiku-4-5-20251001`, temperature 0, max tokens 64000, timeout 900 s. Each condition allows at most two identical attempts, accepts the first strict-valid response, and applies no repair or manual modification. Output validation remains the canonical strict validator.
