# Formal Organizer Variant C v1 report

## Outcome

- Status: **12/12 structurally valid; 0 failures**.
- Total mapping: **838 Medium → 186 Coarse**.
- Overall reduction: **77.80%**; total singleton phases: **26/186 (13.98%)**.
- Four controlled-v2 results were fully revalidated and copied byte-identically; they were not called again.
- Eight remaining videos each received exactly one new Organizer call; no retry or replacement call occurred.
- Prompt SHA-256: `c726fe9ac9dc3b37cd9993004880415ecb5c6c80948c2090163c3d4159061b76`.
- Schema SHA-256: `20b247005d3714d1807a806f1d4632585b3d9d3bd6f0f7e54399bc33abd82723`.
- Model: `claude-haiku-4-5-20251001`, temperature `0`, max_tokens `64000`.
- Visual-only, every `overlapping_asr=[]`, no question/option/answer/gold data, and no Fine caption generation.

## Per-video results

| Video UID | Source | Medium→Coarse | Reduction | Singleton | Max group | Tokens in/out | API latency | Cost | Stop |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `115774b6-534d-444f-b7aa-d1b834eb0ee7` | reused controlled v2 | 98→22 | 77.55% | 0/22 (0.00%) | 7 | 7160/1837 | 17.788s | $0.016345 | `end_turn` |
| `41a86310-2cc1-48f9-b5b5-6b495a95fbac` | reused controlled v2 | 62→4 | 93.55% | 0/4 (0.00%) | 26 | 5086/542 | 7.141s | $0.007796 | `end_turn` |
| `4572b198-2c1c-4920-bcf0-95fcebe12261` | new single call | 67→9 | 86.57% | 0/9 (0.00%) | 12 | 4677/1189 | 14.087s | $0.010622 | `end_turn` |
| `6fd90f8d-7a4d-425d-a812-3268db0b0342` | new single call | 80→24 | 70.00% | 0/24 (0.00%) | 6 | 6268/1759 | 18.417s | $0.015063 | `end_turn` |
| `70f2a750-f403-41b8-aabb-480eb3ab4ed4` | new single call | 38→9 | 76.32% | 0/9 (0.00%) | 7 | 3207/854 | 9.610s | $0.007477 | `end_turn` |
| `71fbc5bf-7e2a-415d-86bc-3a948742e904` | reused controlled v2 | 90→11 | 87.78% | 0/11 (0.00%) | 17 | 6170/1036 | 10.732s | $0.011350 | `end_turn` |
| `7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b` | reused controlled v2 | 126→24 | 80.95% | 6/24 (25.00%) | 71 | 9505/1111 | 11.478s | $0.015060 | `end_turn` |
| `7e512589-aa97-41e8-83d3-af2e83e4fd06` | new single call | 78→20 | 74.36% | 4/20 (20.00%) | 7 | 6896/1315 | 15.274s | $0.013471 | `end_turn` |
| `819c8af7-851f-434f-ab32-318285bc54b1` | new single call | 40→11 | 72.50% | 1/11 (9.09%) | 5 | 3421/872 | 9.704s | $0.007781 | `end_turn` |
| `a6d45e95-8dc0-4932-83bf-ec53e265a16a` | new single call | 40→12 | 70.00% | 1/12 (8.33%) | 6 | 2822/678 | 6.353s | $0.006212 | `end_turn` |
| `d3a0899e-2093-454c-9f65-30087883193a` | new single call | 83→20 | 75.90% | 3/20 (15.00%) | 20 | 5717/1315 | 14.092s | $0.012292 | `end_turn` |
| `db3f7933-dfa0-4678-9d4f-393b628ded45` | new single call | 36→20 | 44.44% | 11/20 (55.00%) | 4 | 2688/900 | 8.516s | $0.007188 | `end_turn` |

## Cost and timing

- Reused controlled-v2 historical cost: **$0.050551**; historical API latency: 47.139s.
- Eight new calls incremental cost: **$0.080106**; new API latency: 96.053s.
- Twelve-map total Organizer cost represented by this formal set: **$0.130657**.
- Tokens across all twelve: 63617 input / 13408 output.

Costs use the frozen recorded rates of $1/M input tokens and $5/M output tokens and are estimates rather than a provider invoice.

## Validation

All twelve outputs have strictly increasing, unique boundaries; correct final boundaries; exact once-only Medium coverage; continuous complete time coverage; identical prompt/schema/config hashes; and `end_turn` stop reasons. Each formal case stores sanitized input, complete raw provider response, usage, parsed output, parsed map, validation, result, and SHA-256 records. The four reused cases additionally store `reuse_verification.json`.

## Statistical anomalies and accepted cases

- `db3f7933…` has the highest singleton ratio: 11/20 (55%) and only 44.44% reduction. Its timeline contains many short changes among fridge, food, grill, drawers, laundry, coffee, washing, television, and outdoor bag handling. This is flagged as a fragmentation/statistical warning, not a structure failure; no automatic merging was applied.
- `7ddbf8a2…` has the largest group: 71 Mediums. Controlled semantic review found that M056–M125 contain identical sustained phone-use captions, with only the final M126 becoming vague, so this large group is an accepted known warning rather than an invalid boundary result.
- `41a86310…` has a 26-Medium laptop-disassembly group. This is accepted together with the documented short-event summary limitation.
- `d3a0899e…` has a 20-Medium group. Its captions describe a continuous wood-painting sequence followed by brush/can handling, so the size alone is not treated as anomalous semantics.

No video failed structural validation. These flags are reported for later retrieval analysis and were not used to alter the frozen maps.

## Accepted limitations

- `navigation_summary` can omit brief events.
- Some brief events may appear only in `uncertainty_notes`; `41a86310…` chair movement and mouse use are the controlled example.
- Current Coarse lexical routing primarily relies on `navigation_summary`, so H-6 and H-15 may lose recall at the Coarse gate.
- H-30 is reserved as the high-recall diagnostic upper bound that removes the narrow front-layer hard gate.
- No parent–child Coarse routing is implemented or planned for this frozen Organizer version.

These are recorded experimental limitations and are not treated as blockers for adopting Variant C.

## Scope stop

This run produced only the formal twelve-video Organizer maps and validation artifacts. It did not package, migrate, start Planner/Inspector, run VideoSEAL smoke, or run Eval300.
