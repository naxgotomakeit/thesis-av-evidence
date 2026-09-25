# Eval300 Dense Text Embedding Build Report

Status: **COMPLETE / VALID**

## Scope and method

- Sole protocol source: `dense_semantic_hierarchical_beam_b_frozen_20260827T194716Z`.
- Scope: exactly the 12 Eval300 Variant C videos present in the frozen materialized index.
- Model: `text-embedding-3-large`.
- Requested and verified dimensions: 3072.
- Coarse text field: unchanged `navigation_summary`.
- Medium text field: unchanged `qwen_caption`.
- No caption, summary, Organizer, hierarchy or Fine SigLIP asset was modified or regenerated.

## Counts

| Video UID | Coarse | Medium | Success | Failure |
|---|---:|---:|---:|---:|
| 115774b6-534d-444f-b7aa-d1b834eb0ee7 | 22 | 98 | 120 | 0 |
| 41a86310-2cc1-48f9-b5b5-6b495a95fbac | 4 | 62 | 66 | 0 |
| 4572b198-2c1c-4920-bcf0-95fcebe12261 | 9 | 67 | 76 | 0 |
| 6fd90f8d-7a4d-425d-a812-3268db0b0342 | 24 | 80 | 104 | 0 |
| 70f2a750-f403-41b8-aabb-480eb3ab4ed4 | 9 | 38 | 47 | 0 |
| 71fbc5bf-7e2a-415d-86bc-3a948742e904 | 11 | 90 | 101 | 0 |
| 7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b | 24 | 126 | 150 | 0 |
| 7e512589-aa97-41e8-83d3-af2e83e4fd06 | 20 | 78 | 98 | 0 |
| 819c8af7-851f-434f-ab32-318285bc54b1 | 11 | 40 | 51 | 0 |
| a6d45e95-8dc0-4932-83bf-ec53e265a16a | 12 | 40 | 52 | 0 |
| d3a0899e-2093-454c-9f65-30087883193a | 20 | 83 | 103 | 0 |
| db3f7933-dfa0-4678-9d4f-393b628ded45 | 20 | 36 | 56 | 0 |
| **Total** | **186** | **838** | **1024** | **0** |

## API usage

- Successful API calls: 8.
- Failed API attempts/retries: 0.
- Input tokens: 31,868.
- Total tokens reported by the embedding API: 31,868.
- Batch size: 128 nodes; all eight batches were saved atomically.

## Validation

- Expected nodes present: 1024/1024.
- Missing or duplicate stable node IDs: 0.
- Parent mappings: every Medium maps to exactly one Coarse.
- Array shape: every row is 3072-dimensional float32.
- Non-finite values: 0.
- Text SHA mismatch: 0.
- Model/dimension metadata mismatch: 0.

Each video has separate Coarse and Medium NPY arrays plus JSON node metadata binding row index, stable node ID, parent relationship, source text SHA, model and dimensions. `completed_batches/` and `build_state.json` provide safe resume evidence; an invalid existing batch is never silently overwritten.

No GPU, Planner, Inspector, Retriever, smoke or formal experiment was started.

