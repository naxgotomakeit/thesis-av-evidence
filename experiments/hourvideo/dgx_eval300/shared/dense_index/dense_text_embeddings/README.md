# Eval300 Dense-semantic text embeddings

This is the independent versioned embedding asset for the frozen Beam-B protocol.

- `cases/<video>/`: Coarse/Medium float32 NPY arrays and node metadata.
- `completed_batches/`: atomic resumable API batch artifacts.
- `api_call_log.jsonl`: successful and failed API attempts with usage.
- `SOURCE_LINEAGE.json`: protocol and source-file lineage.
- `VALIDATION_REPORT.json`: machine-readable validation.
- `FINAL_EMBEDDING_BUILD_REPORT.md`: human-readable summary.
- `build_embeddings.py`: exact builder.
- `build_state.json`: terminal build state.
- `MANIFEST.sha256`: complete content manifest excluding itself and Python bytecode caches.

