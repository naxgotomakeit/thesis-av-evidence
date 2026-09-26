# Unresolved issues

1. No standalone identity/target-control mapping file was saved when the correction input package was built. Exact deterministic reconstruction succeeded, but this remains a provenance defect.
2. 7 correction records contain `navigation_images` or `spatial_layout_images` option paths. Frozen builder inspection indicates those option assets were not sent as image blocks: the model received the path strings in question/options text. This was an original request limitation, not merely a correction-export omission. The conclusion is high-confidence from frozen code and input rows, although a byte-for-byte serialized wire payload was not persisted.
3. File/hash validity and full reviewer completion do not by themselves establish semantic quality. Control agreement and rationale differences must be reviewed.
4. No correction labels have been merged into the 900-row frozen audit, and no final arm-level support rates were generated.
