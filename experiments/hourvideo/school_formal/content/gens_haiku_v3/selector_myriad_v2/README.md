# GenS V2 Eval300 selector manifest-only package

This package contains GenS selector output, not final A-E predictions. It contains no images, video, model weights, embeddings, raw model responses, generated token IDs, or frame cache payloads.

Downstream usage:

1. Read `selected_frames_downstream.jsonl` in file order.
2. On the target machine, resolve each `relative_frame_path` against an existing frame-cache root.
3. Recompute every referenced file SHA-256 and compare it with `frame_sha256` before use.
4. If a same-named target frame has a different SHA-256, it must not be substituted; retrieve the original verified frame again.
5. Sort `selected_frames` by `chronological_order` before sending images downstream. The API receives only the question, options A-E, and those chronologically ordered images.

`selection_provenance.jsonl` is audit-only and must never be sent to the API. Its winning-option telemetry and per-option score vectors were deliberately removed for downstream isolation. Only aggregate CLIP rank/score and GenS selection telemetry remain.

No gold file may be read at any point. For comparisons with Uniform or Ours, use exactly the same API model, prompt, decoding, parser, and retry protocol.

The package does not copy frames. The manifest path `<video_id>/frame_NNNNN.jpg` is a portable logical path; the target deployment is responsible for mapping it to its local verified cache layout.
