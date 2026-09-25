# Targeted rerun gate

The H-15 acceptance run passed the fallback-specific gate: the forced full-video Inspector call decoded and sent 64 images, recorded 64 unique strictly increasing timestamps, began at 0 seconds, covered through 98.4375% of the video duration, and did not exceed the decoder-reported duration.

The same run failed the frozen-protocol gate. All 16 hierarchical retrieval calls returned `ValueError: Either model_file or model_proto must be specified.` and recorded `hierarchy_used=false`. The original Eval300 attempt recorded successful hierarchy use. Expanding the remaining 12 runs under this state would not isolate the fallback correction.

No remaining targeted rerun was started. The acceptance artifact is retained as infrastructure evidence and is not selected into Corrected Eval300.
