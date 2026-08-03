# R3-v2 Coarse semantic Organizer canary v1

This isolated canary preserves the useful global caption-semantic mapping behavior of the historical frozen Organizer while narrowing the model responsibility to contiguous semantic grouping, a concise `navigation_summary`, and uncertainty notes.

The sole model input is the complete ordered set of 30 byte-identical canonical repaired Qwen captions. Detector/tracking summaries, ASR, historical summaries, questions, options, rung identity, and diagnostic targets are excluded. One global Haiku call sees the whole timeline; it is not an adjacent-boundary classifier.

Code validates inclusive Medium-index ranges and deterministically generates Coarse IDs, source IDs, timestamps, durations, Fine mappings, half-open temporal audio attachments, provenance, and adjacency. Storyline is disabled. The resulting map and `coarse_summary = navigation_summary` compatibility view are navigation aids for Planner query formulation only. They are not Sufficiency evidence and must never hard-filter Medium candidates.

Manual review of `review.html` is required before a semantic-map freeze. A grouping difference from the historical 11-Coarse map is not itself a failure.
