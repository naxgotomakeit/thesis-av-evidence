# R3_2 frozen exact-interleaved AV Organizer canary v1

This experiment preserves the current canonical repaired Qwen caption text and
canonical ASR, but restores the historical Organizer's input topology: 30 visual
nodes and 107 audio nodes each appear once in one timestamp-ordered interleaved
stream. Stage 1 globally groups that stream. Stage 2 performs local fusion using
the validated global phase label and limited neighbouring context.

Historical phase outputs are absent from generation and loaded only after the
candidate map is complete. Mechanical map fields are deterministic. Storyline,
hard pruning, Planner calls, Retrieval, and Sufficiency are disabled.
