# Reviewed visual evidence cache v1 — live Gemini canary

This isolated canary deterministically selects one previously retrieved Fine
frame, verifies an empty `weapon_visibility` cache entry, makes at most one
Gemini 3.5 Flash REST image call, validates and stores the observation, then
repeats the same lookup locally. The second path must be a cache hit with zero
additional model calls.

It does not integrate with R1_AV/R3_2, run Sufficiency/final answering, or
perform evidence-conflict resolution.
