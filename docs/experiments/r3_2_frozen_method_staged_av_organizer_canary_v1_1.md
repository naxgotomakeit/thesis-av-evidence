# R3_2 frozen-method staged AV Organizer canary v1.1

This rerun corrects one contract mismatch in v1: v1 imported the later
minimal-evidence map's three-visual hard cap into the historical Stage 1 phase
grouping contract. The historical Stage 1 only requested a few salient IDs.

v1.1 retains the qualitative "few salient IDs" prompt and uses a broad
validation-only ceiling. It makes a fresh Stage 1 call and does not reuse or
repair the rejected v1 response. All other frozen-method, zero-repair,
no-Storyline, and no-hard-pruning policies remain unchanged.
