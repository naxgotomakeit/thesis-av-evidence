# R3_2 frozen-method staged AV Organizer canary v1

This canary recreates the successful historical Organizer method without using
the historical phase result. Stage 1 sees the complete canonical caption + ASR
timeline and proposes 5–12 major incident phases, phase labels, boundary reasons,
and a small salient evidence set. Stage 2 receives each validated phase record,
its local evidence, and limited neighbouring context, then produces a grounded
phase account.

The historical nine-phase count, boundaries, labels, summaries, and evidence IDs
are absent from all model inputs. They are loaded only after the candidate map is
complete for descriptive comparison.

Code, not the model, generates Coarse IDs, exact Medium-derived times, complete
Medium/Fine coverage, and deterministic ASR attachment. Stage 3/Storyline, hard
pruning, and coarse ranking priors are disabled.
