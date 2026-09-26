# ABD evidence-audit correction inputs (blinded)

This portable package contains only opaque audit inputs for evidence-support review.
Each `items/E####/input.json` contains the question/options, predicted option, original
reason, actual model-visible map (if any), relative paths to copied image bytes, resolved
timestamps and the exact model-visible timestamp text. `map.raw.json` preserves the
frozen map bytes. No identity mapping, arm name, gold, correctness, previous audit
label/rationale, or statistical result is included. ABD was single-turn, so tool_feedback
is empty. Item order is deterministically shuffled.
