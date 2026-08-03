# R1_AV/R3_2 reviewed-visual cache integration canary v1

This isolated canary places the immutable reviewed-visual cache between the
question-scoped Gate and Gemini review. It reuses the prior one-image live cache
as a read-only seed, batches only missing image/scope pairs, updates only the
authorized requirements, and recomputes reliability status deterministically.

The exact-scope conflict resolver prefers reviewed visual evidence over captions
and detector observations only for the same image hash, Fine ID, timestamp, and
fact scope. It never globally suppresses other times or ASR.

No Sufficiency, final answer, or QA call is made.
