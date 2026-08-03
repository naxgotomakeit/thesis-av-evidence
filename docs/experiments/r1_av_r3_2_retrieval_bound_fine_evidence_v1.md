# R1_AV / R3_2 retrieval-bound Fine evidence v1

This no-API experiment corrects the Gemini review boundary without adding
cross-question memory. Both representations reuse the same frozen Fine SigLIP
reranker after their existing Medium rankings. Only explicitly selected Fine
IDs may be supplied to a later Gemini review. Expanding a selected Medium or a
claim time range into all available images is forbidden.
