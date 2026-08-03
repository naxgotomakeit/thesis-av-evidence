# EgoPolice R1_AV dual-channel retrieval smoke v1

This isolated smoke gives the shared Planner the R1_AV chronological map containing five structural visual nodes and 107 independent exact-ASR nodes. It ranks all 30 Mediums using the frozen structured-fallback plus SigLIP path and independently ranks all 107 ASR records using exact-transcript lexical overlap.

The map is navigation only. Audio supports audible statements or mentions, not visual confirmation or physical-event occurrence. Coarse hard pruning, Coarse prior, Fine reranking, Sufficiency, Final Gemini and QA are disabled.
