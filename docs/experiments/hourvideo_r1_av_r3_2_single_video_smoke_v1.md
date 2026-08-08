# HourVideo R1_AV / R3_2 single-video smoke v1

This isolated smoke runs one 26.58-minute HourVideo dev video and one frozen
text-option question through both candidate pipelines. It is an interface and
cost diagnostic, not a five-video or dataset result.

The shared question, options, 1-fps frame cache, 15-second Fine nodes,
45-second Medium nodes, SigLIP embeddings and timestamped Whisper ASR are
identical. Gold is excluded until both blind predictions have been serialized.

- R1_AV uses YOLOv8x-OIV7 + BoT-SORT structured fallback, a deterministic
  structural/audio navigation map, all-Medium retrieval, requirement
  Sufficiency, selective Gemini review and text-only Final Gemini.
- R3_2 uses local Qwen dense Medium captions, one global AV semantic Organizer,
  semantic-Coarse-first Sufficiency, local descent only when requested,
  selective Gemini review and the same text-only Final Gemini.

The run intentionally preserves weak results and contract violations. It does
not tune top-k or retry semantics after the official answer is loaded.
