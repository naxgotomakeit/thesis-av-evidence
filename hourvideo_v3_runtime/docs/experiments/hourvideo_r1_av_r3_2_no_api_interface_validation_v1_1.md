# HourVideo R1_AV / R3_2 no-API interface validation v1.1

This isolated experiment validates the real HourVideo dev50 question/options and media interfaces without running any model or API. It projects all 1,182 questions into a safe online payload containing only question ID, video UID, question text, and five ordered options. Correct labels, reference timestamps, task metadata, canary strings, and scoring metadata remain offline.

R1_AV and R3_2 receive byte-identical question/options and deterministic option-centric requirements. Their runtime map bindings remain different by design, but rung names are not exposed in the shared model-facing question payload. The experiment validates the existing 50-video/141,550-frame cache, video/audio availability, and all image-option assets.

This is an interface dry run, not a five-video performance run. Planner, Retrieval, Sufficiency, review, Final Gemini, and answer generation remain unexecuted. Passing permits predeclaring and indexing five pilot videos. It does not claim that the 25 image-option questions are answerable by the current text-only final solver; those require a separate option-image review adapter.
