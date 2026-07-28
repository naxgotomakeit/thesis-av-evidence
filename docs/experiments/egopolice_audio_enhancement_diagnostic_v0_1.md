# EgoPolice Audio Enhancement Diagnostic V0.1

This is an isolated manual-review experiment based on the frozen `egopolice-v0-naive-av-storyline-freeze` tag. It changes audio evidence extraction only; visual hierarchy, captions, storyline generation, and Claude answer generation remain untouched.

The speech branch preserves original VAD/ASR and runs a second English `transcribe` pass. Suspect segments receive at most one context-expanded fallback and retain both records. The SED branch uses the AudioSet-pretrained AST baseline `MIT/ast-finetuned-audioset-10-10-0.4593` with fixed 10-second windows and 5-second hops. It emits `gunshot_like_candidate`, never `gunshot_confirmed`.

This run is manual-review only. It does not read annotation Ground Truth, compute precision/recall, call Claude, produce a storyline, or modify annotation files. Full WAVs and review clips stay on the external SSD; repository outputs contain lightweight JSON/CSV/HTML/report/provenance files.
