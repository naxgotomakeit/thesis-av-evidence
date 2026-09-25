# Missing or not fully frozen items

1. The original standalone entry script that produced the shared 1-FPS frame
   cache and the original standalone SigLIP embedding-generation entry script
   were not found in the current school project trees. The package retains the
   frame/download audits, downstream hierarchy preparation entry, exact hashes,
   and offline-cost report, but it does not claim those two missing launch
   scripts are reproducible from this import alone.
2. A standalone Direct Eval300 canonical-summary/scoring builder was not found.
   The formal runner/candidate/launch code and complete canonical output are
   retained, but the final canonical report construction cannot currently be
   attributed to a preserved script.
3. The Direct, GenS v3 and Full Staged formal manifests do not freeze the school
   orchestrator hostname. They establish that execution used the school-side
   workspace and external Anthropic API, but the exact launch node remains
   unconfirmed. Capacity-aware model-serving hosts are frozen and retained.
4. The school GenS package contains the Myriad V2 selector output and provenance
   manifests, not the Myriad-side selector-generation source tree. Its original
   selector SHA-256 and its use by the formal school run are frozen.
5. R1 detector telemetry does not freeze a hostname. R3 caption generation
   records `canada-l.cs.ucl.ac.uk`, and the R3 Organizer records
   `cream.cs.ucl.ac.uk`.

