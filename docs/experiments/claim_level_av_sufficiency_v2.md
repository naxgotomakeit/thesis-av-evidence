# Claim-level AV Sufficiency v2

This isolated EgoPolice 226 experiment replaces only the sufficiency stage with
text-only Haiku claim inference, claim-level sufficiency assessment, and
uncertainty-triggered raw-visual-review requests.

The formal model input is limited to per-question evidence already selected by
the frozen rich AV downstream: visual captions, exact ASR transcripts,
timestamps, typed temporal links, and uncertainty metadata. It contains no raw
images, prior sufficiency decisions, prior claim ledger, Gemini answers, ground
truth, or unrelated phases.

The model is `claude-haiku-4-5-20251001`, with the frozen sufficiency temperature
and output-token budget. Each of the six questions receives exactly one formal
call. There is no semantic retry or second model-based JSON repair.

Old sufficiency, claim-gate, final-answer, and selective-Organizer artifacts are
comparison-only controls loaded after all new formal outputs have been produced.
The candidate claim cache means “supported by the current audiovisual index,”
not ground-truth verified.

Run a no-API validation:

```powershell
py scripts/experiments/run_claim_level_av_sufficiency_v2.py
```

Run the six formal calls:

```powershell
py scripts/experiments/run_claim_level_av_sufficiency_v2.py --allow-api-calls
```
