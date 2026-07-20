# Planner taxonomy validation v0.1

## Research question

Can question-only evidence requirements be described stably and usefully along three independent axes—temporal scope, evidence nature, and modality requirement—before any canonical routing change is considered?

This is an isolated teacher-annotation experiment. It neither validates answer correctness nor changes canonical Ours-v0.1.

## Experimental taxonomy

### Temporal scope

- `local`: one contiguous temporal neighborhood is likely sufficient, including a short local transition.
- `multi_event`: two or more distinguishable events or states and their relation are likely required.
- `global`: broad coverage, distributed stages, overall activity, or video-level progression is likely required.
- `unclear`: question text alone does not support a reliable decision.

The criterion is the minimum temporal evidence scope likely required, not keywords or question length.

### Evidence nature

- `static`: state, object, attribute, scene, configuration, or identity evidence is primary.
- `dynamic`: motion, action, change, order, interaction, or process evidence is primary.
- `uncertain`: question text cannot support a primary choice, or both are necessary without a defensible primary.

An optional different `nature_secondary` and `nature_ambiguity` flag preserve genuine overlap.

### Modality requirement

- `visual`: visual evidence is semantically indicated as sufficient or primary.
- `audio`: speech, sound, noise, or other auditory evidence is explicitly required.
- `audio_visual`: auditory evidence likely must be linked to visible context.
- `indeterminate_from_question`: wording alone cannot safely select a modality.

Dataset identity is deliberately withheld from annotation; modality availability is separate audit metadata.

## Sampling

The frozen seed is `20260719`. Representative samples contain 100 EgoSchema and 100 EgoSound questions selected by deterministic SHA-256 rank without balancing. Their distributions estimate natural prevalence within the available local sources.

The disjoint diversity stress samples contain up to 50 questions per dataset selected by deterministic question-text-only cue buckets. They stress taxonomy boundaries and must not be mixed into prevalence estimates.

After the first annotation pass, 20 questions per dataset are frozen for independent reannotation, combining low-confidence/ambiguous, high-confidence, and label-coverage strata.

## Annotation and stability design

The existing configured Anthropic planner provider is reused with temperature zero and strict structured JSON. The first pass uses one compact frozen rubric. The stability pass uses a semantically equivalent but independently phrased compact rubric with the same configured model. Questions are batched, but each receives an independent record.

No chain-of-thought or answer generation is requested. Each axis has a short descriptive reason and confidence. `taxonomy_failure` records missing distinctions that cannot be represented even by conservative labels.

## Information boundary

The provider sees only an opaque batch-local item ID and question text. The following are excluded from all annotation prompts:

- gold answers and correctness;
- multiple-choice options;
- gold or dataset timestamps;
- dataset identity;
- video frames or paths;
- audio;
- retrieved evidence or prior model output.

No video model, retrieval, final QA, or canonical pipeline stage executes.

## Analyses

Representative and stress distributions remain separate. The experiment reports per-axis class/confidence/ambiguity, independent-pass agreement and confusion pairs, cross-axis tuples and normalized mutual information, taxonomy failures, and provisional routing actionability. Provisional resource-policy strings are analysis only and are not a router.

Axis conclusions use frozen thresholds and are independently classified as `SUPPORTED`, `NEEDS_REVISION`, or `REJECT_OR_MERGE`. Human review remains necessary because teacher consistency is not ground truth.

## Limitations

- Same-model rephrased-rubric agreement measures rubric stability, not independent human validity.
- Question text may intrinsically underdetermine modality and evidence scope.
- The stress set is lexically enriched and is not a prevalence sample.
- Provisional actionability does not show that a routing policy improves QA or efficiency.
- Optional cached KTS/CoMET overlap is structural diagnostic only; it is not temporal ground truth.

## Non-changes

Canonical Ours-v0.1, Task5A/B/C, Task6, KTS/CoMET routing, evidence selection, and final QA remain unchanged.

## Completed run

- Provider/model: Anthropic / `claude-haiku-4-5-20251001`.
- First pass: 300 questions in 15 successful structured-output calls.
- Stability pass: 40 questions in 2 successful calls.
- Provider request attempts: 20 total. Three initial HTTP 400 schema-subset rejections produced no annotation response; they are preserved separately from 17 successful model calls.
- Successful-call tokens: 33,527 prompt and 48,224 completion.
- Successful-call API wall-clock: 292.715 seconds.
- Estimated cost: unavailable because the repository has no maintained Anthropic price table.

All three axes are classified `NEEDS_REVISION` under the frozen assessment thresholds. Scope agreement is 0.675; primary-nature agreement is 0.825 but ambiguity agreement is 0.550; modality agreement is 0.775; full tuple agreement is 0.400. These are same-model, independently phrased-rubric results and require human validation.
