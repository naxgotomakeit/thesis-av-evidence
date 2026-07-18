# Current Canonical Baseline

The source of truth is `config/canonical_pipeline.json`. Versions are explicit;
they are never inferred from filename order or modification time.

| Task | Canonical version | Canonical source lineage |
|---|---|---|
| Task 5A | v2 | `src/question_planner/task5a.py` |
| Task 5B | v1.1 | reusable `build_case`; v1.1 packaging integrated in `canonical_pipeline/retrieval.py` |
| Task 5C | v1.2 | final sufficiency/acoustic rules from `task5c_v1_1.py` and `task5c_v1_2.py` |
| Task 6 | v1.2 | final role, relation, frame-order and accounting helpers from Task 6 modules |
| Task 7A | v1 | `src/final_qa/task7a_preflight.py` |
| Task 7B | v3 | `task7b_gemini.py` plus v0.2 validator semantics; historical disk label `v0_3` maps to logical v3 |

## Runtime semantics

Historical Task 5C v1/v1.1/v1.2 and Task 6 v1/v1.1/v1.2 files document how
the research behavior evolved. They are **not** sequential runtime stages.

The canonical runner performs one Task 5C sufficiency decision and executes
fallback at most once. It then builds one final Task 6 v1.2 packet directly.
Intermediate JSON files may be saved for audit but are not handoff requirements.

For generalized execution, the case allowlist is supplied by an explicit
manifest. The online Task 5B boundary encodes each new raw question with the
validated Task 4 CLIP, Sentence-T5, and CLAP text encoders as required, then
scores the existing question-independent indexes by normalized cosine
similarity. Historical `outputs/retrieval/<case_id>` score files are regression
references only and are not runtime dependencies. Query model loading,
encoding, and similarity search are measured as online work.

Task 5C v1.2 materializes already-selected acoustic intervals into local WAV
evidence when their role is `direct_evidence`, `temporal_anchor`, or `resolver`.
This is evidence packaging, not a new retrieval or selection rule; canonical
correctness does not depend on a historical Task 5C clip already existing.

## Technical contracts (no research-policy change)

The canonical producer contract distinguishes candidate entities, retained
evidence entities, media assets, relations, evidence groups, and packet
records. Every Task 6 retained entity is either serialized into exactly one
model-facing Task 7A group or explicitly marked non-model-facing with a reason.
Model-facing relations are unique directed edges whose endpoints both exist in
the supplied evidence set. Frame and WAV assets never inflate evidence counts.

Task 7A embeds the exact Pydantic schema consumed by logical Task 7B v3; the
schema no longer has a separate illustrative copy. Future runs carry a
secret-free fingerprint of the Git revision, canonical versions/configuration,
prompt and schema hashes, model settings, manifest, and every used offline
index bundle. Every newly encountered video/index bundle is validated against
the persistent encoder contract without reloading the encoder.

Regression mode injects the frozen Task 5A plan and, where required, the frozen
Task 5C fallback evidence. It makes zero external calls. Live mode has explicit
planner, fallback and Gemini client boundaries and remains opt-in. The first
integrated live verification completed for `00006_3` and `00061_5` with no
category-4 research-behavior mismatch; this is an integration milestone rather
than a benchmark claim.

## GitHub snapshot scope

Commit source, tests, documentation, canonical configuration and sanitized
configuration templates. Local EgoSound manifests and annotations are excluded
with the dataset because redistribution/licensing has not been established. Do
not commit `.env`, API credentials, generated outputs, source media, extracted
WAV/JPG files, embeddings, model weights, provider caches, or machine-local path
configuration.
