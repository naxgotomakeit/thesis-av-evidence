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
