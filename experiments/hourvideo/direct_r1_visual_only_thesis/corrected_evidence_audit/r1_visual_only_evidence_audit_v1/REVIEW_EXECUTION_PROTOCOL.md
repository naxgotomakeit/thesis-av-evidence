# R1 visual-only evidence audit execution protocol

## Frozen basis

The governing criteria are the byte-identical copy at
`../r1_visual_only_evidence_audit_preparation_v1/audit_inputs/AUDIT_CRITERIA_FROZEN.md`
(SHA-256 `7e9e25909a97ade16b86c896e5b086f5148b7d3f1bda8a8118db8c07f318f4e3`).
No support category or evidentiary boundary is added or removed.

## Complete execution instruction

For every route, inspect the complete review package, the complete map referenced by
`map_review_path`, every actually transmitted image in recorded order (using the
contact sheet for overview and source frame where detail is needed), its requested
and resolved timestamp, and tool feedback. Judge support for the selected option,
not the truth of the answer in the unseen video. Never infer absence from a missing
frame. Duration, frequency, order, exhaustive lists, and global absence require
commensurate observed coverage. Do not use filenames, answer wording, or the model's
reason alone as evidence. Do not add frames or consult the source video.

Emit exactly the old nine-field schema. For answered Direct routes,
`direct_evidence_source` must be one of `map`, `images`, `map_and_images`, or `none`,
as the old written criteria require. This is a clarification of the provenance
field only; it does not change `option_support` or any evidence threshold. The one
route without a final answer is recorded as `no_final_answer`, with no fabricated
support judgment or reason label.

During review, do not load identity mapping, gold, correctness, old labels for the
175 rerun routes, old answers, answer-change flags, or score summaries. Freeze and
validate all 175 records before identity or score linkage. The 125 reuse records are
not rejudged; their old labels are linked only after the new 175 freeze.

## Execution identity and limitations

- Reviewer: the current Codex assistant exposed by this session, identified only as
  Codex based on GPT-5.
- Exact checkpoint/build: not exposed and therefore unknown.
- Temperature/sampling seed: not exposed and therefore unknown.
- Review tools: local read-only file inspection and local image viewing; no external
  model/API and no QA execution.
- Batch format: old package format; at most 10 routes and 50 images. This prepared
  population yields 54 batches.
- Blinding: neutral IDs and separation from private identity mapping; gold,
  correctness, old rerun labels, and rerun/reuse identity hidden. This R1-only audit
  does not claim cross-method blinding.

The historical reviewer runtime is not recoverable. This run claims protocol and
material-format continuity, not byte-identical reviewer execution.

## Historical enum clarification

The old R1 frozen labels contain 106 `direct_evidence_source=not_applicable` records
(`map_and_images=121`, `images=60`, `none=13`, `map=0`). The old final report already
states that `not_applicable` was inconsistent for Direct. The field is descriptive
provenance and was not used to compute the six-category support totals. Restricting
new answered Direct records to the four values explicitly named in the old criteria
therefore does not alter support labels or their aggregation. Old records and parser
assets remain untouched.
