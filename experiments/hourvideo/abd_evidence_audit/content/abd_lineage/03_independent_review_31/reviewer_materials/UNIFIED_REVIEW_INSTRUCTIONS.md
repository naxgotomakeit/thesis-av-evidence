# Unified independent evidence-review instructions

Use `AUDIT_CRITERIA_FROZEN.md` as the governing standard. Review each anonymous record independently.
Judge only whether the model-selected option is supported by evidence that was actually supplied in the
record. Do not re-answer the question. Do not seek identity, method, gold, correctness, prior reviews, or
other video evidence. If an original answer reason is ever present, it is a claim to verify and is not evidence.

Apply these five labels:

- `supported`: clear supplied evidence distinguishes and establishes the selected answer.
- `partially_supported`: supplied evidence supports a key part of the selected answer, but another necessary
  condition is not established. Scene relevance alone is not partial support.
- `unsupported`: evidence is technically reviewable but insufficient to support the selected answer.
- `contradicted`: supplied evidence explicitly refutes the selected answer. Omission, non-observation, or the
  appearance of another object is not automatically a contradiction.
- `unreviewable`: technical inability to inspect the supplied package. A missing modality is not itself a
  technical failure.

For counts, duration, immediately-next events, ordering, and whole-video non-occurrence, require evidence
that covers the corresponding condition. Do not treat sample-frame count or summary length as event count or
duration. When an answer option is only an image-path string, do not treat the option image as visible and do
not retrieve it; decide whether the remaining supplied evidence establishes the selected option's identity.
Do not fill missing facts from common sense or the answer wording.

For every item, read the complete map when present and visually open every supplied image. Timestamps are the
exact model-visible labels and must remain paired with their following images. Record one JSON object with:

- `review_id`
- `option_support`
- `evidence_references`
- `concise_rationale`
- `missing_key_conditions`
- `review_flag`
- `review_flag_reason`
- `map_read` (boolean; true iff a supplied map was completely read)
- `images_opened` (integer; must equal the supplied image count)

Use references such as `map:C07(585-675s)` or `image:frame_003@90.000s`. Preserve unresolved ambiguity in
`review_flag`; do not revise a label merely to make it agree with an unknown earlier reviewer.

This supplements the historical frozen standard by making the five-label operational boundaries, missing-
modality handling, image-path-option handling, timestamp pairing, and viewing-attestation fields explicit.
The historical per-item prompt and decoding parameters were not preserved, so this is an independent review
under the recovered written standard, not an exact replay of historical execution.
