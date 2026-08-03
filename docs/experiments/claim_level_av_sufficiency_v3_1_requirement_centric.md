# Claim-level AV Sufficiency V3.1 requirement-centric contract

V3.1 changes only required-claim generation. Instead of freely generating
claims and then associating them with zero or more requirements, the model must
emit exactly one primary assessment claim for every predeclared requirement.

The status, confidence, support-mode, evidence, and time taxonomies are
unchanged from V3. Auxiliary claims use the same atomic semantic fields but do
not contain a requirement ID and never count toward coverage.

Coverage validation compares the generated and declared requirement-ID sets
exactly. Missing, duplicate, or unknown IDs fail without retry, remapping,
fuzzy matching, or semantic repair. Reversed claim event times also fail.

Projection derives status lists, criticality, reusability, evidence time
envelopes, answerability, and review candidates. It hashes every semantic
object before and after projection. The legacy adapter reuses each
model-generated assessment claim directly. A model-generated `not_found`
assessment therefore remains the same claim; no sentinel is synthesized.

The first live canary uses only the frozen Uniform R2@50 Weapon packet and makes
exactly one Haiku call. It stops after legacy compatibility validation and does
not run reliability, handoff, Gemini, final answers, other Uniform questions,
Selective, or Dense R3.
