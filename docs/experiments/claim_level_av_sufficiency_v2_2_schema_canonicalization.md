# Claim-level Sufficiency schema canonicalization v2.2

V2.2 composes atomic-status projection with two audited reference/control rules:
`required_claims` text maps only by a unique normalized exact claim-text match,
otherwise validation fails; an explicit visual-review request is authoritative
for the validation/downstream review-control view. Semantic claim fields remain
unchanged and every mutation is audited. No fuzzy matching or event rule exists.
