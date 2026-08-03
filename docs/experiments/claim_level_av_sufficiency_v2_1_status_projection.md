# Claim-level AV Sufficiency v2.1 status projection

This version treats each atomic `claims[].status` as the canonical classification.
The four redundant lists are deterministic, order-preserving projections for
`supported`, `uncertain`, `conflicted`, and `not_found`. `required_claims` and
every claim object remain unchanged. Raw responses are journaled before parsing.
No event-, question-, actor-, timestamp-, or video-specific rule is used.

The frozen v2 validator already treats atomic status as authoritative: it maps
each redundant list to an expected atomic status and rejects mismatches. V2.1
removes that redundant-generation failure mode without changing claim semantics.
