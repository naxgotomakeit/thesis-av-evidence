# Claim-level AV Sufficiency V3 minimal contract

## Scope

This isolated experiment replaces the monolithic V2 output contract with three
layers. It does not modify V2, call a model, rerun retrieval, or make claims
about EgoPolice 226 quality.

## Layer 1: semantic payload

The future LLM output contains only `question_id` and atomic claims. A claim
contains `claim_id`, predeclared `requirement_ids`, `claim_text`, `claim_type`,
`status`, `support_mode`, `evidence_ids`, `confidence`,
`remaining_uncertainty`, `rejected_inferences`, and optional
`claim_event_time`.

The status enum remains `supported`, `uncertain`, `conflicted`, and
`not_found`. Requirement IDs must already exist in the input requirement
specification. There is no text-to-ID recovery.

## Layer 2: deterministic projection

Code derives the four status lists, required-claim references,
answer-criticality, reusability, evidence modality, evidence time envelope,
answerability, top-level uncertainty, and a review-candidate view. The candidate
view does not decide that review will occur; Gemini Stage A retains that role.

`evidence_time_envelope` is provenance localization. It is not an assertion of
event start or completion. `claim_event_time` is an optional model assertion and
is rejected if reversed.

For this 226 compatibility smoke, stable requirements come from the frozen slot
template. IDs are mechanically formed as `question_id::slot_id`; criticality is
the frozen `essential OR conditional_essential` value. This is input data, not
event-specific program logic.

## Layer 3: legacy adapter

The adapter emits the fields consumed by the frozen V2.1 reliability gate,
handoff, and Gemini review pipeline. Compatibility fields never appear in the
LLM schema. The old `time_range` field uses `claim_event_time` when present and
otherwise the evidence envelope strictly as a localization fallback.

The empty legacy `reasoning_summary` exists only because the old validator
requires a string; V3 does not invent an explanation.

## No-API compatibility method

Uniform R2@50, Selective R2@50, and Dense R3 complete pre-Sufficiency packets
are loaded without rerunning Organizer, Planner, retrieval, or audio selection.
For each question, a clearly labelled `not_found` contract fixture is generated.
It asserts no video fact. The fixture validates:

1. V3 input construction and stable requirements;
2. semantic validation and deterministic projection;
3. V2 result-validator compatibility;
4. loading by the existing reliability feature extractor/gate and handoff
   router.

This smoke does not measure Sufficiency accuracy and is not a live inference.

## Run

```powershell
python scripts/experiments/run_egopolice_sufficiency_v3_contract_smoke_v1.py
```
