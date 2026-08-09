# HourVideo V6: coarse-locked claim/audit loop — design plan

## Status

Implemented and run end-to-end against real data on 2026-08-06 for the diagnosed video
(`a6d45e95-8dc0-4932-83bf-ec53e265a16a`). Code lives at
`src/experiments/hourvideo_r1_av_r3_2_coarse_locked_claim_loop_v6/` (mirrored into
`hourvideo_v3_runtime/` as a portable snapshot — that tree has no `outputs/` of its own; the real
ten-video pilot output that V6's `source_experiment` points at lives under the repo-root `outputs/`,
not under `hourvideo_v3_runtime/outputs/`). It does not change V1/V3/V3.1 behavior; those remain frozen.

### First real run: result and four bugs only visible against real data

r3_2 (map-first) resolved `option_a` from Coarse/Medium map evidence alone in round 0 — no visual
escalation at all — and selected **A**, matching gold, with an honest caveat about the interleaved
boundary. r1_av escalated through all 3 claim/audit rounds and confidently selected **C**, wrong
against gold. This is exactly the kind of side-by-side R1-vs-R3 trace the framework exists to produce;
it is a single-video anecdote, not an accuracy claim.

Four defects only surfaced by running against the real ten-Coarse-region video, none catchable by
schema/validator unit tests alone:

1. **Anthropic's structured-output "compiled grammar" has a size limit independent of the documented
   `oneOf`/`minItems` restrictions.** The original `coarse_judgments` design (an object keyed by every
   Coarse region, repeated per requirement) returned `400 invalid_request_error: "compiled grammar is
   too large"` on a 12-Coarse-region map with 5 requirements. Fixed by switching to an array with one
   shared item schema (`{coarse_id, selected, reason}`) instead of one full sub-schema copy per
   Coarse region — same information, no repeated sub-schema.
2. **`const` is not supported on array/object types in Anthropic's structured output** ("Schema type
   'const - complex' is not supported"), only on primitives (`const: false` on a boolean works
   elsewhere in this codebase). This ruled out forcing `requested_coarse_ids: []` via schema when a
   requirement had nothing excluded to request. Fixed by keeping an unconstrained array in that case
   and sanitizing the response afterward (`_call_sfc_claim` clips `requested_coarse_ids` to the allowed
   set post-hoc) instead of relying on the schema to prevent it.
3. **Gemini returned `HTTP 400 invalid_request` once a locked-Coarse candidate set produced too many
   Fine images in one call.** A legitimately broad but correct lock (7 of 12 Coarse regions, ~23
   Mediums) produced 46 candidate Fine images; 30 worked in a bisection test, 46 did not. Added
   `ranking.max_total_fine_evidence_per_claim` (20) as a hard cap, truncating the already
   highest-relevance-first Fine list. This is the concrete form the deferred "claim-adaptive sampling
   density" open item turned out to need in practice — it is a safety cap, not the eventual
   dense-near-a-boundary-vs-spread-across-a-region design.
4. **Both `_call_sfc_claim` and `_execute_claim` could receive an internally-inconsistent structured
   response from the provider** (e.g. Gemini's returned `observations` list not matching the requested
   Fine ID order/coverage) on an intermittent, non-reproducible basis — re-running the identical call
   immediately afterward succeeded. Neither is a code bug; `temperature: 0.0` does not guarantee
   bit-exact determinism from a hosted LLM. Added a bounded retry (`max_validation_retries`, default 2)
   around both calls: retry once on a validation `ValueError`, then raise.

Practical note: `planner_max_tokens` had to go from 2400 (v1's original value, sized for a short
`suggested_coarse_ids` array) to 8000, since the planner now writes a reason string for every Coarse
region per requirement rather than just naming the winners.

## Why V6 exists

The thesis's core experimental variable is the **indexer** (Fine/Medium/Coarse hierarchy construction — V3's fixed 15s/45s windows vs V3.1's adaptive DINO/CoMET/Safe-Merge/Fluid-Loose frontier, and future indexer variants). The reported ablation compares indexer variants while holding the downstream (planner → retrieval → sufficiency → visual review → final) fixed. V3 stays completely frozen as that control baseline; it must not be edited for this work.

A single-video closed-loop diagnostic on V3 (documented in `hourvideo_r1_av_r3_2_ten_video_pilot_v3_1.md`) exposed that the *downstream* has its own defects independent of which indexer feeds it — defects severe enough that a better indexer might not even show up as a difference in the final answer, because downstream currently ignores the map's own structure. V6 is a new, general-purpose downstream line that fixes this, built by forking V3's actual code (not V5's, which never fully worked and is import-disconnected from V3). V5's *ideas* (typed decisions, sufficiency-driven escalation, abstain-instead-of-guess) carry over; its code and its failed full-relation-contract-compiler mechanism do not.

V6's generality is itself a claimed contribution, not merely ablation plumbing: other related work generally does not adapt its downstream matching logic for adaptive/variable-granularity indexers. See `hourvideo_v6_indexer_downstream_scope` (assistant memory) for the fuller rationale.

## Diagnosed problems V6 must fix

Traced from a real duration-comparison question (`a6d45e95-8dc0-4932-83bf-ec53e265a16a`, "how long was recurring kitchen-organization activity") run through the unchanged V3 pipeline:

1. **Planner uses one generic template for every question.** `option_requirements()` (`hourvideo_r1_av_r3_2_ten_video_pilot_v1/live.py:46-59`) always emits one hypothesis-to-verify per answer option, regardless of whether the question needs map-reading, boundary-finding, or plain object lookup. No question ever gets routed differently.
2. **Planner's Coarse-region enumeration is silently incomplete.** For a recurring/interruptible activity, planner named only the most recent matching Coarse region (e.g. C06/C07/C09) and never mentioned earlier matching regions (e.g. C03). Because omitted regions are simply absent from the output, this failure is invisible downstream — nothing can ever notice or correct it.
3. **Retrieval does not use Coarse as a search boundary at all.** `_rank_requirement` (`live.py:247-267`) ranks *every* Medium node in the whole video by `0.6×SigLIP + 0.3×lexical`, and `_requirement_retrieval` (`live.py:287-328`) only uses planner's `suggested_coarse_ids` as a soft sort-priority before a global Top-K truncation (`top_k_medium_per_requirement`, currently 3). Consequences: (a) hinted regions can still be dropped if more than K of them compete for the same K slots; (b) never-hinted regions can still be selected purely because they score well on embedding similarity. The Coarse-level map — the actual object of the indexer ablation — is not functioning as a hard retrieval scope, so a better indexer's Coarse segmentation may not even change what evidence gets used.
4. **R1 and R3 evidence construction is asymmetric.** `_initial_evidence` (`live.py:347-379`): the R1 branch depends on retrieval's Top-K-truncated `selected_medium_ids`; the R3 branch (`live.py:364-375`) instead reads planner's `suggested_coarse_ids` directly with no truncation at all. The same planner output produces different downstream coverage behavior on the two sides.
5. **The post-visual-review "sufficiency" step is dead code, not a check.** `project_resolved_sufficiency` (`hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3/core.py:141-179`) blindly promotes a requirement to `supported` whenever Gemini's visual-comparison status says `supported` (line 150-152) — it never inspects *what* was supported, only the label. In the diagnosed case, Gemini asserted specific numeric durations (~200s organization, ~100s cooking-prep) inside free-text rationale with nothing forcing it to expose checkable evidence for that arithmetic, and the code accepted the label at face value.
6. **`not_observed` never downgrades anything, and Final's guard is too narrow.** The code comment at `core.py:159` states plainly that `not_observed` "preserves stronger caption/ASR support but cannot create it" — i.e. it is a no-op. Final's only guard (`core.py:280-282`) blocks selecting an option explicitly rejected by `not_found`/`conflicted` while a stronger candidate exists; it does not block an option whose apparent support is really just caption-derived inference that visual review never actually confirmed.
7. **No policy ties "how much to trust the map" to the map's own reliability.** There is currently no mechanism that checks the map first and only escalates to (costly) visual review when the map itself is insufficient — and no way to make that escalation threshold a function of indexer quality, which is the actual ablation variable.

## Architecture

### Roles

**Planner** — stays lightweight; it also parses the question/options and emits requirements, so map-reading must stay cheap (Coarse-level navigation summaries only, never Medium/Fine detail). Per question, it now outputs two things instead of one:

1. A free-text search description / query variants — the same role `search_description`/`query_variants` already play today; used later to rank Medium/Fine *within* whatever Coarse region ends up locked.
2. A judgment for **every** Coarse region present in this video's map, not just the ones it favors: `{coarse_id, selected: bool, reason: string}`. This replaces today's `suggested_coarse_ids` array, which only lists winners and leaves losers invisible. Making every exclusion explicit-with-a-reason converts problem 2 (silent incomplete enumeration) from an unrecoverable failure into a recoverable one: a later stage can read the stated reason and challenge it.

Schema note: this is a flat, per-known-ID-keyed structure (coarse_ids are enumerable per video, exactly like the existing `_keyed_planner_schema` pattern at `live.py:100-123`), so it stays inside the JSON-schema subset Claude's structured-output mode actually supports (no `oneOf`/`anyOf`, no `minItems`/`maxItems`/numeric bounds — see the V5.3 postmortem below).

**SFC (sufficiency check)** — no longer a pass-through validator. It reads the Coarse regions planner marked `selected: true`, plus their Medium-level detail (captions/ASR/detector summaries), and decides whether it can answer or exclude options directly from that alone. It also reads planner's stated reasons for the *excluded* regions: if a reason looks weak given what SFC now sees in the included regions' detail, it pulls that **specific** excluded region back in — a targeted, named expansion, never a full re-scan of the whole map (SFC re-scanning everything would just duplicate planner's job).

- If answerable from what it has: SFC emits a **claim** — a natural-language assertion, scoped to one option/requirement (existing one-requirement-per-option granularity from `option_requirements()` is kept unchanged). Comparative content ("kitchen time exceeds cooking time") is written as prose inside that one option's claim — deliberately *not* factored into a separate cross-option relation structure. See the V5.3 postmortem for why.
- If not answerable: the claim instead states why not, and names the specific gap Gemini should investigate next (e.g. "confirm the start and end boundary of kitchen-organization in region C03").

**Gemini** — executes exactly what the claim asks: verify a specific claim, or investigate a specific named gap. It reports back in free text; there is no requirement to expose structured, checkable sub-fields (e.g. explicit boundary-frame IDs). SFC's next-round audit judges the *reasonableness* of Gemini's stated rationale, not its structural completeness — this is a deliberately light-weight check, not a full evidence-completeness verifier.

### The loop

```
① SFC reads locked-Coarse Medium detail
   → answerable from map alone → claim = assertion → go to ②
   → not answerable → claim = why-not + what's-needed-next → go to ②

② Gemini executes the claim (verify assertion, or investigate named gap)
   → reports back in free text

③ SFC audits Gemini's rationale for reasonableness
   → reasonable, resolves the requirement → done, proceed toward Final
   → not reasonable / still insufficient → new claim (may widen the locked
     Coarse set if that's the identified gap) → back to ②

capped at 3 rounds total
```

If the loop resolves within budget, Final proceeds as today (`_final`/`_final_schema`, `live.py:212-224, 429-438` — unchanged).

If the loop exhausts its 3-round budget without a confident resolution: **do not withhold an answer.** HourVideo is scored multiple-choice, so a withheld answer is strictly worse than a guess under standard scoring. Submit the closest map-derived guess, but the `caveats` field (already present in `_final_schema`) must say plainly that the answer is an unconfirmed map-based guess, not visually confirmed. The earlier framing of "hard-gate on `not_observed`" was really about *not dishonestly labeling a low-confidence answer as `supported`* — it was never about refusing to submit a choice.

### Retrieval becomes Coarse-scoped

This is the core structural fix for problem 3. Today's `_rank_requirement`/`_requirement_retrieval` rank every Medium node in the whole video and only use Coarse hints as a soft sort-priority ahead of a global Top-K cut. In V6, retrieval candidates are restricted to Medium nodes whose `parent_coarse_id` is in the **currently locked** Coarse set — nothing outside that set is eligible at all. The locked set starts as planner's `selected: true` regions and grows only when SFC's audit (step ③ above) names a specific region to pull back in. Growth is always targeted at a named region, never a re-opening of the whole video.

### Fine/Medium sampling within a locked Coarse — placeholder, now with a real cap

A locked Coarse region can span multiple Medium nodes; not every Fine frame inside it can be shown to Gemini (cost, and — discovered against real data, see below — Gemini's own request-size limit). The eventual goal is claim-adaptive sampling: dense sampling right at a named boundary/clue when the claim asks about a precise moment, spread-out sampling across the whole region when the claim asks about an overall pattern. That refinement is **still deferred**. What shipped first: the existing, unmodified per-Medium selection (`rerank_fines_for_medium` + `select_diverse_fines` in `fine_reranking/core.py`, up to `max_fine_per_medium` fines per Medium with `minimum_fine_gap_sec` temporal spacing), re-scoped to run over every Medium node in the locked Coarse set — plus a hard total cap (`ranking.max_total_fine_evidence_per_claim`, 20) added after a real run hit Gemini's request-size limit at 46 images. The cap truncates the already highest-relevance-first list, so it degrades by dropping the least-relevant locked Mediums rather than failing the call. Revisit the claim-adaptive refinement once this cap's own failure modes are visible from more runs.

### Requirement granularity — kept unchanged, on purpose

`option_requirements()` still emits exactly one requirement per answer option. Claims are free text attached to that existing unit. This is a deliberate constraint, not an oversight: it is what keeps V6's blast radius contained to the files being forked, and it is what V6 must **not** relax.

## Why not reuse V5's code, and why claims stay free text (V5.3 postmortem)

V5.3 (`hourvideo_adaptive_question_operator_loop_v5_3_minimal_contract`) tried to have an LLM generate a complete relation-contract graph in one shot — discriminators, mutual-exclusion groups, anchor/outcome/clause/soft slots — via Claude's `json_schema` structured-output mode. It failed for two distinct reasons, confirmed from `outputs/experiments/hourvideo_adaptive_question_operator_loop_v5_3_minimal_contract_generation_ten_v1/`:

1. **Real provider limits** (unavoidable, not a prompt problem): Claude's structured-output mode rejects `oneOf`/`anyOf`/`allOf` unions and strips `minItems`/`maxItems`/`minimum`/`maximum`. A polymorphic `proposition.kind` had to be flattened into one object with 15 optional fields, with correctness pushed into Python post-hoc validation.
2. **Even after working around (1), single-pass generation was ~50% unreliable**: 5 of 10 real test cases needed `contract_repair_required` — wrong role classification, empty mutual-exclusion arrays, dangling ID references — and V5.3 was shelved before ever trying a repair loop (`generation_ten_v1.md` line 6: "zero repair calls").

V6 avoids this failure mode entirely by never asking an LLM to generate a formal cross-requirement relation graph. Claims are always free-text prose scoped to a single existing requirement (one per option). Comparative or relational content is expressed as normal sentences inside that scope, verified independently — there is no cross-option schema for an LLM to get wrong. Question-type/typed-operator thinking from V5 is carried over as *design influence* (route differently by what evidence-gathering action a question needs), not as reused schema or code.

## Implementation basis

V6 is a fork of the actual working V3 pipeline, not a clean-room rewrite and not a fork of V5:

- `hourvideo_r1_av_r3_2_ten_video_pilot_v1/live.py` — planner, retrieval, initial evidence, initial sufficiency.
- `hourvideo_r1_av_r3_2_question_symmetric_visual_review_v3/core.py` — visual comparison, resolved sufficiency, final answer.

Both files must remain byte-identical in place; V6 lives in new sibling modules under a new experiment name, writing to a new output root (V3's stage-cached JSON files, e.g. `planner.json`, are read by file-existence-check, so any schema change against the *same* output directory would silently serve stale cached data rather than error).

## Open items carried forward

- Claim-adaptive Fine/Medium sampling density (deferred placeholder, see above).
- `preflight`'s image-transmission budget formula (`live.py:476-490`, `max_images = option_count * 2 * top_k_medium_per_requirement * max_fine_per_medium`) assumes a fixed global Top-K and will need re-deriving once retrieval is Coarse-scoped; this is bookkeeping, not an architectural decision, and can wait until implementation.
- How `evaluate()` should report/score the caveats-flagged "budget exhausted, map-based guess" answers as a distinct category (vs. confidently-resolved answers) for later analysis — not yet decided, low priority until there is live data to look at.
