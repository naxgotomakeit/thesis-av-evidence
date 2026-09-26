# ABD formal candidate — offline preflight

Status: **PASS with explicit network/token uncertainty and a hard budget-authorization block**.

No API, smoke, online token count, connection test, formal generation, scoring, or judge audit was run.

## Passed offline checks

- Frozen tasks: A/B/D each 300; total 900; concurrency 1.
- Order: `canonical GenS question order; rotate ABD/BDA/DAB by zero-based question index modulo three`. Each arm appears exactly 100 times in each of the three within-question positions.
- All 900 final SDK-shaped payloads were serialized using complete maps and real JPEG base64, passed through a capture-only simulated Messages client, and compared after capture. D minus map equals B for 300/300 questions; A/D maps match for 300/300.
- B/D contain 5676 resolved-time text blocks paired with 5676 original JPEG transmissions; no cache-control or backend/result/gold fields were present.
- SDK internal retries and outer retries are both zero. One durable request-start corresponds to at most one physical request.
- Durable store/runtime tests cover normal, invalid, missing usage, provider error/unknown outcome, response recovery, terminal replay prevention, budget reservations, and exclusive runner locking.
- Generation and scorer are separate programs. No score or gold was loaded in this preflight.

## Capacity — estimate, not exact verification

- Assumed context screen: 200000 tokens, not network-verified in this round.
- Largest offline high estimate including the 512-token output reserve: 67181.4 tokens.
- Minimum estimated margin under that assumption: 132818.6 tokens.
- Unresolved tasks under the offline screen: 0.
- Exact provider token counts: **not measured**. Map/text estimates use historical usage calibration; images use dimensions and historical GenS calibration. No map/image truncation or per-arm output reduction is implemented.

## Updated cost and candidate budget scenario

Primary no-cache/no-retry base: $22.452; estimated range $21.023–$24.145. The all-responses-at-512 ceiling scenario is $25.670.

The US$30 value is only a candidate hard-cap scenario. It is **not authorized**. Admission requires `spent + unresolved reservations + $0.25 <= $30`; unknown outcomes retain their reservation and are never treated as zero. The $0.25 hold exceeds the locally assumed maximum one-request cost at 200k input plus 512 output tokens, but that context premise remains network-unverified.

## Still unverified online

- Endpoint/credential validity, live request acceptance, authoritative token counts/billing, and server-side context behavior.
- The formal launcher additionally requires a separately reviewed authorization file matching the frozen config and manifest SHA. No such file has been created.

## Recovery and future launch

Re-running the same guarded command acquires an exclusive process lock. Terminal success/failure tasks are skipped. A durable provider response without terminal state is parsed and finalized without a provider call. A request-start without a durable response becomes `pending_provider_outcome_review` and is not resent automatically.

Future command template, **not run**:

```bash
<PROJECT_MSC_USER_ROOT>/miniconda3/envs/qaego4d_vllm/bin/python scripts/run_abd_formal_v1.py --execute --approval-token APPROVE_ABD_EVAL300_FORMAL_V1 --approval-file <reviewed-authorization.json>
```

Primary comparison is D vs B; supplementary comparison is D vs A. Each arm uses all 300 questions, with failure/missing counted incorrect. C remains the completed historical Direct result.
