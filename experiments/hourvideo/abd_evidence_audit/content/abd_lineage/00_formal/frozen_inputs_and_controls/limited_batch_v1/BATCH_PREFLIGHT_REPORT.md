# ABD limited-batch v1 — offline preflight

Status: **PASS; authorization not issued; formal batch not run**.

This adds execution-range control only. The base payload/runtime identity remains commit `27e1350a581fc28ccf710890d19bf644f9f68c2c` and manifest SHA `24e6305d798378c352099ee963b00efb54d5b40b545c34221949dfba36cd36dc`.

## Exact scope

| execution_index | task_id | arm |
|---:|---|---|
| 0 | `A:6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31` | A |
| 1 | `B:6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31` | B |
| 2 | `D:6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31` | D |

- Question: `6fd90f8d-7a4d-425d-a812-3268db0b0342_8_31`
- Maximum provider requests: 3
- Incremental batch cap: US$0.75
- Per-request reservation: US$0.25
- Fourth task is outside the authorization-bound scope and unreachable.

## Offline simulation

The real limited runner path invoked the existing frozen `_execute_task` three times through a capture-only simulated transport. It produced A, B, D in order, then stopped. A second invocation issued zero new requests. Captured model-visible payloads were exactly equal to the existing frozen builder output.

Unit tests cover scope and identity tampering, batch-budget admission, persistent request count/cost, terminal and unknown-outcome recovery, authorization failure, payload equality, and exclusive temporary state.

`authorization_TEMPLATE_NOT_AUTHORIZED.json` is deliberately non-executable. No signed authorization file was created.

Future command template, not executed:

```bash
<PROJECT_MSC_USER_ROOT>/miniconda3/envs/qaego4d_vllm/bin/python scripts/run_abd_limited_batch_v1.py --execute --approval-token APPROVE_ABD_FIRST_QUESTION_BATCH_V1 --approval-file <reviewed-limited-batch-authorization.json>
```

Real API calls: **0**. Formal experiment launches: **0**.
