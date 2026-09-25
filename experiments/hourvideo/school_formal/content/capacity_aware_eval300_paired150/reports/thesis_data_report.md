# V6.6.2 Local Context-Limited Eval300: complete thesis data report

Audit date: 2026-08-31  
Experiment: `hourvideo_v6_6_2_local_context_limited_eval300_v1`  
Dataset: HourVideo  
Evaluation population: 300 questions from 12 videos  
Canonical result status: `CANONICAL_COMPLETE`  

## 1. Purpose and research question

This experiment evaluates whether two different video-index representations can be consumed by the same local, hierarchical, iterative reasoning workflow.

The primary experimental question is:

> When model roles, downstream prompts, evidence budgets, retry rules and final-answer logic are held fixed, how does a structural object/action index (R1) compare with a semantic-caption hierarchy (R3) inside V6.6.2?

The experiment is also a local-model baseline for later API experiments. It records where the local deployment fails because of context capacity, contract non-compliance or semantic reasoning limitations.

This experiment does **not** test whether a larger API model is better than the local models. It establishes the frozen local workflow and its costs, failure modes and capacity limits.

## 2. Experimental variables

### 2.1 Primary independent variable

| Route | Indexer/Map representation | Meaning |
|---|---|---|
| R1 (`r1_av`) | Structural retrieval map based on objects, local tracking summaries and explicit action/state-change cues | Designed to expose object presence, tracking and action/state-change structure without describing the map as semantic video captions |
| R3 (`r3_2`) | Coarse/Medium semantic-caption hierarchy | Coarse navigation summaries are derived from Medium semantic captions, with exact caption/ASR provenance where available |

The index representation is the intended experimental variable. Both routes use the same question, five options, model family, role decomposition, Shared/Fine/Final objectives, retry budget, investigation rounds and answer logic.

### 2.2 Controlled variables

| Variable | Frozen value |
|---|---|
| Dataset and question order | Same ordered Eval300 |
| Case identity | `question_id` |
| Routes per question | R1 and R3 |
| Planner core task | Same Coarse-region locking task |
| Planner temperature | 0.0 |
| Planner output schema | Same `selected_coarse_ids` contract |
| Planner maximum output | 8,192 tokens |
| Shared objective | Investigate the single underlying answer-critical fact/relation |
| Fine objective | Neutral image observations plus per-requirement confirmed/refuted/inconclusive assessments |
| Final objective | Select exactly one of the five options from Shared's evidence report |
| Maximum investigation rounds | 3 |
| Validation retries | 2 retries after the initial attempt; maximum 3 actual attempts per logical call |
| Maximum Shared citations | 16 unique evidence IDs |
| Shared established-facts length | 1,536 characters |
| Shared gap-reason length | 768 characters |
| Maximum Fine evidence per Medium | 2 |
| Minimum temporal gap between selected Fine evidence | 10 seconds |
| Maximum Fine evidence per claim | 20 |
| Maximum Fine evidence per batch | 16 |
| Fine ranking weights | visual 0.6; lexical 0.3 |
| Pipeline question concurrency | 1 |
| External API | None |
| Provider interface | Local OpenAI-compatible endpoints |

### 2.3 Route-specific input description

The Planner's common task, selection rules, schema and output budget are the same. Only the representation description differs:

- R1 explicitly tells the Planner that the map contains object detections, local tracking summaries and action/state-change cues. It warns that these are structural retrieval cues, not semantic captions or answer labels.
- R3 explicitly tells the Planner that the map is a Coarse/Medium semantic-caption hierarchy. It warns that caption summaries are retrieval cues and are not proof beyond their text.

No extra reasoning strategy, priority rule or answer hint is given to either route.

### 2.4 Dependent variables

The canonical analysis reports:

- fixed-denominator accuracy;
- strict completion and missing predictions;
- `normal_success`, `downgraded_recovery`, `failed` and `context_overflow_pre_model`;
- `confirmed` and `budget_exhausted_guess` reasoning termination;
- Planner and post-Planner calls, tokens and model latency;
- physical Fine image transmissions;
- route end-to-end latency;
- selected-final-path cost versus actual total cost including all retries;
- failure classification.

## 3. Evaluation population

The formal source contains:

| Quantity | Value |
|---|---:|
| Videos | 12 |
| Questions | 300 |
| Questions per video | 25 |
| Answer options per question | 5 |
| Nominal R1/R3 routes | 600 |
| Shared hierarchy Coarse nodes | 186 |
| Shared hierarchy Medium nodes | 838 |
| Shared hierarchy Fine nodes | 2,509 |

Ordered Eval300 UID SHA:

`6db7b3cc672919a60c4b0c4a84f9f2e039e1673bdeddcfe619fdce92f1cb6fa1`

Ordered 600-route manifest SHA:

`d7ca4f3012e6fe2d5aaf4bead281d2905359313d6c7ed11e8e2a172386b6c871`

Three questions overlap with Pilot10. Therefore:

- full set: Eval300;
- held-out set: Eval297, defined before scoring by excluding the three Pilot10 question IDs;
- not Eval290.

The three excluded question IDs are:

1. `819c8af7-851f-434f-ab32-318285bc54b1_11_3`
2. `70f2a750-f403-41b8-aabb-480eb3ab4ed4_21_26`
3. `a6d45e95-8dc0-4932-83bf-ec53e265a16a_8_18`

## 4. Local context-capacity condition

The Qwen3-8B Planner deployment was frozen at:

```text
max_model_len = 36,544
Planner max output tokens = 8,192
eligible iff planner_input_tokens + 8,192 <= 36,544
```

No Map truncation, compression or input reduction was allowed.

### 4.1 Token ranges

| Route | Min input | Max input | Min required total | Max required total |
|---|---:|---:|---:|---:|
| R1 | 1,950 | 22,481 | 10,142 | 30,673 |
| R3 | 14,596 | 47,091 | 22,788 | 55,283 |

### 4.2 Frozen eligibility

| Population | Eligible | Context overflow | Fixed denominator |
|---|---:|---:|---:|
| R1 | 300 | 0 | 300 |
| R3 | 150 | 150 | 300 |
| Total routes | 450 | 150 | 600 |

The R3 capacity limitation is a deployment constraint, not a post-hoc filtering rule. The 150 overflowing R3 routes were fixed before model execution and recorded as:

```text
formal_status = context_overflow_pre_model
model requests = 0
prediction = missing
counts in fixed denominator = true
scored as incorrect = true
```

The capacity-feasible paired subset is the pre-execution intersection of R1 and R3 eligibility. Because all R1 questions are eligible, this is exactly the frozen ordered list of 150 R3-eligible questions. It is not reduced based on completion, prediction presence or correctness.

Paired-150 ordered-list SHA:

`0219ac63d3e577ae2b1aebd36e9ec8a5793d4597c3013d73ff12fd18b8c20057`

## 5. V6.6.2 workflow

```text
Question + five options + route-specific Map
                    |
                    v
Frozen Planner (Qwen3-8B)
  - one plan per option requirement
  - locks plausibly relevant Coarse regions
                    |
                    v
Shared Investigation (Qwen3-8B)
  - reasons about one underlying answer-critical fact/relation
  - may request selectable Coarse regions
  - consumes returned Fine observations on the next round
                    |
              if Fine is requested
                    v
Fine retrieval + visual review (Qwen2.5-VL-7B)
  - excludes previously observed Fine IDs
  - retrieves only new Fine evidence
  - returns neutral per-image observations
  - returns claim-level confirmed/refuted/inconclusive assessments
                    |
                    +------ Observation feedback to Shared
                    |
                    v
Direct Final (Qwen3-8B)
  - sees Shared's final report and at most 16 cited evidence rows
  - emits exactly one option
  - unresolved cases must be marked best_guess
```

Planner generation cost is separated from post-Planner online inference cost.

## 6. Guard and contract settings

The guards are interface and accounting controls. They do not add new semantic evidence or correct the model's reasoning.

### 6.1 Shared Coarse contract

- `selectable_coarse_ids`: Coarse regions with at least one unobserved Fine node.
- `exhausted_coarse_ids`: Coarse regions with no remaining unobserved Fine node.
- Previously selected Coarse regions may be selected again when they still contain unseen Fine evidence.
- Re-selecting a Coarse region can return only previously unseen Fine IDs.
- If no selectable Coarse remains, Shared must return `requested_coarse_ids=[]`.
- The payload explicitly includes selectable, exhausted and previously selected IDs, plus the unobserved Fine count for each selectable Coarse.

### 6.2 Citation contract

- Citation IDs are stably de-duplicated while preserving first occurrence order.
- Original count, de-duplicated count and duplicates are audited.
- Up to 16 unique citations are valid.
- More than 16 unique citations trigger validation retry.
- Unique citations are never silently truncated.

### 6.3 Retry contract

```text
max_validation_retries = 2
maximum actual requests = initial attempt + 2 retries = 3
```

A retry repairs only the same stage and the same evidence input. It cannot increase the evidence budget or silently alter investigation state.

Invalid Shared outputs do not enter valid investigation history. If all Shared attempts fail:

- retain the last valid Shared state when one exists;
- otherwise create an explicit local unresolved safe state;
- record `downgraded_recovery`;
- continue to Direct Final.

A route is marked `failed` only when no valid Final prediction survives the frozen retry contract or when a hard execution/input failure occurs.

### 6.4 Orthogonal terminal status

Execution state:

- `normal_success`;
- `downgraded_recovery`;
- `failed`.

Reasoning termination:

- `confirmed`;
- `budget_exhausted_guess`.

These dimensions are recorded separately; completion does not hide downgraded recovery.

## 7. Models and serving configuration

### 7.1 Qwen3-8B semantic service

| Setting | Value |
|---|---|
| Roles | Planner, Shared, Final |
| Host | `goosander-l.cs.ucl.ac.uk` |
| GPU | NVIDIA GeForce RTX 3090 Ti, GPU 0 |
| Driver | 590.44.01 |
| Model endpoint name | `qwen3-8b-planner` |
| Context | 36,544 |
| Max sequences | 1 |
| dtype | float16 |
| Quantization | bitsandbytes |
| Temperature | 0.0 |
| Thinking | disabled |
| Model/config/tokenizer manifest SHA | `b197b68d7a8745f71eddedf6f39f39ddad5c117110f41c545fd77d3cd5dc920c` |

Stage output limits:

- Planner: 8,192 tokens;
- Shared: 2,400 tokens;
- Final: 3,600 tokens.

### 7.2 Qwen2.5-VL-7B visual service

| Setting | Value |
|---|---|
| Role | Fine visual observation |
| Host | `aylesbury-l.cs.ucl.ac.uk` |
| GPU | NVIDIA GeForce RTX 3090 Ti, GPU 0 |
| Driver | 610.43.02 |
| Model endpoint name | `qwen2.5-vl-7b-visual` |
| Context | 32,768 |
| Max sequences | 1 |
| dtype | float16 |
| Images per prompt limit | 64 |
| Temperature | 0.0 |
| Thinking | disabled |
| Fine output limit | 3,600 tokens |
| Model/config/tokenizer manifest SHA | `05f389fb3d79a9a35b0d0952ac4369346763380108e1788700184aba32b34cd9` |

Both services used:

```text
provider = local_openai
api_key = EMPTY
external paid API = false
pipeline concurrency = 1
```

Service-start failures before endpoint lock were recorded but are excluded from formal route costs: one remote PATH lacked `ninja`, and one alternative visual host failed tokenizer/processor initialization. The final endpoints were locked before the live gate and remained fixed for the formal run.

## 8. Prompt freeze

| Prompt | Content SHA |
|---|---|
| Planner R1 | `c3e27709f94f823f5470e13e21082abdde5dd35b3429913c0d89077387afd799` |
| Planner R3 | `d9b9ccadad1d713620a027118e1ab4919f9bb119b070e340a6f3f9944f908100` |
| Shared | `a1017b5a3542fde6b907e35e8f050623826902c0e15a0696c3c43cb74967a19b` |
| Fine | `f36170283fae6c6e61f05539a88387e9afc10f3694635e75d2d10001e99fcf42` |
| Final | `c7145de296ed0ff79f5b05e1f0bd99e7eddb3d76c0253fde235913ad214223b1` |

The prompt was frozen before final capacity calculation and formal model execution. Accuracy was not used to adjust the prompt.

## 9. R1/R3 asset freeze

Formal R3 used:

`${R3_V74_WORKSPACE}/work_index`

| Asset | SHA-256 |
|---|---|
| R3 source tree | `c0403d8216ba840116f9adbe2501631d63c362595a685cf3e1c3519dba14873e` |
| R3 work-index tree | `ff5d17a92e80a4dddeb5189e1d898647a595f0635c9da446eb61bf32dbbd7d65` |
| Materialization manifest | `4d376dacdebbd5ed2e0fb9229038d5f25aa33565d1e8db3ae3045da4ccefdb87` |
| Final migration archive v3 | `fa2c0c5ff82109daaf36249ebc8ec5b685e4ee73320bb910d6d2d923b347ff6b` |

R1 and R3 share the same downstream configuration. Video-level index assets are reused across the 25 questions belonging to each video; they are not copied 300 times.

## 10. Metric and accounting definitions

### 10.1 Accuracy

```text
accuracy = number of exact option-ID matches / frozen denominator
```

Missing predictions, terminal failures and context overflow remain in the denominator and count as incorrect.

### 10.2 Completion

Strict completion requires a durable valid Final prediction. `downgraded_recovery` with a valid prediction is complete but remains separately labelled. Overflow and failed routes without prediction are incomplete.

### 10.3 Planner cost

Planner calls, input/output tokens and latency are reported separately. Planner cost is not included in post-Planner route E2E.

### 10.4 Post-Planner actual cost

Actual total cost includes every Shared, Fine and Final attempt, including:

- successful attempts;
- schema/validation failures;
- max-token attempts;
- provider errors;
- all retries;
- attempts on routes that eventually fail.

### 10.5 Selected-final-path cost

Selected-final-path cost contains accepted/success attempts only and only for routes with a durable prediction. It is reported for comparison but is not the authoritative resource total.

### 10.6 E2E latency

Post-Planner route E2E starts immediately before online route reasoning and ends at durable terminal prediction/failure. It includes local retrieval, model requests and retries. It excludes Frozen Planner generation.

The 150 pre-model context-overflow routes do not receive an artificial zero latency and are excluded from latency distributions.

## 11. Accuracy results

Wilson 95% confidence intervals are reported as descriptive binomial intervals.

| Route/population | Correct | Denominator | Accuracy | Wilson 95% CI |
|---|---:|---:|---:|---:|
| R1 full Eval300 | 81 | 300 | **27.000%** | 22.290–32.291% |
| R1 held-out Eval297 | 80 | 297 | **26.936%** | 22.209–32.252% |
| R3 capacity-aware Eval300 | 52 | 300 | **17.333%** | 13.470–22.023% |
| R3 held-out Eval297 | 51 | 297 | **17.172%** | 13.309–21.873% |
| R1 paired context-feasible 150 | 51 | 150 | **34.000%** | 26.903–41.896% |
| R3 paired context-feasible 150 | 52 | 150 | **34.667%** | 27.520–42.580% |

The R3 full-set result is deliberately capacity-aware: its denominator is 300 even though only 150 routes could be executed under the frozen local context limit.

## 12. Paired-150 comparison

Because the paired population was frozen before execution, failed or missing routes remain in its denominator.

| Paired outcome | Questions |
|---|---:|
| Both R1 and R3 correct | 37 |
| R1 correct, R3 wrong | 14 |
| R1 wrong, R3 correct | 15 |
| Both wrong | 84 |
| Total | 150 |

The paired accuracy difference is:

```text
R3 - R1 = 34.667% - 34.000% = +0.667 percentage points
```

An exact two-sided McNemar test on the 29 discordant pairs gives `p = 1.0`. This experiment therefore does not provide evidence of an accuracy difference between R1 and R3 on the frozen common context-feasible population.

This does **not** establish equivalence. The interval uncertainty is wide, and both routes use the same capacity-limited local models.

## 13. Completion and terminal states

### 13.1 Full populations

| Route | Predictions | Missing | Strict completion |
|---|---:|---:|---:|
| R1 Eval300 | 292 | 8 | **97.333%** |
| R3 capacity-aware Eval300 | 145 | 155 | **48.333%** |

For R3's 150 executed routes only:

```text
145 / 150 = 96.667% completion
```

### 13.2 Paired-150

| Route | Predictions | Missing | Strict completion |
|---|---:|---:|---:|
| R1 | 148 | 2 | **98.667%** |
| R3 | 145 | 5 | **96.667%** |

### 13.3 Execution status

| Route/population | normal_success | downgraded_recovery | failed | context overflow |
|---|---:|---:|---:|---:|
| R1 Eval300 | 44 | 248 | 8 | 0 |
| R3 executed 150 | 72 | 73 | 5 | 0 |
| R3 full Eval300 | 72 | 73 | 155 | 150 of the 155 failures |

R1 has a high downgraded-recovery rate: `248/300 = 82.667%`. R3's executed population has `73/150 = 48.667%` downgraded recovery.

### 13.4 Reasoning termination

Reasoning termination is defined only for routes with a valid prediction.

| Route | confirmed | budget_exhausted_guess | Predictions |
|---|---:|---:|---:|
| R1 | 12 | 280 | 292 |
| R3 executed | 39 | 106 | 145 |

Most R1 predictions (`280/292 = 95.890%`) were budget-exhausted guesses. For R3, `106/145 = 73.103%` were budget-exhausted guesses.

## 14. Failure classification

### 14.1 R1 failures

| Failure kind | Count |
|---|---:|
| Direct Final contract exhausted | 5 |
| Runtime context overflow during the downstream route | 2 |
| Fine/Final visual response reached max tokens | 1 |
| **Total failed** | **8** |

The two R1 runtime-context failures differ from pre-model Planner eligibility: their Planner inputs fit the frozen capacity rule, but a later Shared request plus its reserved output exceeded the local semantic service context.

### 14.2 R3 failures

| Failure kind | Count |
|---|---:|
| Context overflow before any model request | 150 |
| Direct Final contract exhausted | 2 |
| Fine/Final visual response reached max tokens | 3 |
| **Total failed in fixed denominator** | **155** |

No unexplained Pipeline crash was found.

## 15. Frozen Planner cost

All 450 eligible routes obtained a valid Frozen Planner.

| Planner accounting | Calls | Input tokens | Output tokens | Model latency |
|---|---:|---:|---:|---:|
| Accepted Planner outputs | 450 | 4,942,537 | 353,151 | 5,362.902 s |
| Actual attempts including retry | 451 | 4,946,231 | 361,343 | 5,450.752 s |

One Planner attempt failed and was recovered by the frozen retry policy. Actual Planner model latency was approximately **1.514 hours**.

## 16. Post-Planner total cost

### 16.1 Authoritative actual totals including retries

| Route | Calls | Input tokens | Output tokens | Physical Fine images | Model latency |
|---|---:|---:|---:|---:|---:|
| R1, 300 routes | 3,115 | 39,522,139 | 2,733,517 | 10,567 | 42,039.906 s |
| R3, 150 executed routes | 1,449 | 12,425,668 | 1,280,158 | 5,172 | 18,313.389 s |
| **Combined post-Planner** | **4,564** | **51,947,807** | **4,013,675** | **15,739** | **60,353.295 s** |

Combined post-Planner model latency is approximately **16.765 model-hours**. These are sequentially served model-attempt latencies, not energy measurements.

### 16.2 Selected-final-path totals

| Route | Calls | Input tokens | Output tokens | Physical Fine images | Model latency |
|---|---:|---:|---:|---:|---:|
| R1 | 1,699 | 16,081,690 | 1,241,543 | 6,266 | 20,867.002 s |
| R3 | 858 | 6,720,599 | 620,078 | 3,022 | 10,078.688 s |

Selected-final-path totals exclude failed/validation attempts and all attempts on routes without a prediction. They must not be presented as the actual experiment cost.

### 16.3 Overall local-model totals

Planner actual attempts plus post-Planner actual attempts give:

| Quantity | Total |
|---|---:|
| Actual model requests | **5,015** |
| Input tokens | **56,894,038** |
| Output tokens | **4,375,018** |
| Physical Fine images | **15,739** |
| Summed model-attempt latency | **65,804.046 s = 18.279 h** |

### 16.4 NEW supplementary analysis: paired-150 resource cost

**Added on 2026-08-31 after the original thesis-data summary was assembled.** This is a new read-only aggregation over the pre-execution frozen `common_eligible_question_ids.txt` population. It does not rerun any Planner or Pipeline model, does not replace the full-population totals above, and does not select questions based on completion, prediction presence or correctness.

The paired population contains exactly 150 unique question IDs for each route. Its frozen ordered-list SHA-256 is `0219ac63d3e577ae2b1aebd36e9ec8a5793d4597c3013d73ff12fd18b8c20057`. All 150 R1 rows and all 150 R3 rows remain in the aggregation, including failed attempts and routes without a durable prediction.

#### 16.4.1 Authoritative paired post-Planner cost including all retries

| Metric | R1 paired-150 | R3 paired-150 | R3 reduction from R1 |
|---|---:|---:|---:|
| Actual model requests | 1,831 | 1,449 | **382 (20.863%)** |
| Input tokens | 20,362,068 | 12,425,668 | **7,936,400 (38.976%)** |
| Output tokens | 1,493,281 | 1,280,158 | **213,123 (14.272%)** |
| Total tokens | 21,855,349 | 13,705,826 | **8,149,523 (37.288%)** |
| Physical Fine-image transmissions | 7,222 | 5,172 | **2,050 (28.385%)** |
| Summed model-attempt latency | 22,890.635 s | 18,313.389 s | **4,577.247 s (19.996%)** |

These are the authoritative resource totals for the controlled paired comparison. They include every Shared, Fine and Final attempt, including validation failures, max-token responses, retries and attempts on routes that did not ultimately produce a prediction. Request-start and attempt-end counts match one-to-one for all 1,831 R1 attempts and all 1,449 R3 attempts.

Per frozen paired question, R1 used 12.207 requests, 145,702.327 total tokens and 48.147 Fine-image transmissions; R3 used 9.660 requests, 91,372.173 total tokens and 34.480 Fine-image transmissions.

#### 16.4.2 Paired selected-final-path cost

| Metric | R1 | R3 |
|---|---:|---:|
| Requests | 1,067 | 858 |
| Input tokens | 10,073,487 | 6,720,599 |
| Output tokens | 800,913 | 620,078 |
| Physical Fine-image transmissions | 4,252 | 3,022 |
| Model latency | 13,434.976 s | 10,078.688 s |

This secondary table retains only accepted/success attempts on routes with a durable prediction: 148 R1 routes and 145 R3 routes. Because completion differs, it is not the primary fixed-population cost comparison and must not replace the actual-total table in Section 16.4.1.

#### 16.4.3 Planner and complete-Pipeline paired cost

Frozen Planner generation remains separate from post-Planner online inference. R3's larger semantic hierarchy makes its Planner stage more expensive even on the context-feasible subset.

| Planner metric | R1 paired-150 | R3 paired-150 |
|---|---:|---:|
| Actual requests | 150 | 150 |
| Input tokens | 880,811 | 2,799,986 |
| Output tokens | 112,417 | 117,447 |
| Total tokens | 993,228 | 2,917,433 |
| Model latency | 1,370.546 s | 2,299.801 s |

After adding actual Planner attempts to all actual post-Planner attempts, the complete local workflow remains less resource-intensive for R3 on the paired subset:

| Complete-Pipeline metric | R1 paired-150 | R3 paired-150 | R3 reduction from R1 |
|---|---:|---:|---:|
| Actual model requests | 1,981 | 1,599 | **382 (19.283%)** |
| Input tokens | 21,242,879 | 15,225,654 | **6,017,225 (28.326%)** |
| Output tokens | 1,605,698 | 1,397,605 | **208,093 (12.960%)** |
| Total tokens | 22,848,577 | 16,623,259 | **6,225,318 (27.246%)** |
| Physical Fine-image transmissions | 7,222 | 5,172 | **2,050 (28.385%)** |
| Summed model-attempt latency | 24,261.182 s | 20,613.189 s | **3,647.992 s (15.036%)** |

The latency reduction is descriptive and hardware-specific. The controlled evidence-efficiency finding is that, on the same frozen 150 questions and under the same local workflow contract, R3 achieves statistically indistinguishable paired accuracy while using fewer post-Planner requests, tokens and visual inspections. This does not remove R3's separately reported context-capacity limitation on the other 150 Eval300 questions.

No external token charge was incurred because all calls were to local services. No authoritative electricity price, GPU rental rate or power telemetry was recorded; therefore the report does not convert local execution into a monetary cost.

The observed formal tmux interval was approximately **18 hours 14 minutes**, from the 2026-08-30 evening launch to final aggregation at 2026-08-31 13:03 BST. This wall interval is descriptive and includes serial orchestration, local retrieval, serialization and request overhead; it is not interchangeable with summed model-attempt latency.

## 17. Post-Planner cost by stage

### 17.1 R1

| Stage | Calls | Input tokens | Output tokens | Images | Model latency |
|---|---:|---:|---:|---:|---:|
| Shared Investigation | 2,110 | 32,432,762 | 1,847,413 | 0 | 24,937.781 s |
| Fine claim execution | 689 | 6,441,039 | 791,268 | 10,567 | 16,036.280 s |
| Direct Final | 316 | 648,338 | 94,836 | 0 | 1,065.845 s |
| **R1 total** | **3,115** | **39,522,139** | **2,733,517** | **10,567** | **42,039.906 s** |

### 17.2 R3

| Stage | Calls | Input tokens | Output tokens | Images | Model latency |
|---|---:|---:|---:|---:|---:|
| Shared Investigation | 953 | 8,912,399 | 807,365 | 0 | 9,430.908 s |
| Fine claim execution | 334 | 3,200,590 | 401,056 | 5,172 | 8,116.329 s |
| Direct Final | 162 | 312,679 | 71,737 | 0 | 766.152 s |
| **R3 total** | **1,449** | **12,425,668** | **1,280,158** | **5,172** | **18,313.389 s** |

The stage names are V6.6.2-native stages and must not be renamed as VideoSEAL stages.

## 18. Post-Planner E2E latency

### 18.1 Full route populations

| Route | Latency samples | Sum | Mean | Median | P90 | P95 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 Eval300 | 300 | 42,123.317 s | 140.411 s | 133.700 s | 218.310 s | 249.568 s | 25.075 s | 320.890 s |
| R3 executed | 150 | 18,341.848 s | 122.279 s | 121.942 s | 244.205 s | 256.071 s | 3.887 s | 305.678 s |

R3's 150 overflow rows are excluded from the latency distribution rather than being inserted as zero-second tasks.

### 18.2 Paired context-feasible 150

| Route | Count | Mean | Median | P90 | P95 |
|---|---:|---:|---:|---:|---:|
| R1 | 150 | 152.912 s | 150.305 s | 232.646 s | 259.065 s |
| R3 | 150 | 122.279 s | 121.942 s | 244.205 s | 256.071 s |

Absolute latency is descriptive because the two models ran on separate RTX 3090 Ti hosts, and other baselines may run on different hardware. Accuracy, completion, failure counts, fixed denominators and model-call counts may be compared across machines; absolute E2E/GPU time must not be used to claim a hardware-independent speedup over VideoSEAL.

## 19. Live-gate provenance

Gate question:

`4572b198-2c1c-4920-bcf0-95fcebe12261_13_1`

The original formal resume mechanism reused the two durable gate trajectories instead of rerunning that question. Canonical aggregation therefore:

- keeps one R1 and one R3 record for this identity;
- labels them `live_gate_promoted_single_canonical_route`;
- counts no additional gate route;
- prevents the gate from becoming a 301st R1 or 151st executed R3 route;
- prevents cross-route gate telemetry from contaminating another route's totals.

## 20. Canonical reconstruction and validation

The legacy top-level `live/validation_report.json` mixed cumulative cross-side/live-gate telemetry and was overwritten by the final per-side pass. It remains on disk unchanged but is marked:

`INVALID_AGGREGATION_DO_NOT_USE`

Canonical aggregation was rebuilt from question-level route status, Final prediction, attempt telemetry and route E2E events.

Validation results:

| Check | Result |
|---|---|
| Canonical routes | 600/600 |
| Unique `(question_id, route)` identities | 600/600 |
| R1 identities | 300 |
| R3 identities | 300 |
| R3 executed | 150 |
| R3 pre-model overflow | 150 |
| Frozen paired set | 150 |
| Request-start/attempt-end pairing | PASS |
| Overflow with model request | 0 |
| Overflow assigned zero E2E | 0 |
| Gold loaded before structural validation | No |
| Structural validation | PASS |
| Complete regression tests | **541 passed** |

The public `score_fixed_population` evaluator was loaded only after the no-gold structural validation passed.

## 21. Reproducibility SHA

| Artifact | SHA-256 |
|---|---|
| Raw formal output tree | `7cbd0892a09247a774675b53d0893d739bca91e1d8142fd9e44ce853d82a26b2` |
| Raw output freeze manifest | `303cdc9e2018a6acca07cc4dd16efa1a7eb198fd7a55f41c4fc2c130fef2c084` |
| Canonical routes | `b3da9581594ef5c30fb725d08d803568e23eaca15d77df5bba9bc82382984b5a` |
| Accuracy | `10f616fe9479fc61e19b17e9cdde30698a85b05733c00182af04abc7ae44abd1` |
| Completion | `099fb554856b712714f0365bd5359ccc484d069dcf35a77e34d20412ffc0391f` |
| Latency | `dd9ae333daa91b3aaacc081d3127d375dcfb91cd30f3ac30eb5d2d599b20de75` |
| Calls/tokens/images | `e2395015d000ef5c3fd08a5f1f06b4aad38ddeebecdadee06db9d986af74fdeb` |
| Failure classification | `7b61e18a706bad2ce85190ded7a357286070eee1e7c1a9cc7c07476eb2a9c601` |
| Canonical validation | `34776a048e900af905aadbf6615661a536d9dfacc7ac7bb091d75743c5d9261d` |
| Canonical manifest | `7041993d0a432c4181a5c18890e44398ee1af28b6c7e502ea66f61630e313fc1` |
| Config | `ed7c3da5e86b5e35bd182ec307d27aaaf323497c999776c1a815fcc3fde84e4e` |

The raw formal tree contains 5,114 files and 152,042,089 bytes. It was rehashed after canonical aggregation and remained byte-for-byte unchanged.

## 22. Interpretation for the thesis

The full capacity-aware Eval300 result should be used to describe deployability under the frozen local context limit:

- R1 can run all 300 questions and obtains 27.0% accuracy.
- R3 can run only 150 questions because its semantic hierarchy produces much larger Planner inputs; counting all 150 overflow routes as incorrect gives 17.33% full-set accuracy.

The paired-150 result should be used for the controlled R1/R3 evidence-representation comparison:

- R1: 34.0%;
- R3: 34.67%;
- difference: +0.67 percentage points for R3;
- exact McNemar `p=1.0`;
- R3 uses 20.86% fewer post-Planner model requests, 37.29% fewer total post-Planner tokens and 28.39% fewer Fine-image transmissions when every attempt and retry is counted over the same frozen 150 questions.

Therefore, this local experiment does not show a meaningful paired accuracy difference between the two Indexer representations, but the newly added paired-cost aggregation shows that R3 consumes less downstream evidence-processing resource on the common feasible population. The complete Planner-plus-Pipeline accounting still favours R3 by 19.28% in requests, 27.25% in total tokens and 28.39% in Fine-image transmissions, even though R3's longer Map makes its Planner stage more expensive. Separately, the experiment shows a substantial representation-capacity difference: the uncompressed R3 semantic Map exceeds the local Planner context limit for half of Eval300, whereas R1 fits for all questions.

The high number of downgraded recoveries and budget-exhausted guesses supports a more limited conclusion:

> The guards largely preserved executable structure and durable predictions, but the local models frequently failed to reach a confirmed evidence state. The experiment therefore exposes local context and model-compliance limitations; it does not demonstrate that guard-heavy orchestration solves semantic video reasoning.

The results cannot isolate model capacity from workflow complexity because both R1 and R3 use the same Qwen3-8B/Qwen2.5-VL-7B local stack. A later Staged API experiment can retain the workflow while changing model capability, but it is a new experimental factor and must be reported as such.

## 23. Thesis-ready concise statement

> V6.6.2 compared a structural object/tracking/action-state Map (R1) with a semantic-caption hierarchy (R3) under the same local hierarchical reasoning workflow. The experiment used 300 HourVideo questions from 12 videos, Qwen3-8B for Planner/Shared/Final, Qwen2.5-VL-7B for Fine visual observation, temperature zero, concurrency one, three investigation rounds and at most three attempts per contract-bound model call. Under the frozen 36,544-token Planner context, all 300 R1 routes were feasible but only 150 R3 routes were feasible; the other 150 were retained as pre-model context-overflow errors in R3's fixed denominator. Full-set accuracy was 27.0% for R1 and 17.33% for capacity-aware R3. On the pre-execution paired context-feasible subset, R1 obtained 34.0% and R3 34.67%; the one-question difference was not significant under an exact paired McNemar test (`p=1.0`). A new read-only paired-cost aggregation found that R3 used 20.86% fewer post-Planner requests, 37.29% fewer post-Planner tokens and 28.39% fewer Fine-image transmissions over the same frozen 150 questions; after including the more expensive R3 Planner, the complete R3 workflow still used 19.28% fewer requests and 27.25% fewer tokens. R1 produced predictions for 292/300 questions, while R3 produced 145/300 overall and 145/150 among executable routes. The complete local experiment used 5,015 model requests, 56.894 million input tokens, 4.375 million output tokens and 15,739 Fine image transmissions. These results show that the semantic hierarchy reduced downstream evidence-processing cost on its context-feasible population but imposed a major local context-capacity burden on the full benchmark; contract guards preserved execution more effectively than they resolved semantic reasoning.

## 24. Evidence locations

- Formal configuration: `configs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1.json`
- Prompt freeze: `outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/prompts/frozen_v1`
- Capacity records: `outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/eligibility`
- Frozen Planner: `outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/frozen_planner`
- Raw formal routes: `outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/live/cases`
- Canonical summary: `outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/canonical_summary_v1`
- Canonical manifest: `outputs/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/canonical_summary_v1/canonical_manifest.json`
- Canonical aggregation code: `src/experiments/hourvideo_v6_6_2_local_context_limited_eval300_v1/canonical_summary.py`
- Reusable fixed-population aggregator: `src/experiments/canonical_question_route_summary_v1/core.py`

## 25. Exclusions and limitations

Not included in the reported costs:

- offline R1/R3 index construction;
- source MP4 storage;
- model-weight storage;
- failed service-start attempts before final endpoint lock;
- electricity and power consumption;
- internal institutional GPU price;
- any API charge, because no external API was used;
- VideoSEAL or GenS preprocessing and online execution;
- later Staged API or Direct API experiments.

The following limitations must accompany thesis claims:

1. R3 full Eval300 accuracy is jointly affected by evidence representation and the frozen local context limit; use paired-150 for the controlled representation comparison.
2. Paired-150 is context-feasible, not a random sample of Eval300.
3. Both routes use capacity-limited local models, so a null paired difference does not prove the representations are equivalent under stronger models.
4. Absolute timing is hardware-specific and must not support cross-machine speed claims.
5. `downgraded_recovery` predictions are valid outputs but explicitly indicate contract recovery, not normal reasoning completion.
6. `budget_exhausted_guess` indicates that the workflow reached its frozen budget without confirming the answer-critical fact.
7. The selected-final-path accounting is not a substitute for actual cost including retries.
8. The legacy top-level `validation_report.json` is invalid for analysis; only the versioned canonical summary should be used.
