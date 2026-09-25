# Direct-v1.2 3/16 R1/R3 Eval300: complete thesis data report

Audit date: 2026-09-05  
Experiment: `direct_v1_2_3x16_r1_r3_eval300_formal_v1`  
Dataset: HourVideo Eval300  
Evaluation population: 300 questions from 12 videos  
Canonical result status: `FINAL CANONICAL DIRECT R1/R3 EVAL300 RESULT`  

## 1. Purpose and research question

This experiment evaluates whether two reusable offline video-map representations support different accuracy and online evidence cost when consumed by the same single multimodal Direct agent.

The primary experimental question is:

> When the model, Direct prompt, tool contract, controller, visual budget, retry policy, frame resolver, cache policy and pricing are held fixed, how does a structural R1 navigation map compare with a hierarchical semantic R3 navigation map on HourVideo Eval300?

The experiment tests the thesis hypothesis that a reusable offline video organisation can guide question-conditioned navigation and selective inspection of original video evidence. The map is treated as a navigation and semantic guide rather than a mandatory substitute for the original frames.

Unlike the earlier V6.6.2 local pipeline, Direct-v1.2 does not decompose online inference into Planner, Shared Investigation, Fine reasoning and Final stages. One Claude multimodal agent reads the native map, decides whether visual evidence is needed, selects temporal locations, integrates returned original frames and emits the final A–E answer.

## 2. Experimental variables

### 2.1 Primary independent variable

| Route | Native map representation | Intended role |
|---|---|---|
| R1 | Structural/ASR-oriented navigation representation with object, action, state and tracking information where available | Expose structural events and temporal cues that resolve back to original frames |
| R3 | Hierarchical Coarse/Medium/Fine semantic organisation with temporal event structure | Expose semantic event relationships that resolve back to original frames |

The native map representation is the intended method difference. No method-specific online retrieval, Top-K frame preselection or requirement decomposition is introduced.

### 2.2 Controlled variables

| Variable | Frozen value |
|---|---|
| Dataset and question order | Same frozen HourVideo Eval300 |
| Questions | 300 |
| Routes | 300 R1 + 300 R3 |
| Model | `claude-haiku-4-5-20251001` |
| Provider | Anthropic |
| Prompt version | `direct_v1.2` |
| Temperature | 0.0 |
| Maximum output | 512 tokens per provider response |
| Provider timeout | 120 seconds |
| Direct agent | Same implementation for R1 and R3 |
| Tool contract | Same dynamic `inspect_frames` and `final_answer` tools |
| Maximum new physical images per round | 3 |
| Maximum unique physical images per question | 16 |
| Maximum route turns | 32 |
| Provider transport retry | One retry maximum |
| Structural action correction | One correction opportunity maximum |
| Frame resolution | Same deterministic timestamp-to-original-1-fps-frame resolver |
| Prompt cache | Anthropic ephemeral cache boundary after verbatim native map; 5-minute TTL |
| Formal budget | USD 50 hard cap |
| Route concurrency | Sequential; no concurrency inside a same-map block |
| Route order | For each video, all R1 questions followed by the matching R3 questions |

### 2.3 Excluded online components

The formal Direct path does not invoke:

- the old Planner or Planner-only output;
- Shared Investigation;
- the old Fine reasoning module;
- a staged Final module;
- SigLIP query retrieval or Top-K frame preselection;
- requirement-level Coarse locking;
- VideoSEAL;
- GenS;
- previous Local, Planner-API, smoke or pilot predictions.

## 3. Evaluation population and route order

| Quantity | Value |
|---|---:|
| Videos | 12 |
| Questions | 300 |
| Answer options per question | 5 |
| R1 routes | 300 |
| R3 routes | 300 |
| Total routes | 600 |
| Durable terminal routes | 600 |
| Predictions | 598 |
| Terminal failures | 2 |

Frozen population and ordering fingerprints:

| Artifact | SHA-256 |
|---|---|
| Eval300 population manifest | `25d98f2e0140ef0a85fcbd66fb40b9e844ed79e64e4d28cfecab1a74805aa859` |
| Ordered 600-route source | `d7ca4f3012e6fe2d5aaf4bead281d2905359313d6c7ed11e8e2a172386b6c871` |
| Question-input closure | `1fed7afcf22d7f7e77fa7220132ba2e7cd69cd7ac1fdefedc79c24df64602a4f` |
| Formal route order | `be6f43dea7bc37892d634aed898438e86d2e8e71e110776c503405acee73640d` |
| R1 map closure | `05ca1d6f54763aecce80483f2c08be1c37e55cdf929c314619c3d0948e22463c` |
| R3 map closure | `dde14151685bc4b58d85847d5ac8fd24cf1af34a15ab01d69675dc8e8ddfdae8` |
| Original-frame hierarchy closure | `aedde1c200fef7a6f186f8326de3d370416ff367cc83530853dde184c5b37221` |

The observed `route_running` sequence matched the 600-entry candidate order exactly. Every question therefore contributes one R1 and one R3 observation to the paired analysis.

## 4. Direct-v1.2 workflow

```text
Question + five options + route-specific native map
                         |
                         v
Single Direct multimodal agent
  - reads and reasons over the map first
  - identifies unresolved answer-critical facts
  - decides whether original visual evidence is needed
  - chooses event-informed temporal locations
                         |
              +----------+----------+
              |                     |
              v                     v
      inspect_frames           final_answer
      <=3 new images           exactly A-E
              |
              v
Deterministic timestamp-to-original-frame resolution
              |
              v
Same Direct agent integrates map + prior observations + new observations
              |
              +---- reassess, inspect again or answer
```

The agent may answer from the map without requesting an image. Visual inspection is adaptive: 16 unique images is a hard ceiling, not a target. There is no padding and no silent clipping.

## 5. Dynamic visual budget and action contract

The available `inspect_frames` schema changes with the remaining unique-image budget:

```text
remaining >= 3  -> inspect_frames maxItems = 3
remaining = 2   -> inspect_frames maxItems = 2
remaining = 1   -> inspect_frames maxItems = 1
remaining = 0   -> inspect_frames unavailable; final_answer only
```

Duplicate physical frames do not consume the unique-image budget twice. If all requested timestamps resolve to previously seen physical frames, no image is retransmitted and the same logical conversation continues.

Provider responses must contain exactly one logical structured action. A first structurally invalid response is retained and billed, transports no invalid images and receives one correction opportunity:

- a tool-linked `tool_result` correction is used when one usable `tool_use_id` exists;
- an ordinary user correction message is used when no single usable tool identity exists.

A second invalid action terminates the route as `structural_action_correction_exhausted`. Provider transport retry and structural correction are independent mechanisms.

## 6. Durable formal execution

The validated lifecycle ordering was:

```text
route_running durable
-> request_start durable
-> provider request
-> attempt_end durable
-> ledger reconciliation
-> controller validation
-> controller_result durable
-> image/state update
-> repeat as needed
-> terminal artifact
-> atomic terminal route status
```

Formal journal reconciliation found:

| Event/accounting item | Count |
|---|---:|
| `route_running` | 600 |
| `request_start` | 3,320 |
| `attempt_end` | 3,320 |
| Ledger attempt IDs | 3,320 |
| Ledger reconciliations | 3,320 |
| Controller results | 3,321 |
| Terminal artifacts | 600 |
| Terminal statuses | 600 |
| Unmatched requests | 0 |
| Provider transport retries | 0 |
| Structural correction attempts | 17 |

The additional controller result is the synthetic `runtime_failure:turn_limit_exhausted` terminal decision after 32 completed provider turns; it does not represent an unrecorded provider request or charge.

## 7. Model, cache and pricing configuration

### 7.1 Provider configuration

| Setting | Value |
|---|---|
| Provider | Anthropic |
| Model | `claude-haiku-4-5-20251001` |
| Temperature | 0.0 |
| Maximum output tokens | 512 |
| Timeout | 120 seconds |
| Provider transport retry | 1 maximum |
| Structural correction retry | 1 maximum |
| Maximum turns | 32 |

### 7.2 Prompt caching

The stable cacheable request prefix was:

```text
Direct system instruction
-> verbatim native video map
-> Anthropic ephemeral cache boundary
-> question and options
-> route-specific conversation and images
```

Cache reuse was observed from provider telemetry but was never assumed for scientific correctness or prospective cost. Same-video R1 and R3 blocks use stable within-method prefixes; R1 and R3 do not share a map prefix because their native representations differ.

### 7.3 Frozen cache-aware pricing

| Token class | USD per million tokens |
|---|---:|
| Ordinary input | $1.00 |
| Five-minute cache creation | $1.25 |
| Cache read | $0.10 |
| Output | $5.00 |

Every actual attempt is included, including rejected structured actions and correction attempts. The older standard-rate-equivalent field is not authoritative.

## 8. Implementation and launch freeze

| Component | SHA-256 |
|---|---|
| Provider configuration | `581f7bb119660096820011bea6c03970c6af3391f3a2a3bff428377ff77f4c8a` |
| Prompt implementation | `6b574c6d8b5e549ea72138404decb2bd437ab646d2a36904a0128e6e9e711330` |
| Frozen system-prompt text | `6e938f9fc09af7d3ce94d75ee7c6d1588b053aeb79f6b0855137b581e1b3a22d` |
| Policy | `65a9f423167c324f47f78bc9467681b62ab9411fd8cc2c1768908ea7d76ecdba` |
| Controller | `abeaf63456078db862426690076eb9c0e90923f3fecc355261c2ec91316e118e` |
| Anthropic provider adapter | `dba180326d726dddaaf6880e4cca03575881aabd7b08622d53be39f81d9699cf` |
| Actions | `0ad965d5f0fe6f4278a864245a283d2ddeeb2532beec99bad727ef739ca63f24` |
| Action schema | `4ab64f5a27dcd9a5b6431b161479f89eeb077c757d9ec5beebc40312c626396f` |
| State machine | `ba92e9652cd9bb280c68b8b7b38a1aea23afcc3282019be2947f335ab7040a36` |
| Frame resolver | `080ceecb3a083012f838040009fafd5d2f72e7c6b2299d4f7e4aaa24b07f1b2e` |
| Pricing | `fd099f3b377392826d2ae0daa04f2b48069ca3faff2f45e2e53d6217fce588b3` |
| Formal runtime | `99074127803d44b0d316c9b5533d694e9082be5cd2211527d0d963525801b0ca` |
| Formal launch gate | `6774aae99265f627297475990d47be7e382b6f416201614c229ee633e7f55551` |
| Real formal runner | `8dbf727cad929d89dbb65dcc2613d51a15540d21e5538e8abf4b47f619a470b1` |
| Candidate builder | `4f8effe8635e8013615de61b563cbae0d5aa3330447bafafe5ac976d1a692515` |

The mandatory launch gate verified these fingerprints, the candidate SHA, launch-lock SHA, population, route order, input closures, scientific scalar values and namespace/ledger state before provider construction.

## 9. Metric and accounting definitions

### 9.1 Accuracy

```text
accuracy = exact A-E prediction matches / 300 frozen questions per method
```

Missing predictions and terminal failures remain in the denominator and count as incorrect. Completed-only accuracy is secondary and is not used as the headline result.

### 9.2 Completion

Completion means a valid durable A–E prediction is present. A durable terminal state without a prediction is incomplete.

### 9.3 Visual evidence

`unique_images` counts unique physical original frames transmitted to the provider. Duplicate requests that resolve to an already-seen frame do not increment this count.

`successful_inspection_rounds` counts accepted `inspect_frames` actions. Structurally rejected inspection requests are provider attempts and costs, but not successful scientific inspection rounds.

### 9.4 Provider attempts and cost

Actual provider totals include every completed request, including structurally rejected responses and the subsequent correction request. Cache-aware USD is calculated only from actual provider telemetry.

### 9.5 Latency

Provider/API model latency is the sum of provider-request latency within each route. Route wall time includes orchestration, frame resolution, serialization, journal writes and provider requests. No failure receives a zero-latency imputation.

## 10. Accuracy results

Wilson 95% confidence intervals are descriptive binomial intervals over the fixed denominator.

| Method | Correct | Denominator | Accuracy | Wilson 95% CI | Predictions |
|---|---:|---:|---:|---:|---:|
| R1 Direct | 88 | 300 | **29.333%** | 24.469–34.721% | 300 |
| R3 Direct | 103 | 300 | **34.333%** | 29.189–39.874% | 298 |

The full-population difference was:

```text
R3 - R1 = 34.333% - 29.333% = +5.000 percentage points
```

## 11. Paired R1/R3 accuracy

| Paired outcome | Questions |
|---|---:|
| Both R1 and R3 correct | 55 |
| R1 correct, R3 wrong | 33 |
| R1 wrong, R3 correct | 48 |
| Both wrong | 164 |
| **Total** | **300** |

The exact two-sided McNemar test on the 81 discordant pairs gives:

```text
p = 0.119274
```

A paired bootstrap with 100,000 question-level resamples, seed `20260904`, and a percentile interval gives:

```text
R3 - R1 accuracy difference 95% CI = [-0.667, +11.000] percentage points
```

No statistically significant paired accuracy difference was detected. This does **not** establish that R1 and R3 are equivalent.

## 12. Completion and terminal states

### 12.1 Fixed populations

| Method | Predictions | Missing | Completion | Terminal success | Terminal failed |
|---|---:|---:|---:|---:|---:|
| R1 | 300 | 0 | **100.000%** | 300 | 0 |
| R3 | 298 | 2 | **99.333%** | 298 | 2 |

### 12.2 Paired completion

| Paired outcome | Questions |
|---|---:|
| Both complete | 298 |
| R1 only complete | 2 |
| R3 only complete | 0 |
| Neither complete | 0 |

R3 minus R1 completion was `-0.667` percentage points. The exact two-sided McNemar result was `p = 0.5`; the paired percentile-bootstrap 95% interval was `[-1.667, 0.000]` percentage points.

## 13. Failure classification

Both failures occurred on R3 and remain incorrect in the fixed denominator.

### 13.1 Turn-limit exhaustion

| Field | Value |
|---|---|
| Question ID | `6fd90f8d-7a4d-425d-a812-3268db0b0342_11_29` |
| Video ID | `6fd90f8d-7a4d-425d-a812-3268db0b0342` |
| Durable category | `runtime_failure:turn_limit_exhausted` |
| Prediction | Missing |
| Provider attempts | 32 |
| Correction attempts | 0 |
| Successful inspection rounds | 32 |
| Unique images | 7 |
| Route API cost | $0.3518972 |
| Provider latency | 56.695 s |
| Route wall time | 62.571 s |

The final controller record is a synthetic turn-33 exhaustion decision with no additional provider request. The route repeatedly requested already-seen or otherwise non-budget-increasing evidence and reached the global 32-turn guard.

### 13.2 Structural correction exhaustion

| Field | Value |
|---|---|
| Question ID | `a6d45e95-8dc0-4932-83bf-ec53e265a16a_11_15` |
| Video ID | `a6d45e95-8dc0-4932-83bf-ec53e265a16a` |
| Durable category | `structural_action_correction_exhausted:frame_resolution:requested timestamp outside video duration: 1900.0` |
| Prediction | Missing |
| Provider attempts | 3 |
| Correction attempts | 1 |
| Successful inspection rounds | 1 |
| Unique images | 3 |
| Route API cost | $0.0133769 |
| Provider latency | 4.751 s |
| Route wall time | 5.019 s |

The original invalid action and correction attempt were both retained and billed. The correction again requested timestamp `1900.0` outside the frozen video duration, so the route terminated under the one-correction rule.

No failed route was replayed or repaired.

## 14. Visual evidence usage

### 14.1 Distribution summary

| Method | Total images | Mean | Median | Std. dev. | P25 | P75 | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 4,125 | 13.750 | 15 | 3.068 | 12 | 16 | 16 | 16 | 16 |
| R3 | 2,838 | 9.460 | 9 | 3.776 | 8 | 12 | 15 | 16 | 16 |

### 14.2 Image-count buckets

| Unique images per route | R1 | R3 |
|---|---:|---:|
| 0 | 0 | 2 |
| 1–3 | 0 | 24 |
| 4–6 | 8 | 34 |
| 7–9 | 59 | 151 |
| 10–12 | 18 | 22 |
| 13–15 | 83 | 43 |
| Exactly 16 | 132 | 24 |
| **Total** | **300** | **300** |

R1 reached the 16-image ceiling on `132/300 = 44.0%` of routes. R3 reached it on `24/300 = 8.0%`.

### 14.3 Successful inspection rounds

| Method | Total | Mean/question | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 1,566 | 5.220 | 6 | 6 | 7 | 12 |
| R3 | 1,138 | 3.793 | 3 | 6 | 7 | 32 |

The R3 maximum of 32 belongs to the turn-limit failure and does not violate the image budget: repeated requests need not transmit a new unique physical frame.

### 14.4 Paired visual difference

| Metric, R3 minus R1 | Mean difference | Paired bootstrap 95% CI | Relative reduction in aggregate |
|---|---:|---:|---:|
| Unique images/question | **-4.290** | -4.767 to -3.813 | **31.200%** |
| Successful inspection rounds/question | **-1.427** | -1.723 to -1.103 | **27.331%** |

All 300 paired questions contribute to these resource comparisons, including the two failed R3 routes with their actual observed resource usage.

## 15. Provider requests, tokens and measured API cost

### 15.1 Authoritative totals including all attempts

| Metric | R1 | R3 | Total |
|---|---:|---:|---:|
| Provider attempts | 1,876 | 1,444 | 3,320 |
| Structural corrections | 10 | 7 | 17 |
| Provider transport retries | 0 | 0 | 0 |
| Ordinary input tokens | 10,223,155 | 6,188,889 | 16,412,044 |
| Cache creation tokens | 918,876 | 1,415,926 | 2,334,802 |
| Cache read tokens | 35,104,218 | 43,169,630 | 78,273,848 |
| Output tokens | 177,925 | 144,978 | 322,903 |
| Cache-aware API USD | **$15.7717968** | **$13.0006495** | **$28.7724463** |

### 15.2 Per-question cost

| Method | Attempts/question | Mean USD/question | Median | P90 | P95 | Maximum |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 6.253 | $0.052573 | $0.054961 | $0.076446 | $0.094832 | $0.179688 |
| R3 | 4.813 | $0.043335 | $0.029869 | $0.087624 | $0.112930 | $0.351897 |

R3's higher P90/P95 and maximum coexist with a lower mean and median. The long upper tail is influenced by cache creation and the 32-turn failure, while most R3 routes use fewer attempts than R1.

### 15.3 R3 relative to R1

| Metric | R3 minus R1 | Percentage change |
|---|---:|---:|
| Provider attempts | -432 | **-23.028%** |
| Ordinary input tokens | -4,034,266 | -39.462% |
| Cache creation tokens | +497,050 | +54.093% |
| Cache read tokens | +8,065,412 | +22.975% |
| Output tokens | -32,947 | -18.518% |
| API USD | -$2.7711473 | **-17.570%** |

The semantic R3 prefix produces more cache creation and cache-read tokens, but fewer requests and substantially fewer ordinary input/output tokens reduce its total measured spend.

### 15.4 Paired request and cost differences

| Metric, R3 minus R1 | Mean paired difference | Paired bootstrap 95% CI |
|---|---:|---:|
| Provider attempts/question | **-1.440** | -1.730 to -1.123 |
| Ordinary input tokens/question | **-13,447.553** | -15,876.951 to -10,860.105 |
| Cache creation tokens/question | **+1,656.833** | +108.660 to +3,314.046 |
| Cache read tokens/question | **+26,884.707** | +17,667.904 to +36,909.959 |
| Output tokens/question | **-109.823** | -135.177 to -82.337 |
| API USD/question | **-$0.009237** | -$0.013485 to -$0.004685 |

These intervals are descriptive paired bootstrap intervals, not a family of multiplicity-adjusted hypothesis tests.

## 16. Prompt-cache behaviour

| Metric | R1 | R3 |
|---|---:|---:|
| Cache creation events | 47 | 46 |
| Cache-read events | 1,829 | 1,398 |
| Cache creation tokens | 918,876 | 1,415,926 |
| Cache read tokens | 35,104,218 | 43,169,630 |
| Cache-read share of input-related tokens | 75.907% | 85.022% |

The share denominator is ordinary input plus cache creation plus cache-read tokens. Provider telemetry demonstrates substantial cache reuse, but not that every same-video request was warm. Formal cost uses observed token classes only.

Cost components were:

| Component | R1 | R3 | Total |
|---|---:|---:|---:|
| Ordinary input | $10.2231550 | $6.1888890 | $16.4120440 |
| Cache creation | $1.1485950 | $1.7699075 | $2.9185025 |
| Cache read | $3.5104218 | $4.3169630 | $7.8273848 |
| Output | $0.8896250 | $0.7248900 | $1.6145150 |
| **Total** | **$15.7717968** | **$13.0006495** | **$28.7724463** |

## 17. Latency

### 17.1 Provider/API model latency

| Method | n | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 300 | 13.255 s | 13.681 s | 16.925 s | 18.070 s | 27.224 s |
| R3 | 300 | 10.451 s | 9.345 s | 15.974 s | 17.130 s | 56.695 s |

### 17.2 Route wall-clock latency

| Method | n | Mean | Median | P90 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 300 | 14.190 s | 14.724 s | 18.053 s | 19.221 s | 28.044 s |
| R3 | 300 | 11.175 s | 9.956 s | 17.176 s | 18.355 s | 62.571 s |

R3 minus R1 mean route wall time was `-3.014` seconds, a `21.243%` reduction. The paired bootstrap 95% interval was `[-3.637, -2.358]` seconds. The corresponding mean provider-latency difference was `-2.804` seconds, with interval `[-3.383, -2.196]`.

The observed sequential formal interval was approximately 2 hours 7 minutes, from the first durable route event at 20:29 UTC to completion at 22:36 UTC on 2026-09-04. This interval is descriptive and includes orchestration overhead.

## 18. Compact cost-effectiveness comparison

| Metric | R1 Direct | R3 Direct | R3 relative to R1 |
|---|---:|---:|---:|
| Accuracy | 29.333% | 34.333% | +5.000 pp |
| Prediction completion | 100.000% | 99.333% | -0.667 pp |
| Images/question | 13.750 | 9.460 | -31.200% |
| Inspection rounds/question | 5.220 | 3.793 | -27.331% |
| Provider attempts/question | 6.253 | 4.813 | -23.028% |
| API cost/question | $0.052573 | $0.043335 | -17.570% |
| Mean route wall time | 14.190 s | 11.175 s | -21.243% |

Within this controlled Direct experiment, R3 had a numerically higher accuracy and lower mean visual/request/cost/latency use. The paired accuracy result was not statistically significant, whereas the paired bootstrap intervals for the reported mean resource differences excluded zero. This supports a conservative resource-efficiency observation, not a claim that R3 is universally superior or accuracy-equivalent.

## 19. Exploratory question-type analysis

HourVideo supplies task labels independently of these predictions, so no post-hoc subjective categories were created. Selected categories with at least 15 Eval300 questions are shown below.

| Authoritative task label | n | R1 accuracy | R3 accuracy | R1 images/q | R3 images/q |
|---|---:|---:|---:|---:|---:|
| Perception: factual recall | 63 | 31.75% | 49.21% | 13.90 | 8.43 |
| Perception: sequence recall | 22 | 18.18% | 31.82% | 13.77 | 9.36 |
| Spatial proximity | 23 | 13.04% | 13.04% | 13.13 | 11.04 |
| Spatial relationship | 46 | 17.39% | 19.57% | 13.96 | 10.61 |
| Temporal duration | 31 | 41.94% | 41.94% | 11.61 | 7.32 |
| Temporal frequency | 37 | 27.03% | 27.03% | 14.24 | 10.24 |
| Temporal prerequisites | 15 | 60.00% | 53.33% | 15.33 | 10.80 |

These category results are exploratory. Several remaining categories have very small sample sizes, and no multiple-comparison correction was applied. The primary inference remains the pre-specified full paired Eval300 comparison.

## 20. Canonical reconstruction and validation

Canonical scoring was performed only after the no-gold structural audit passed. The authoritative HourVideo gold was then used only to derive correctness fields and was not written into raw route artifacts.

| Check | Result |
|---|---|
| Canonical route rows | 600/600 |
| Unique `(question_id, method)` identities | 600/600 |
| R1 identities | 300 |
| R3 identities | 300 |
| Paired question IDs | 300 |
| Fixed-denominator totals | PASS |
| Paired correctness table sums to 300 | PASS |
| Paired completion table sums to 300 | PASS |
| Request-start/attempt-end/ledger pairing | PASS |
| Visual 3/16 constraints | PASS |
| Total API spend reproduction | PASS |
| Gold loaded before structural validation | No |
| Raw artifacts modified by aggregation | No |
| API/model calls during aggregation | 0 / 0 |
| Final canonical validation | **PASS** |

Paired bootstrap settings:

```text
seed = 20260904
iterations = 100,000
resampling unit = question ID pair
interval = 2.5th/97.5th percentile
```

## 21. Reproducibility SHA

| Artifact | SHA-256 |
|---|---|
| Candidate manifest v3 | `c73aeaf41d5bf284ade54198525d8396bf3a9fa5e45587aedd49dd3744dc557d` |
| Launch lock v3 | `ebd2014fcd952779444064510d39147149c174bc1ff2bc7fa231dfd589c09d5c` |
| Immutable raw closure, 1,507 files | `a83e53138ff945efb96ba87b1e80cb9c8ed6dc35449898514b9ccab869c37d29` |
| Authoritative HourVideo gold | `e1af087df035d524ee64d92d34d2e81f29461fdeb5fa72cfda679d7c9909daf3` |
| Canonical structural validation | `d1cc7c08ec701f22612e81866a125fa553be09f66e004b10fe0bdd2505674fc8` |
| Canonical accuracy | `753a1fdd79a253f91d8fcd9437c0dab1ac2dc01ad583ed85992fa852c3047e57` |
| Canonical completion | `c6909106bfbbb0d743dad7fda08ea30192338a335c068a785d4deafe5782316c` |
| Paired accuracy | `3b16bd09551d0d2bec27b4d280adbede4ec9b823c184c1fcac235bd89638fd9a` |
| Visual efficiency | `63a168b78c78db4ac9a89af6c6362d4c21958ad67f0806ff33d7c8c52f367682` |
| API cost | `15384c17fc895aa6bd7ae9ba1874503443df2990aab9c4e7cb2e2e0db9db473f` |
| Cache summary | `20b51d807e37b03aeb0dd740d0a748898054bdbbc7d392bae8ef0cffa33c44d1` |
| Latency | `5cd6a344a0a26872255a1d8fd63649d50141739b8f12b13935c048b4790b2c81` |
| Paired efficiency | `e3f79c3dd1b41be949d662307f7c38e6a2e9b14e442d990d683d6874fc1c0669` |
| Failure classification | `5b4a6d0f553672ebbf6a6980343db3aa9dced727a6856a601c9396b14fff0b41` |
| Canonical route CSV | `35516eb63fffbcb91e6fb93a3d08fcf6470a83c60ab51bf07d4cd2de32f59ad6` |
| Canonical final validation | `f823fe2104b428b890f02597e2cc5edd40c194a3763b15f783ef86a90175f433` |

The exact raw closure hash was recomputed before and after canonical scoring and remained unchanged.

## 22. Interpretation for the thesis

The headline fixed-denominator results are:

- R1 Direct: `88/300 = 29.33%`;
- R3 Direct: `103/300 = 34.33%`;
- R3 minus R1: `+5.00` percentage points;
- exact paired McNemar: `p = 0.119274`;
- paired-bootstrap difference interval: `[-0.67, +11.00]` percentage points.

The experiment therefore does not detect a statistically significant paired accuracy difference. The numerical direction favours R3, but the uncertainty interval includes zero and must be reported.

The resource results consistently favour R3 on the same 300 questions:

- 31.20% fewer unique original-frame transmissions;
- 27.33% fewer successful inspection rounds;
- 23.03% fewer provider attempts;
- 17.57% lower measured API spend;
- 21.24% lower mean route wall time;
- far fewer routes reaching the 16-image ceiling: 24 for R3 versus 132 for R1.

These observations are consistent with the R3 semantic hierarchy providing a more economical navigation substrate for this Direct agent. However, the controlled factor is the complete native representation, including its semantic content and size; the experiment does not isolate which property of R3 produces the difference.

The Direct-v1.2 result should also be distinguished from the earlier local staged pipeline. Direct uses a stronger external multimodal model and collapses Planner/Shared/Fine/Final into one agent, so differences between the Direct and Local experiments combine model and workflow changes. They are not a clean causal estimate of model scaling alone.

## 23. Thesis-ready concise statement

> Direct-v1.2 compared a structural/ASR-oriented native navigation map (R1) with a hierarchical semantic native map (R3) using the same Claude Haiku 4.5 multimodal agent, prompt, controller, frame resolver, retry rules and adaptive visual policy on 300 HourVideo questions. The agent could request at most three new physical frames per inspection round and sixteen unique frames per question, with failed routes retained in the fixed denominator. R1 produced 300 predictions and answered 88 questions correctly (29.33%, Wilson 95% CI 24.47–34.72%); R3 produced 298 predictions and answered 103 correctly (34.33%, 29.19–39.87%). The paired difference was +5.00 percentage points for R3, but no statistically significant paired difference was detected by exact McNemar testing (`p=0.119`; paired-bootstrap 95% CI -0.67 to +11.00 percentage points), which does not establish equivalence. R3 used 31.20% fewer unique images, 27.33% fewer successful inspection rounds, 23.03% fewer provider attempts, 17.57% less measured API spend and 21.24% less mean route wall time. R1 reached the 16-image ceiling on 132 routes, compared with 24 for R3. The complete run made 3,320 provider attempts and cost $28.7724463 under observed cache-aware telemetry. Two R3 routes ended in durable failure—one at the 32-turn guard and one after structural correction exhaustion—and both counted as incorrect.

## 24. Evidence locations

- Authoritative candidate: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/formal_manifest_final_candidate_no_api_v3.json`
- Launch lock: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/formal_launch_lock_v3.json`
- Raw journals: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/journals`
- Raw route statuses: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/route_status`
- Raw route artifacts: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/route_artifacts`
- Canonical summary: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/canonical_summary_v1`
- Canonical manifest: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/canonical_summary_v1/canonical_manifest.json`
- Route-level canonical table: `outputs/direct_v1_formal/direct_v1_2_3x16_r1_r3_eval300_formal_v1/canonical_summary_v1/route_level_canonical.csv`

## 25. Exclusions and limitations

Not included in the reported external API cost:

- offline R1/R3 map construction;
- source-video and frame storage;
- local orchestration CPU, storage and filesystem cost;
- network infrastructure cost outside provider billing;
- pilot, smoke or fake-provider runs;
- GenS or VideoSEAL;
- any later scoring or thesis-writing computation.

The following limitations should accompany thesis claims:

1. The paired accuracy difference is not statistically significant; absence of detection is not evidence of equivalence.
2. Both methods use one frozen model and Direct prompt. Results need not generalise to other models, prompts or visual budgets.
3. R1 versus R3 changes the complete native map representation, including content and prefix length; individual representation features are not separately identified.
4. Prompt-cache behaviour is observed rather than guaranteed. R1 and R3 have different native map sizes, so cache token classes differ even though cache policy is controlled.
5. The two R3 failures remain in the fixed denominator. Completed-only accuracy must not replace the primary fixed-denominator result.
6. Absolute latency depends on provider and network conditions during this run and should not support hardware-independent speed claims against the Local pipeline or other systems.
7. Question-type results are exploratory; some categories are small and no multiple-comparison correction was applied.
8. Measured API USD covers actual provider telemetry only and excludes offline indexing and infrastructure costs.
9. The Direct and Local experiments differ in both workflow and model stack; cross-experiment accuracy differences are descriptive rather than a clean causal comparison.
