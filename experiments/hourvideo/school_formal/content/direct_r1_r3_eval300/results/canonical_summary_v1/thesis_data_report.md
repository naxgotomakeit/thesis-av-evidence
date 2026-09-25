# Direct-v1.2 R1/R3 Eval300 canonical result

This report scores the immutable 600-route Direct-v1.2 3/16 raw closure against the authoritative frozen HourVideo annotations. Accuracy uses the fixed denominator of 300 questions per method; failed or missing predictions are incorrect. Gold was loaded only after the no-gold structural validation passed.

## Thesis-ready main table

| Metric | R1 Direct | R3 Direct |
|---|---:|---:|
| Correct / 300 | 88 / 300 | 103 / 300 |
| Accuracy | 29.33% | 34.33% |
| Wilson 95% CI | [24.47%, 34.72%] | [29.19%, 39.87%] |
| Predictions / 300 | 300 / 300 | 298 / 300 |
| Images / question | 13.75 | 9.46 |
| Inspection rounds / question | 5.22 | 3.79 |
| Provider attempts / question | 6.253 | 4.813 |
| API cost / question | $0.05257 | $0.04334 |
| Mean route wall latency | 14.19 s | 11.18 s |
| Total API cost | $15.7717968 | $13.0006495 |

## Paired accuracy and completion

The paired correctness table was: both correct 55; R1 correct/R3 wrong 33; R1 wrong/R3 correct 48; both wrong 164. R3 minus R1 was 5.00 percentage points (100,000-resample paired percentile bootstrap 95% CI [-0.67, 11.00] pp). The exact two-sided McNemar p-value was 0.119274. No statistically significant paired difference was detected; this does not establish equivalence.

Completion was 300/300 for R1 and 298/300 for R3. The completion table was: both complete 298; R1-only 2; R3-only 0; neither 0. The exact two-sided McNemar p-value was 0.5.

## Resource comparison

Relative to R1, R3 used 1287 fewer unique images (31.20% reduction), 428 fewer successful inspection rounds (27.33% reduction), 432 fewer provider attempts (23.03% reduction), and $2.7711473 less measured API spend (17.57% reduction). Mean route wall time changed by -3.01 s (-21.24%).

The main paired mean-difference bootstrap intervals (R3 minus R1) were: images [-4.767, -3.813]; provider attempts [-1.730, -1.123]; API USD [-0.01348, -0.00468]; route wall latency [-3.637, -2.358] seconds.

## Visual usage

R1 transmitted 4125 unique physical images (median 15); 132/300 routes reached the 16-image ceiling. R3 transmitted 2838 images (median 9); 24/300 reached the ceiling. No route exceeded 16 unique images and no accepted turn transported more than three new images.

## Cache-aware cost

Measured formal spend was $28.7724463. The authoritative rates were $1.00/M ordinary input, $1.25/M five-minute cache creation, $0.10/M cache read, and $5.00/M output. Provider telemetry recorded 35,104,218 R1 and 43,169,630 R3 cache-read tokens; no assumption was made that every same-video request was warm.

## Durable failures

Two R3 routes failed and were scored incorrect: one `runtime_failure:turn_limit_exhausted`, and one `structural_action_correction_exhausted:frame_resolution:requested timestamp outside video duration: 1900.0`. R1 had no terminal failures. No failed route was retried after termination.

### Thesis-ready key findings

1. R1 achieved 29.33% accuracy and R3 achieved 34.33% on the same fixed 300-question denominator; the paired difference was 5.00 percentage points. No statistically significant paired difference was detected; this does not establish equivalence.
2. R3 used 31.20% fewer unique visual frames than R1 (9.46 versus 13.75 per question).
3. R3 used 27.33% fewer successful inspection rounds.
4. R3 made 23.03% fewer provider attempts.
5. R3 cost 17.57% less in measured external API USD ($13.0006495 versus $15.7717968).
6. R3 mean route wall time was 21.24% lower (11.18 versus 14.19 seconds).
7. The 16-image ceiling was reached by 132/300 R1 routes and 24/300 R3 routes.
8. The only final failures were the two durable R3 failures described above; both remained incorrect in the fixed denominator.

Interpretation is associational within this controlled comparison: the online Direct agent, controller, model, and budget were held fixed, while the intended method difference was the native R1 versus R3 map representation.
