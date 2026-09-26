# ABD strict-support disagreement statistics

Strict support is derived only as `label == supported`; all original five-class labels are preserved.
No label was merged or replaced. All percentage-point effects use the fixed arm denominator 300 and
mean only: if labels in this comparison subset were substituted while every other record stayed fixed.

## Direct answers

- Of the 31 five-class disagreements, 17 remain non-supported in all three rounds and never cross the strict-supported boundary.
- 14 records cross the supported boundary at least once.
- Of the 10 records with three different five-class labels, 1 remains non-supported in all three rounds and 9 cross the boundary.
- Of the 20 Sol review-flag records, 12 always remain non-supported and 8 cross the boundary. A review flag is not itself a boundary change.

## Overall 2x2 comparisons

| comparison | n | S→S | S→non-S | non-S→S | non-S→non-S | binary agreement | net supported | net pp/300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| controls60_original_abd_vs_correction | 60 | 3 | 8 | 4 | 45 | 48 (80.0%) | -4 | -1.333 |
| disagreements31_original_abd_vs_sol | 31 | 1 | 7 | 4 | 19 | 20 (64.5%) | -3 | -1.000 |
| disagreements31_correction_vs_sol | 31 | 2 | 2 | 3 | 24 | 26 (83.9%) | +1 | +0.333 |

## Arm-specific hypothetical effects

| comparison | arm | reviewed | S→non-S | non-S→S | crossings | net supported | net pp/300 |
|---|---|---:|---:|---:|---:|---:|---:|
| controls60_original_abd_vs_correction | A | 16 | 1 | 0 | 1 | -1 | -0.333 |
| controls60_original_abd_vs_correction | B | 13 | 1 | 0 | 1 | -1 | -0.333 |
| controls60_original_abd_vs_correction | D | 31 | 6 | 4 | 10 | -2 | -0.667 |
| disagreements31_original_abd_vs_sol | A | 6 | 0 | 1 | 1 | +1 | +0.333 |
| disagreements31_original_abd_vs_sol | B | 8 | 1 | 0 | 1 | -1 | -0.333 |
| disagreements31_original_abd_vs_sol | D | 17 | 6 | 3 | 9 | -3 | -1.000 |
| disagreements31_correction_vs_sol | A | 6 | 0 | 2 | 2 | +2 | +0.667 |
| disagreements31_correction_vs_sol | B | 8 | 0 | 0 | 0 | +0 | +0.000 |
| disagreements31_correction_vs_sol | D | 17 | 2 | 1 | 3 | -1 | -0.333 |

## Boundary-crossing records among the 31

| ID | question_id | arm | original ABD | correction | Sol | ABD S? | correction S? | Sol S? | Sol review flag |
|---|---|---|---|---|---|---:|---:|---:|---:|
| E0061 | db3f7933-dfa0-4678-9d4f-393b628ded45_11_2 | D | supported | contradicted | partially_supported | 1 | 0 | 0 | 1 |
| E0269 | 7e512589-aa97-41e8-83d3-af2e83e4fd06_3_26 | D | supported | partially_supported | partially_supported | 1 | 0 | 0 | 1 |
| E0021 | 7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b_8_18 | D | supported | contradicted | partially_supported | 1 | 0 | 0 | 1 |
| E0113 | 6fd90f8d-7a4d-425d-a812-3268db0b0342_15_2 | D | supported | unsupported | unsupported | 1 | 0 | 0 | 1 |
| E0330 | a6d45e95-8dc0-4932-83bf-ec53e265a16a_7_7 | D | partially_supported | contradicted | supported | 0 | 0 | 1 | 0 |
| E0756 | 7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b_4_2 | D | partially_supported | supported | unsupported | 0 | 1 | 0 | 1 |
| E0705 | 6fd90f8d-7a4d-425d-a812-3268db0b0342_4_1 | D | unsupported | supported | supported | 0 | 1 | 1 | 0 |
| E0666 | db3f7933-dfa0-4678-9d4f-393b628ded45_17_1 | D | partially_supported | supported | supported | 0 | 1 | 1 | 0 |
| E0657 | 41a86310-2cc1-48f9-b5b5-6b495a95fbac_8_11 | D | supported | partially_supported | unsupported | 1 | 0 | 0 | 1 |
| E0122 | 70f2a750-f403-41b8-aabb-480eb3ab4ed4_17_8 | B | supported | unsupported | partially_supported | 1 | 0 | 0 | 1 |
| E0311 | 7ddbf8a2-5b3e-44cd-b9dc-db17ec06831b_4_19 | D | partially_supported | supported | contradicted | 0 | 1 | 0 | 0 |
| E0072 | 71fbc5bf-7e2a-415d-86bc-3a948742e904_4_12 | A | partially_supported | unsupported | supported | 0 | 0 | 1 | 0 |
| E0448 | d3a0899e-2093-454c-9f65-30087883193a_8_21 | A | supported | partially_supported | supported | 1 | 0 | 1 | 0 |
| E0112 | a6d45e95-8dc0-4932-83bf-ec53e265a16a_5_26 | D | supported | contradicted | partially_supported | 1 | 0 | 0 | 1 |

## Interpretation boundary

The 31 records were selected because their first two five-class labels disagreed. Within them, 17/31 disagreements are internal to the non-supported categories, while 14/31 cross strict support at least once. This describes the selected subset only and is not a 900-record error estimate or a validation of the 116 correction targets.
