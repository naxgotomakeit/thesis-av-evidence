# 226 Direct-v1.2 open-ended V/AV-Speech six-question summary

Status: `ROUTES_FROZEN`; completed routes: `8/12`.

All 12 route attempts were executed in the frozen order. Four routes terminated after the one allowed structural correction because the provider returned a `reason` longer than the frozen 240-character tool contract. They were not repaired or replayed.

Raw ASR was not supplied to the Direct answerer. The same semantic-only projection set `exact_source_asr=[]` in both maps; all Organizer-generated structure, summaries, uncertainty notes, and exact visual captions were retained.

| Question | V frames | AV frames | Jaccard | Navigation | Answer comparison |
|---|---:|---:|---:|---|---|
| q_global_summary | 3 | 0 | 0.000 | MAJOR_NAVIGATION_CHANGE | NOT_COMPARABLE_BOTH_ROUTE_FAILURE |
| q_weapon_visible | 3 | 3 | 1.000 | SAME_NAVIGATION | DIFFERENT_WORDING_SAME_SUBSTANCE |
| q_visible_injury | 3 | 3 | 0.500 | MINOR_NAVIGATION_CHANGE | SUBSTANTIVE_ANSWER_CHANGE |
| q_medical_assistance | 3 | 5 | 0.143 | MAJOR_NAVIGATION_CHANGE | NOT_COMPARABLE_ROUTE_FAILURE |
| q_handcuffing | 9 | 9 | 0.800 | MINOR_NAVIGATION_CHANGE | DIFFERENT_WORDING_SAME_SUBSTANCE |
| q_handcuff_before_medical | 12 | 6 | 0.500 | MAJOR_NAVIGATION_CHANGE | NOT_COMPARABLE_ROUTE_FAILURE |

No gold/correctness was read or evaluated. Human stage annotations were used only after all 12 route attempts were frozen.
