# R1/R3-v2 map-aware all-Medium retrieval pair v1

This experiment compares R1 structured-fallback retrieval with R3-v2 canonical-caption retrieval under one shared Planner and ranking policy. Each Planner sees its rung's navigation map, but Coarse suggestions are audit-only soft hints. Every query ranks all 30 Medium nodes, Coarse prior weight is zero, and Fine reranking is disabled.

The only planned differences are the map and Medium lexical representation. Organizer, Sufficiency, temporal review, Final Gemini, answer generation, QA, and HourVideo are outside scope.

Preflight and live run:

```powershell
C:\Users\72977\miniforge3\envs\thesis_av\python.exe scripts/experiments/run_r1_r3_v2_map_aware_all_medium_retrieval_pair_v1.py --preflight
C:\Users\72977\miniforge3\envs\thesis_av\python.exe scripts/experiments/run_r1_r3_v2_map_aware_all_medium_retrieval_pair_v1.py
```
