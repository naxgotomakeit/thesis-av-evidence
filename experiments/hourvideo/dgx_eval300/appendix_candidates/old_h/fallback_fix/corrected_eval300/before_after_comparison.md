# Eval300 fallback correction: before/after

The corrected view replaces 13 selected attempts one-for-one. Original attempts and their costs remain preserved but are not added to corrected method efficiency.

| Profile | Completed before→after | Correct before→after | Timeout before→after | Strict invalid before→after |
|---|---:|---:|---:|---:|
| H6 | 280→279 | 64→61 | 18→18 | 2→3 |
| H15 | 287→286 | 64→63 | 11→12 | 2→2 |
| H30 | 257→257 | 66→66 | 42→42 | 1→1 |

The invalid first H-15 acceptance is diagnostic-only and is not selected. The accepted H-15 run is `20260826T164103Z-ge2b`.
